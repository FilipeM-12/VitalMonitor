"""alarmes_e_estavel.py — dois submodelos, um do que é ESTÁVEL e outro do que é ALARME, ajudam o modelo principal?

Submodelos (treinados com previsões FORA DA AMOSTRA no desenvolvimento, 3 dobras por doente, para o modelo principal não ver pontuações de um submodelo que
treinou nos mesmos doentes; no teste usam-se os submodelos treinados em todo o desenvolvimento):
  sub_estavel  deteção de anomalias (Isolation Forest) treinada SÓ nos blocos estáveis (|Δ a 5 min| < D/2 = 7,5 mmHg), sobre as características normalizadas
               (sem idade e sexo; em falta passa a 0, a média do z-score). Saída = quão anómalo é o bloco face ao que é estável (maior = mais anómalo).
               Não depende da direção.
  sub_alarme   regressor (XGBoost) treinado SÓ nos blocos que são evento da direção (descida: Δ <= -15; subida: Δ >= +15), a prever o Δ. Saída = o Δ que o bloco
               teria SE fosse um evento. Fora dos eventos extrapola, por isso é só uma característica extra, não um detetor.
Modelo principal: o XGBoost quantílico das 43 características (hiperparâmetros de 01_optuna_tempo.py), com e sem as pontuações:
  base | +estavel | +alarme | +ambos
Mede no TESTE o AUC e a sensibilidade por episódio a 4-10 min/h (fronteira por direção, como em sensibilidade_tempo.py) e o IC 95 % da diferença face à base
por bootstrap de DOENTES. Não se reafina nada: é a comparação a hiperparâmetros fixos.
Uso: python3 alarmes_e_estavel.py     (dados em ./dados)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

AQUI = os.path.dirname(os.path.abspath(__file__))
TF = os.path.normpath(os.path.join(AQUI, "..", "..", "3_tendencia_modelo_final", "tendencia_tempo_fixo"))        # tempo_fixo.py e os best_params_*.json só existem em tendencia_tempo_fixo/
sys.path.insert(0, TF)
import tempo_fixo as tf  # noqa: E402

DADOS = os.environ.get("DADOS", f"{AQUI}/dados")
NJOBS = int(os.environ.get("NJOBS", "14"))
D = 15
B = 1000
BLOCO_SEG, GAP_MERGE, PRE_CREDITO, DUTY_MAX = 15, 4, 8, 0.30
ALVOS = np.array([4, 6, 8, 10])
TAU = {'descida': 0.10, 'subida': 0.90}
SINAL = {'descida': -1.0, 'subida': 1.0}

dev, V = tf.preparar_grade(DADOS, 'dev')
hold, _ = tf.preparar_grade(DADOS, 'holdout')
for d in (dev, hold):
    d['subject_id'] = d.subject_id.astype(str)
FEATS = V['tudo']
Z = [c for c in FEATS if c not in ('age', 'gender_bin')]
tem_d, tem_h = dev.delta.notna().values, hold.delta.notna().values
print(f"dev {int(tem_d.sum())} blocos, teste {int(tem_h.sum())} blocos", flush=True)
t0 = time.time()


def isolation(treino_idx, aplicar_idx, frame_t, frame_a):
    f = IsolationForest(n_estimators=100, max_samples=256, random_state=0, n_jobs=NJOBS)
    f.fit(np.nan_to_num(frame_t.loc[treino_idx, Z].values, nan=0.0))
    return -f.score_samples(np.nan_to_num(frame_a.loc[aplicar_idx, Z].values, nan=0.0))


def regressor_alarme(direcao, treino_idx, aplicar_idx, frame_t, frame_a):
    ev = treino_idx[(frame_t.loc[treino_idx, f'y_{direcao}'] == 1).values]
    m = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, n_jobs=NJOBS, tree_method='hist')
    m.fit(frame_t.loc[ev, FEATS].values, frame_t.loc[ev, 'delta'].values)
    return m.predict(frame_a.loc[aplicar_idx, FEATS].values)


# ── pontuações dos submodelos: fora da amostra no desenvolvimento, treinadas em todo o desenvolvimento para o teste ──
idx_d, idx_h = dev.index[tem_d], hold.index[tem_h]
estavel_d = dev.index[tem_d & (dev.delta.abs() < D / 2).values]
dev['sub_estavel'], hold['sub_estavel'] = np.nan, np.nan
for d_ in ('descida', 'subida'):
    dev[f'sub_alarme_{d_}'], hold[f'sub_alarme_{d_}'] = np.nan, np.nan
grupos = dev.loc[idx_d, 'subject_id'].values
for k, (tr, va) in enumerate(GroupKFold(n_splits=3).split(idx_d, groups=grupos)):
    tr_i, va_i = idx_d[tr], idx_d[va]
    est_tr = np.intersect1d(tr_i, estavel_d)
    dev.loc[va_i, 'sub_estavel'] = isolation(est_tr, va_i, dev, dev)
    for d_ in ('descida', 'subida'):
        dev.loc[va_i, f'sub_alarme_{d_}'] = regressor_alarme(d_, tr_i, va_i, dev, dev)
    print(f"  submodelos dobra {k + 1}/3 ({time.time() - t0:.0f} s)", flush=True)
hold.loc[idx_h, 'sub_estavel'] = isolation(estavel_d, idx_h, dev, hold)
for d_ in ('descida', 'subida'):
    hold.loc[idx_h, f'sub_alarme_{d_}'] = regressor_alarme(d_, idx_d, idx_h, dev, hold)
print(f"submodelos prontos ({time.time() - t0:.0f} s)", flush=True)
for d_ in ('descida', 'subida'):                                         # separação que o submodelo de alarme dá por si só (teste)
    y = hold.loc[idx_h, f'y_{d_}'].values
    print(f"  AUC isolado no teste, {d_}: sub_estavel {roc_auc_score(y, hold.loc[idx_h, 'sub_estavel']):.4f} | "
          f"sub_alarme {roc_auc_score(y, SINAL[d_] * hold.loc[idx_h, f'sub_alarme_{d_}']):.4f}", flush=True)


def runs(mask, gap=GAP_MERGE):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    cortes = np.where(np.diff(idx) > gap + 1)[0]
    return [(g[0], g[-1]) for g in np.split(idx, cortes + 1)]


def contagens(y, alarme, eventos):
    """det, n_ev, min de falso alarme, blocos em alarme, blocos, n.º de alarmes, n.º de alarmes falsos (sem episódio)."""
    al = runs(alarme)
    toca = lambda a0, a1: any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (e0, e1) in eventos)
    det = sum(any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (a0, a1) in al) for (e0, e1) in eventos)
    falsos = [(a0, a1) for (a0, a1) in al if not toca(a0, a1)]
    fa = sum((a1 - a0 + 1) * BLOCO_SEG / 60.0 for (a0, a1) in falsos)
    return det, len(eventos), fa, sum(a1 - a0 + 1 for (a0, a1) in al), len(y), len(al), len(falsos)


def fronteira(soma):
    """soma (T, 7) -> (len(ALVOS), 4): sensibilidade, VPP, alarmes/h e falsos alarmes/h interpolados ao longo da fronteira, a ALVOS min/h de falso alarme
    (ciclo de serviço <= 30 %). O VPP é a fração dos ALARMES (e não dos blocos) que tocam um episódio."""
    vazio = np.full((len(ALVOS), 4), np.nan)
    horas = soma[:, 4] * BLOCO_SEG / 3600.0
    sens = soma[:, 0] / np.maximum(soma[:, 1], 1)
    fa = soma[:, 2] / horas
    vpp = 1 - soma[:, 6] / np.maximum(soma[:, 5], 1)
    al_h, fal_h = soma[:, 5] / horas, soma[:, 6] / horas
    ok = (soma[:, 3] / soma[:, 4]) <= DUTY_MAX
    if ok.sum() < 2:
        return vazio
    s, f, v, ah, fh = sens[ok], fa[ok], vpp[ok], al_h[ok], fal_h[ok]
    o = np.argsort(f, kind='stable'); s, f, v, ah, fh = s[o], f[o], v[o], ah[o], fh[o]
    melhor = np.maximum.accumulate(s)
    m = np.concatenate([[True], s[1:] > melhor[:-1]]); s, f, v, ah, fh = s[m], f[m], v[m], ah[m], fh[m]
    if len(s) < 2:
        return vazio
    return np.column_stack([np.interp(ALVOS, f, x, left=np.nan, right=x[-1]) for x in (s, v, ah, fh)])


def por_doente(score, direcao):
    """(n_doentes, T, 5) com a soma, por doente, das contagens de todas as sessões, a 120 limiares comuns."""
    lim = np.percentile(score[np.isfinite(score)], np.linspace(55, 99.95, 120))
    docs = sorted(hold.subject_id.unique())
    pos = {s: i for i, s in enumerate(docs)}
    por = np.zeros((len(docs), len(lim), 7))
    for sid, g in hold.groupby('session_id', sort=False):
        y = g[f'y_{direcao}'].fillna(0).values.astype(bool)
        sc = np.nan_to_num(score[g.index.values], nan=-np.inf)
        ev = runs(y)
        for t, tau in enumerate(lim):
            por[pos[g.subject_id.iloc[0]], t] += contagens(y, sc >= tau, ev)
    return por


VARIANTES = {'base': [], '+estavel': ['sub_estavel'], '+alarme': ['sub_alarme'], '+ambos': ['sub_estavel', 'sub_alarme']}
linhas = []
for direcao in ('descida', 'subida'):
    bp = json.load(open(f"{TF}/best_params_quant_{direcao}_tudo.json"))['best_params']
    dev['sub_alarme'], hold['sub_alarme'] = dev[f'sub_alarme_{direcao}'], hold[f'sub_alarme_{direcao}']
    scores, auc = {}, {}
    for nome, extra in VARIANTES.items():
        cols = FEATS + extra
        m = XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=TAU[direcao], n_jobs=NJOBS, tree_method='hist')
        m.fit(dev.loc[idx_d, cols].values, dev.loc[idx_d, 'delta'].values)
        sc = np.full(len(hold), np.nan); sc[tem_h] = SINAL[direcao] * m.predict(hold.loc[idx_h, cols].values)
        scores[nome] = sc
        auc[nome] = roc_auc_score(hold.loc[idx_h, f'y_{direcao}'].values, sc[tem_h])
        print(f"  {direcao} {nome}: AUC teste {auc[nome]:.4f} ({time.time() - t0:.0f} s)", flush=True)
    por = {n: por_doente(s, direcao) for n, s in scores.items()}
    nd = len(por['base'])
    rng = np.random.default_rng(0)
    pesos = [np.bincount(rng.integers(0, nd, nd), minlength=nd) for _ in range(B)]
    fr = {n: (fronteira(p.sum(0)), np.stack([fronteira(np.tensordot(w, p, axes=(0, 0))) for w in pesos])) for n, p in por.items()}
    NOMES = ('sens', 'vpp', 'alarmes_h', 'falsos_h')
    for nome in VARIANTES:
        row = dict(direcao=direcao, variante=nome, auc_teste=auc[nome], n_doentes=nd)
        for j, mn in enumerate(NOMES):
            for a, v in zip(ALVOS, fr[nome][0][:, j]):
                row[f'{mn}_{a}minh'] = v
        if nome != 'base':
            row['dif_auc'] = auc[nome] - auc['base']
            for j, mn in enumerate(NOMES):
                dp = np.nanmean(fr[nome][0][:, j] - fr['base'][0][:, j])
                db = np.nanmean(fr[nome][1][:, :, j] - fr['base'][1][:, :, j], axis=1)
                row.update({f'dif_{mn}_media': float(dp), f'dif_{mn}_ic_baixo': float(np.nanpercentile(db, 2.5)), f'dif_{mn}_ic_alto': float(np.nanpercentile(db, 97.5))})
                i10 = list(ALVOS).index(10)
                d10 = fr[nome][1][:, i10, j] - fr['base'][1][:, i10, j]
                row.update({f'dif10_{mn}': float(fr[nome][0][i10, j] - fr['base'][0][i10, j]), f'dif10_{mn}_ic_baixo': float(np.nanpercentile(d10, 2.5)), f'dif10_{mn}_ic_alto': float(np.nanpercentile(d10, 97.5))})
        linhas.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k.startswith(('direcao', 'variante', 'auc', 'dif_'))}, flush=True)
    pd.DataFrame(linhas).to_csv(f"{AQUI}/resultados_alarmes_e_estavel.csv", index=False)
print("FIM")
