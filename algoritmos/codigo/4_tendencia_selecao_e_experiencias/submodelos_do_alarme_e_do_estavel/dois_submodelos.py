"""dois_submodelos.py — dois submodelos independentes, um do ALARME e outro do ESTÁVEL, decidem pela comparação.

Versão fiel da ideia (alarmes_e_estavel.py usava as pontuações como características extra de UM modelo supervisionado). Aqui cada submodelo só vê a sua classe:
  modelo do alarme     mistura gaussiana (GMM) ajustada só aos blocos que são evento da direção (descida: Δ <= -15; subida: Δ >= +15)
  modelo do estável    mistura gaussiana ajustada só aos blocos estáveis (|Δ a 5 min| < D/2 = 7,5 mmHg), a mesma nas duas direções
  pontuação            log-verosimilhança no modelo do alarme menos a do estável (razão de verosimilhanças = classificador generativo); maior = mais alarme
Características: as 41 numéricas das 43 (sem idade e sexo), cortadas a +-5 (os z-scores têm caudas muito pesadas) e normalizadas com a média e o desvio dos
blocos estáveis; em falta passa a 0. O estável ajusta-se a uma amostra de N_EST blocos (são ~1 milhão).
Avalia-se (a) a pontuação SOZINHA como detetor e (b) como característica extra do XGBoost quantílico das 43 (previsões fora da amostra no
desenvolvimento, 3 dobras por doente; no teste, modelos ajustados em todo o desenvolvimento), sempre no TESTE, com AUC e sensibilidade por episódio a 4-10 min/h
(fronteira por direção) e IC 95 % da diferença face à base por bootstrap de DOENTES.   Uso: python3 dois_submodelos.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

AQUI = os.path.dirname(os.path.abspath(__file__))
TF = os.path.normpath(os.path.join(AQUI, "..", "..", "3_tendencia_modelo_final", "tendencia_tempo_fixo"))        # tempo_fixo.py e os best_params_*.json só existem em tendencia_tempo_fixo/
sys.path.insert(0, TF)
import tempo_fixo as tf  # noqa: E402

DADOS = os.environ.get("DADOS", f"{AQUI}/dados")
NJOBS = int(os.environ.get("NJOBS", "14"))
D, B, N_EST, N_COMP, CORTE = 15, 1000, 200_000, 8, 5.0
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
idx_d, idx_h = dev.index[tem_d], hold.index[tem_h]
estavel_d = dev.index[tem_d & (dev.delta.abs() < D / 2).values]
t0 = time.time()
print(f"dev {int(tem_d.sum())} blocos ({len(estavel_d)} estáveis), teste {int(tem_h.sum())} blocos", flush=True)


def matriz(frame, idx, mu, sd):
    x = np.clip(np.nan_to_num(frame.loc[idx, Z].values.astype(float), nan=0.0), -CORTE * 1e6, CORTE * 1e6)
    return np.clip((x - mu) / sd, -CORTE, CORTE)


def ajustar_gmm(X, semente=0):
    return GaussianMixture(n_components=N_COMP, covariance_type='full', reg_covar=1e-3, max_iter=100, n_init=1, random_state=semente).fit(X)


def pontuar(treino_idx, aplicar_idx, frame_t, frame_a, direcao):
    """log p(x | alarme) - log p(x | estável), com os dois GMM ajustados só em `treino_idx` (do frame_t)."""
    est = np.intersect1d(treino_idx, estavel_d)
    rng = np.random.default_rng(0)
    est_amostra = rng.choice(est, size=min(N_EST, len(est)), replace=False)
    base = np.nan_to_num(frame_t.loc[est, Z].values.astype(float), nan=0.0)
    base = np.clip(base, -CORTE * 1e6, CORTE * 1e6)
    mu, sd = base.mean(0), base.std(0) + 1e-6
    ev = treino_idx[(frame_t.loc[treino_idx, f'y_{direcao}'] == 1).values]
    g_est, g_alm = ajustar_gmm(matriz(frame_t, est_amostra, mu, sd)), ajustar_gmm(matriz(frame_t, ev, mu, sd))
    Xa = matriz(frame_a, aplicar_idx, mu, sd)
    return g_alm.score_samples(Xa) - g_est.score_samples(Xa)


for d_ in ('descida', 'subida'):
    dev[f'sub_gen_{d_}'], hold[f'sub_gen_{d_}'] = np.nan, np.nan
grupos = dev.loc[idx_d, 'subject_id'].values
for k, (tr, va) in enumerate(GroupKFold(n_splits=3).split(idx_d, groups=grupos)):
    for d_ in ('descida', 'subida'):
        dev.loc[idx_d[va], f'sub_gen_{d_}'] = pontuar(idx_d[tr], idx_d[va], dev, dev, d_)
    print(f"  dobra {k + 1}/3 ({time.time() - t0:.0f} s)", flush=True)
for d_ in ('descida', 'subida'):
    hold.loc[idx_h, f'sub_gen_{d_}'] = pontuar(idx_d, idx_h, dev, hold, d_)
    y = hold.loc[idx_h, f'y_{d_}'].values
    print(f"  AUC da razão de verosimilhanças sozinha no teste, {d_}: {roc_auc_score(y, hold.loc[idx_h, f'sub_gen_{d_}']):.4f} ({time.time() - t0:.0f} s)", flush=True)


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


VARIANTES = {'base': [], '+gen': ['sub_gen'], 'so_gen': None}
linhas = []
for direcao in ('descida', 'subida'):
    bp = json.load(open(f"{TF}/best_params_quant_{direcao}_tudo.json"))['best_params']
    dev['sub_gen'], hold['sub_gen'] = dev[f'sub_gen_{direcao}'], hold[f'sub_gen_{direcao}']
    scores, auc = {}, {}
    for nome, extra in VARIANTES.items():
        if extra is None:                                   # a razão de verosimilhanças sozinha, sem XGBoost
            sc = np.full(len(hold), np.nan); sc[tem_h] = hold.loc[idx_h, 'sub_gen'].values
        else:
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
    pd.DataFrame(linhas).to_csv(f"{AQUI}/resultados_dois_submodelos.csv", index=False)
print("FIM")
