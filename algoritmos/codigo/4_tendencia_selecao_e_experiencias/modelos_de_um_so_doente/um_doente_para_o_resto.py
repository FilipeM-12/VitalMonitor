"""um_doente_para_o_resto.py — treinar com UM doente (o com mais eventos) e testar nos outros.

Modelo: o XGBoost quantílico das 43 características com os hiperparâmetros do modelo geral (01_optuna_tempo.py), treinado só com:
  p006052          o doente com mais blocos de evento do desenvolvimento (168 h, 13 % dos blocos em evento, 5 273 blocos de evento)
  p000652          o doente com mais horas (209 h) mas com a taxa de eventos típica (3,9 %), para separar «muitos eventos» de «muitos dados»
  aleatorio_1..3   o MESMO número de blocos (o de p006052), sorteados ao acaso de todos os doentes do desenvolvimento, para separar «um só doente» de
                   «poucos dados»
  todos            todo o desenvolvimento (a referência de sempre)
Testa-se nos 48 doentes do TESTE (que nenhum modelo viu): AUC, sensibilidade por episódio e VPP a 4-10 min/h (fronteira por direção) e IC 95 % da
diferença face a «todos» por bootstrap de doentes. Para p006052 e p000652 mede-se ainda o AUC nos restantes 301 doentes do desenvolvimento.
Uso: python3 um_doente_para_o_resto.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor

AQUI = os.path.dirname(os.path.abspath(__file__))
TF = os.path.normpath(os.path.join(AQUI, "..", "..", "3_tendencia_modelo_final", "tendencia_tempo_fixo"))        # tempo_fixo.py e os best_params_*.json só existem em tendencia_tempo_fixo/
sys.path.insert(0, TF)
import tempo_fixo as tf  # noqa: E402

DADOS = os.environ.get("DADOS", f"{AQUI}/dados")
NJOBS = int(os.environ.get("NJOBS", "14"))
B = 1000
BLOCO_SEG, GAP_MERGE, PRE_CREDITO, DUTY_MAX = 15, 4, 8, 0.30
ALVOS = np.array([4, 6, 8, 10])
TAU = {'descida': 0.10, 'subida': 0.90}
SINAL = {'descida': -1.0, 'subida': 1.0}
RICO, TIPICO = 'p006052', 'p000652'

dev, V = tf.preparar_grade(DADOS, 'dev')
hold, _ = tf.preparar_grade(DADOS, 'holdout')
for d in (dev, hold):
    d['subject_id'] = d.subject_id.astype(str)
FEATS = V['tudo']
tem_d, tem_h = dev.delta.notna().values, hold.delta.notna().values
idx_d, idx_h = dev.index[tem_d], hold.index[tem_h]
n_rico = int((tem_d & (dev.subject_id == RICO).values).sum())
rng0 = np.random.default_rng(0)
CONJUNTOS = {
    RICO: idx_d[(dev.loc[idx_d, 'subject_id'] == RICO).values],
    TIPICO: idx_d[(dev.loc[idx_d, 'subject_id'] == TIPICO).values],
    **{f'aleatorio_{k}': np.sort(rng0.choice(idx_d, size=n_rico, replace=False)) for k in (1, 2, 3)},
    'todos': idx_d,
}
t0 = time.time()
print(f"dev {len(idx_d)} blocos, teste {len(idx_h)} blocos; blocos de treino do doente rico: {n_rico}", flush=True)


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


linhas = []
for direcao in ('descida', 'subida'):
    bp = json.load(open(f"{TF}/best_params_quant_{direcao}_tudo.json"))['best_params']
    scores, auc, auc_resto = {}, {}, {}
    for nome, treino in CONJUNTOS.items():
        m = XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=TAU[direcao], n_jobs=NJOBS, tree_method='hist')
        m.fit(dev.loc[treino, FEATS].values, dev.loc[treino, 'delta'].values)
        sc = np.full(len(hold), np.nan); sc[tem_h] = SINAL[direcao] * m.predict(hold.loc[idx_h, FEATS].values)
        scores[nome] = sc
        auc[nome] = roc_auc_score(hold.loc[idx_h, f'y_{direcao}'].values, sc[tem_h])
        if nome in (RICO, TIPICO):
            resto = idx_d[(dev.loc[idx_d, 'subject_id'] != nome).values]
            auc_resto[nome] = roc_auc_score(dev.loc[resto, f'y_{direcao}'].values, SINAL[direcao] * m.predict(dev.loc[resto, FEATS].values))
        print(f"  {direcao} treinado em {nome} ({len(treino)} blocos): AUC teste {auc[nome]:.4f}"
              + (f" | AUC nos outros doentes do desenvolvimento {auc_resto[nome]:.4f}" if nome in auc_resto else '') + f" ({time.time() - t0:.0f} s)", flush=True)
    por = {n: por_doente(s, direcao) for n, s in scores.items()}
    nd = len(por['todos'])
    rng = np.random.default_rng(0)
    pesos = [np.bincount(rng.integers(0, nd, nd), minlength=nd) for _ in range(B)]
    fr = {n: (fronteira(p.sum(0)), np.stack([fronteira(np.tensordot(w, p, axes=(0, 0))) for w in pesos])) for n, p in por.items()}
    NOMES = ('sens', 'vpp', 'alarmes_h', 'falsos_h')
    for nome in CONJUNTOS:
        row = dict(direcao=direcao, treinado_em=nome, blocos_treino=len(CONJUNTOS[nome]), auc_teste=auc[nome], auc_outros_dev=auc_resto.get(nome, np.nan), n_doentes=nd)
        for j, mn in enumerate(NOMES):
            for a, v in zip(ALVOS, fr[nome][0][:, j]):
                row[f'{mn}_{a}minh'] = v
        if nome != 'todos':
            for j, mn in enumerate(NOMES):
                dp = np.nanmean(fr[nome][0][:, j] - fr['todos'][0][:, j])
                db = np.nanmean(fr[nome][1][:, :, j] - fr['todos'][1][:, :, j], axis=1)
                row.update({f'dif_{mn}_media': float(dp), f'dif_{mn}_ic_baixo': float(np.nanpercentile(db, 2.5)), f'dif_{mn}_ic_alto': float(np.nanpercentile(db, 97.5))})
                i10 = list(ALVOS).index(10)
                d10 = fr[nome][1][:, i10, j] - fr['todos'][1][:, i10, j]
                row.update({f'dif10_{mn}': float(fr[nome][0][i10, j] - fr['todos'][0][i10, j]), f'dif10_{mn}_ic_baixo': float(np.nanpercentile(d10, 2.5)), f'dif10_{mn}_ic_alto': float(np.nanpercentile(d10, 97.5))})
        linhas.append(row)
    pd.DataFrame(linhas).to_csv(f"{AQUI}/resultados_um_doente_para_o_resto.csv", index=False)
print("FIM")
