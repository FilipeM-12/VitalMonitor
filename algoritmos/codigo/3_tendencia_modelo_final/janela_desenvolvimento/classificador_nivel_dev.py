"""classificador_nivel_dev.py — a referência do classificador para a tabela dos níveis do quantil.
Mesmas previsões fora da dobra (3 dobras por doente, as mesmas dobras), mesmas medidas, mas com o classificador XGBoost do modelo (best_params_clf_<direção>_tudo.json, scale_pos_weight de todo o desenvolvimento).

Baseado em nivel_quantil_dev.py — qual o nível do quantil (tau) a usar? SÓ NO DESENVOLVIMENTO (as escolhas fazem-se no desenvolvimento).

Regressor XGBoost com perda quantílica sobre o Δ a 5 min (por tempo exato), com as 43 características e os hiperparâmetros do modelo final (best_params_quant_<direção>_tudo.json,
afinados com tau = 0,10 na descida e 0,90 na subida; não se reafina por nível). Níveis: descida tau = 0,05; 0,10; 0,20; 0,30; 0,50 | subida tau = 0,95; 0,90; 0,80; 0,70; 0,50.
Mede, com previsões fora da amostra no desenvolvimento (3 dobras por doente): AUC por bloco e sensibilidade e VPP por episódio a 4, 6, 8 e 10 min/h de falso alarme
(fronteira por direção). O classificador de referência é o de 02_classificador_vs_quantilica.py (AUC fora da amostra no desenvolvimento 0,8123 e 0,8316).
Uso: python3 nivel_quantil_dev.py [pasta_dados] [n_threads]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

AQUI = os.path.dirname(os.path.abspath(__file__))
TF = os.path.normpath(os.path.join(AQUI, "..", "tendencia_tempo_fixo"))        # tempo_fixo.py e os best_params_*.json só existem em tendencia_tempo_fixo/
sys.path.insert(0, TF)
import tempo_fixo as tf  # noqa: E402

DADOS = sys.argv[1] if len(sys.argv) > 1 else f"{AQUI}/dados"
NJOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 14
D, BLOCO_SEG, GAP_MERGE, PRE_CREDITO, DUTY_MAX = 15, 15, 4, 8, 0.30
ALVOS = np.array([4, 6, 8, 10])
SINAL = {'descida': -1.0, 'subida': 1.0}
TAUS = {'descida': (0.05, 0.10, 0.20, 0.30, 0.50), 'subida': (0.95, 0.90, 0.80, 0.70, 0.50)}

dev, V = tf.preparar_grade(DADOS, 'dev')
dev['subject_id'] = dev.subject_id.astype(str)
FEATS = V['tudo']


def runs(mask, gap=GAP_MERGE):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    cortes = np.where(np.diff(idx) > gap + 1)[0]
    return [(g[0], g[-1]) for g in np.split(idx, cortes + 1)]


def contagens(y, alarme, eventos):
    al = runs(alarme)
    toca = lambda a0, a1: any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (e0, e1) in eventos)
    det = sum(any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (a0, a1) in al) for (e0, e1) in eventos)
    falsos = [(a0, a1) for (a0, a1) in al if not toca(a0, a1)]
    fa = sum((a1 - a0 + 1) * BLOCO_SEG / 60.0 for (a0, a1) in falsos)
    return det, len(eventos), fa, sum(a1 - a0 + 1 for (a0, a1) in al), len(y), len(al), len(falsos)


def fronteira(soma):
    vazio = np.full((len(ALVOS), 2), np.nan)
    horas = soma[:, 4] * BLOCO_SEG / 3600.0
    sens, fa = soma[:, 0] / np.maximum(soma[:, 1], 1), soma[:, 2] / horas
    vpp = 1 - soma[:, 6] / np.maximum(soma[:, 5], 1)
    ok = (soma[:, 3] / soma[:, 4]) <= DUTY_MAX
    if ok.sum() < 2:
        return vazio
    s, f, v = sens[ok], fa[ok], vpp[ok]
    o = np.argsort(f, kind='stable'); s, f, v = s[o], f[o], v[o]
    m = np.concatenate([[True], s[1:] > np.maximum.accumulate(s)[:-1]]); s, f, v = s[m], f[m], v[m]
    if len(s) < 2:
        return vazio
    return np.column_stack([np.interp(ALVOS, f, x, left=np.nan, right=x[-1]) for x in (s, v)])


def por_doente(frame, score, y_col, lim):
    """(n_doentes, T, 7): soma das contagens das sessões de cada doente, a cada limiar."""
    docs = sorted(frame.subject_id.unique()); pos = {s: i for i, s in enumerate(docs)}
    grupos = {}
    for sid, g in frame.groupby('session_id', sort=False):
        grupos.setdefault(pos[g.subject_id.iloc[0]], []).append((g[y_col].fillna(0).values.astype(bool), np.nan_to_num(score[g.index.values], nan=-np.inf)))

    def um(i):
        acc = np.zeros((len(lim), 7))
        for y, sc in grupos[i]:
            ev = runs(y)
            for t, tau in enumerate(lim):
                acc[t] += contagens(y, sc >= tau, ev)
        return i, acc
    out = np.zeros((len(docs), len(lim), 7))
    for i, acc in Parallel(n_jobs=14)(delayed(um)(i) for i in sorted(grupos)):
        out[i] = acc
    return out


def alvo_janela(frame, W, direcao):
    n = int(W * 60 / BLOCO_SEG)
    delta = frame['sbp_target'] - frame.groupby('session_id')['sbp_target'].shift(n)
    y = np.where(delta.isna(), np.nan, ((delta <= -D) if direcao == 'descida' else (delta >= D)).astype(float))
    return delta, y


linhas = []
for direcao in ('descida', 'subida'):
    bp = json.load(open(f"{TF}/best_params_quant_{direcao}_tudo.json"))['best_params']
    dev['delta_W'], dev['y_W'] = alvo_janela(dev, 5, direcao)
    td = dev.delta_W.notna().values
    dv = dev[td]
    X, yb, yr, suj = dv[FEATS].values, dv['y_W'].values, dv['delta_W'].values, dv['subject_id'].values
    folds = list(GroupKFold(n_splits=3).split(X, yb, suj))
    bpc = json.load(open(f"{TF}/best_params_clf_{direcao}_tudo.json"))['best_params']
    pw = (len(yb) - yb.sum()) / max(yb.sum(), 1)
    t0 = time.time()
    oof = np.zeros(len(yb))
    for tr, va in folds:
        oof[va] = XGBClassifier(**bpc, scale_pos_weight=pw, eval_metric='auc', n_jobs=NJOBS, tree_method='hist').fit(X[tr], yb[tr]).predict_proba(X[va])[:, 1]
    sc = np.full(len(dev), np.nan); sc[np.where(td)[0]] = oof
    lim = np.percentile(oof, np.linspace(55, 99.95, 80))
    fr = fronteira(por_doente(dev, sc, 'y_W', lim).sum(0))
    row = dict(direcao=direcao, modelo='classificador', auc_dev=float(roc_auc_score(yb, oof)), **{f'sens_{a}minh': float(v) for a, v in zip(ALVOS, fr[:, 0])},
               **{f'vpp_{a}minh': float(v) for a, v in zip(ALVOS, fr[:, 1])}, minutos=round((time.time() - t0) / 60, 1))
    linhas.append(row)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    pd.DataFrame(linhas).to_csv(f"{AQUI}/resultados_classificador_dev.csv", index=False)
print("FIM")
