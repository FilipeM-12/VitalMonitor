"""comparar_caracteristicas_novas.py — com a janela de 5 min ANCORADA NO TEMPO, que características novas ajudam? (A/B, mesmos hiperparâmetros)

Usa a tabela da tendência (construir_tabela_ppg_morf.py, com as características novas) e a definição por tempo
(cada sessão numa grade regular de 15 s, blocos em falta = NaN; alvo e variações exatos; características em falta ficam NaN, que o
XGBoost trata; a linha só exige o alvo). Variantes, todas com os hiperparâmetros do modelo final (best_params_quant_<direção>.json):
  base      as 19 do modelo final
  morf      19 + ΔW50, Δcrest, Δdecay a 2 e 5 min (6)
  apg       19 + b/a e d/a (nível z, Δ2m, Δ5m) (6)
  novas     19 + w25, w75, inclinação máxima normalizada, reflection index (nível z, Δ2m, Δ5m) (12)
  tudo      as 19 + todas as anteriores (36)
AUC por bloco (OOF no desenvolvimento, 3 dobras por sujeito, e teste) e IC a 95 % da diferença face à base no teste (bootstrap de sujeitos).
Uso: python3 comparar_caracteristicas_novas.py [pasta_dados] [pasta_best_params] [n_jobs]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

DADOS = sys.argv[1] if len(sys.argv) > 1 else "dados"
BPDIR = sys.argv[2] if len(sys.argv) > 2 else "."
NJOBS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
BLOCO_SEG, D, W_MIN = 15, 15, 5
FEAT_BASE = ['hr_mean_z', 'hr_lag1', 'area_mean_global_z', 'w50_global_z', 'alpha_womersley_global_z',
             'ppg_crest_time_global_z', 'ppg_decay_time_global_z', 'age', 'gender_bin']
SLOPE_COLS = ['hr_mean_z', 'area_mean_global_z', 'alpha_womersley_global_z']
MORF = ['w50_global_z', 'ppg_crest_time_global_z', 'ppg_decay_time_global_z']
APG = ['apg_b_a_global_z', 'apg_d_a_global_z']
NOVAS = ['w25_global_z', 'w75_global_z', 'ppg_max_slope_norm_global_z', 'reflection_index_global_z']
N_BOOT = 500


def grade(df):
    partes = []
    for s, g in df.groupby('session_id', sort=False):
        idx = np.arange(g.bloco_id.min(), g.bloco_id.max() + 1)
        partes.append(g.set_index('bloco_id').reindex(idx).assign(session_id=s).rename_axis('bloco_id').reset_index())
    return pd.concat(partes, ignore_index=True)


def preparar(nome, direcao):
    cols = ['subject_id', 'session_id', 'bloco_id', 'sbp_target'] + FEAT_BASE + SLOPE_COLS + APG + NOVAS
    df = pd.read_csv(os.path.join(DADOS, f'{nome}_ppg.csv'), usecols=lambda c: c in set(cols))
    df = df.sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
    df = grade(df)
    df['subject_id'] = df.groupby('session_id').subject_id.transform('first')
    gb = df.groupby('session_id')
    base = []
    for c in SLOPE_COLS:
        for m in (2, 5, 10):
            df[f'd_{c}_{m}m'] = df[c] - gb[c].shift(int(m * 60 / BLOCO_SEG)); base.append(f'd_{c}_{m}m')
    W5 = int(5 * 60 / BLOCO_SEG)
    df['vol_hr_mean_z_5m'] = gb['hr_mean_z'].transform(lambda s: s.rolling(W5, min_periods=W5).std()); base.append('vol_hr_mean_z_5m')

    def familia(cols):
        out = list(cols) if cols is not MORF else []
        for c in cols:
            for m in (2, 5):
                df[f'd_{c}_{m}m'] = df[c] - gb[c].shift(int(m * 60 / BLOCO_SEG)); out.append(f'd_{c}_{m}m')
        return out
    f_morf, f_apg, f_novas = familia(MORF), familia(APG), familia(NOVAS)
    df['delta'] = df['sbp_target'] - gb['sbp_target'].shift(int(W_MIN * 60 / BLOCO_SEG))
    df['y'] = ((df['delta'] <= -D) if direcao == 'descida' else (df['delta'] >= D)).astype(float)
    df = df[df.sbp_target.notna()].dropna(subset=['y', 'delta']).reset_index(drop=True)
    f0 = FEAT_BASE + base
    return df, {'base': f0, 'morf': f0 + f_morf, 'apg': f0 + f_apg, 'novas': f0 + f_novas, 'tudo': f0 + f_morf + f_apg + f_novas}


def ajustar(bp, q, X, y):
    return XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=q, n_jobs=NJOBS, tree_method='hist').fit(X, y)


def ic_dif(y, pa, pb, suj, rng):
    ids = np.unique(suj); pos = {s: np.where(suj == s)[0] for s in ids}; d = []
    for _ in range(N_BOOT):
        idx = np.concatenate([pos[s] for s in rng.choice(ids, len(ids))])
        if y[idx].min() != y[idx].max():
            d.append(roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], pa[idx]))
    return np.percentile(d, [2.5, 97.5])


saida = []
for direcao in ('descida', 'subida'):
    t0 = time.time()
    bp = json.load(open(os.path.join(BPDIR, f'best_params_quant_{direcao}.json')))['best_params']
    sinal, q = (-1.0, 0.10) if direcao == 'descida' else (1.0, 0.90)
    dev, V = preparar('dev', direcao)
    hold, _ = preparar('holdout', direcao)
    print(direcao, len(dev), 'dev,', len(hold), 'teste; prevalência', f"{dev.y.mean():.4f}/{hold.y.mean():.4f}", flush=True)
    P = {}
    for nome, feats in V.items():
        X, yr = dev[feats].values, dev['delta'].values
        oof = np.zeros(len(dev))
        for tr, va in GroupKFold(n_splits=3).split(X, dev.y.values, dev.subject_id.values):
            oof[va] = sinal * ajustar(bp, q, X[tr], yr[tr]).predict(X[va])
        ph = sinal * ajustar(bp, q, X, yr).predict(hold[feats].values)
        P[nome] = (oof, ph)
        print(f'  {nome:6s} {len(feats):2d} caract.  AUC dev {roc_auc_score(dev.y, oof):.4f}  AUC teste {roc_auc_score(hold.y, ph):.4f}  ({time.time() - t0:.0f} s)', flush=True)
    for nome in V:
        if nome == 'base':
            continue
        ic = ic_dif(hold.y.values, P['base'][1], P[nome][1], hold.subject_id.values, np.random.default_rng(0))
        dd = roc_auc_score(dev.y, P[nome][0]) - roc_auc_score(dev.y, P['base'][0])
        dh = roc_auc_score(hold.y, P[nome][1]) - roc_auc_score(hold.y, P['base'][1])
        print(f'  {nome:6s} - base: dev {dd:+.4f}  teste {dh:+.4f}  IC95 [{ic[0]:+.4f}, {ic[1]:+.4f}]', flush=True)
        saida.append(dict(direcao=direcao, variante=nome, n_caract=len(V[nome]), auc_dev=roc_auc_score(dev.y, P[nome][0]),
                          auc_teste=roc_auc_score(hold.y, P[nome][1]), dif_dev=dd, dif_teste=dh, ic_baixo=ic[0], ic_alto=ic[1]))
pd.DataFrame(saida).to_csv('comparar_caracteristicas_novas.csv', index=False)
print('FIM')
