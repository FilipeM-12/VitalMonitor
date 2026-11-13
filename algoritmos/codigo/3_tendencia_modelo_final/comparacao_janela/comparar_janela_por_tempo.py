"""comparar_janela_por_tempo.py — efeito de ancorar no TEMPO a janela de 5 minutos.

Na definição por posição, a variação a 5 min era PA(bloco i) - PA(bloco i-20) e os Δ das características a 2, 5 e 10 min eram shift(8), shift(20) e
shift(40) sobre os blocos de cada sessão. Como há blocos em falta dentro das sessões (janelas rejeitadas), isso cobre mais de 5 min
em ~41 % das linhas do teste. Aqui cada sessão é posta numa grade regular de 15 s (blocos em falta = linhas NaN), e shift(20) passa
a ser exatamente 300 s: o alvo só existe se houver bloco a t-300 s, e o mesmo para os Δ das características.

Variantes, com os mesmos hiperparâmetros (best_params_quant_<direção>.json) e folds por sujeito:
  bloco        definição atual (referência; reproduz AUC 0,7368 / 0,7771 no teste)
  tempo        alvo e Δ por tempo; características que faltem ficam NaN (o XGBoost trata) e a linha só exige o alvo
  tempo_cheio  como 'tempo', mas só linhas com TODAS as características
Mede a prevalência dos eventos e o AUC por bloco (OOF no desenvolvimento, 3 dobras, e teste).

Uso: python3 comparar_janela_por_tempo.py [pasta_dados] [pasta_best_params] [n_jobs]
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


def grade(df):
    """Cada sessão numa grade regular de blocos: os blocos em falta entram como linhas NaN."""
    partes = []
    for s, g in df.groupby('session_id', sort=False):
        idx = np.arange(g.bloco_id.min(), g.bloco_id.max() + 1)
        partes.append(g.set_index('bloco_id').reindex(idx).assign(session_id=s).rename_axis('bloco_id').reset_index())
    return pd.concat(partes, ignore_index=True)


def construir(df, direcao):
    """Devolve (df com colunas de características, lista de nomes) dado df ordenado por (sessão, bloco)."""
    gb = df.groupby('session_id')
    feats = []
    for c in SLOPE_COLS:
        for m in (2, 5, 10):
            df[f'd_{c}_{m}m'] = df[c] - gb[c].shift(int(m * 60 / BLOCO_SEG)); feats.append(f'd_{c}_{m}m')
    W5 = int(5 * 60 / BLOCO_SEG)
    df['vol_hr_mean_z_5m'] = gb['hr_mean_z'].transform(lambda s: s.rolling(W5, min_periods=W5).std()); feats.append('vol_hr_mean_z_5m')
    df['delta'] = df['sbp_target'] - gb['sbp_target'].shift(int(W_MIN * 60 / BLOCO_SEG))
    df['y'] = ((df['delta'] <= -D) if direcao == 'descida' else (df['delta'] >= D)).astype(float)
    df.loc[df['delta'].isna(), 'y'] = np.nan
    return df, FEAT_BASE + feats


def preparar(nome, direcao, modo):
    cols = ['subject_id', 'session_id', 'bloco_id', 'sbp_target'] + FEAT_BASE + SLOPE_COLS
    df = pd.read_csv(os.path.join(DADOS, f'{nome}_ppg.csv'), usecols=lambda c: c in set(cols))
    df = df.sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
    if modo != 'bloco':
        df = grade(df)
        df['subject_id'] = df.groupby('session_id').subject_id.transform('first')
    df, feats = construir(df, direcao)
    obrig = feats + ['y', 'delta'] if modo in ('bloco', 'tempo_cheio') else ['y', 'delta'] + ['hr_mean_z']
    df = df[df.sbp_target.notna()].dropna(subset=obrig).reset_index(drop=True)
    return df, feats


def ajustar(bp, q, X, y):
    return XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=q, n_jobs=NJOBS, tree_method='hist').fit(X, y)


saida = []
for direcao in ('descida', 'subida'):
    t0 = time.time()
    bp = json.load(open(os.path.join(BPDIR, f'best_params_quant_{direcao}.json')))['best_params']
    sinal, q = (-1.0, 0.10) if direcao == 'descida' else (1.0, 0.90)
    for modo in ('bloco', 'tempo', 'tempo_cheio'):
        dev, feats = preparar('dev', direcao, modo)
        hold, _ = preparar('holdout', direcao, modo)
        X, yr = dev[feats].values, dev['delta'].values
        oof = np.zeros(len(dev))
        for tr, va in GroupKFold(n_splits=3).split(X, dev.y.values, dev.subject_id.values):
            oof[va] = sinal * ajustar(bp, q, X[tr], yr[tr]).predict(X[va])
        ph = sinal * ajustar(bp, q, X, yr).predict(hold[feats].values)
        r = dict(direcao=direcao, modo=modo, n_dev=len(dev), n_teste=len(hold), prev_dev=dev.y.mean(), prev_teste=hold.y.mean(),
                 auc_dev=roc_auc_score(dev.y, oof), auc_teste=roc_auc_score(hold.y, ph))
        saida.append(r)
        print(f"{direcao:8s} {modo:12s} linhas dev {r['n_dev']:8d} teste {r['n_teste']:7d} | prevalência dev {r['prev_dev']:.4f} teste {r['prev_teste']:.4f} "
              f"| AUC dev {r['auc_dev']:.4f} teste {r['auc_teste']:.4f}  ({time.time() - t0:.0f} s)", flush=True)
pd.DataFrame(saida).to_csv('comparar_janela_por_tempo.csv', index=False)
print('FIM')
