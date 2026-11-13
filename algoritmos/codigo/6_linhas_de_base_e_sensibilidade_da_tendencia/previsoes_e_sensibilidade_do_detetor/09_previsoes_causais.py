#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gera as previsoes do modelo CAUSAL sobre o holdout CAUSAL.

PORQUE ISTO E PRECISO
─────────────────────
O 08_ponto_operacao_causal.py le as previsoes de previsoes_apg_base_*.csv.gz e
calcula sobre elas o ponto de operacao final com portao. Se estas previsoes vierem do pipeline nao causal, o script reproduz o resultado nao causal casa por casa, sem medir nada.

Aquele ficheiro nao e uma entrada fixa de referencia: e o objeto da analise.
Este script produz a versao causal dele, com o mesmo formato de colunas
(session_id, bloco_id, y, p), para o 08 passar a medir o que deve.

O modelo e o mesmo que foi exportado para o .joblib causal: XGBRegressor
quantilico com os best_params congelados, treinado no dev causal.
"""

import json
import os

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

RAIZ = os.path.dirname(os.path.abspath(__file__))
BLOCO_SEG, D, W_MIN = 15, 15, 5
ESCALAS_MIN = [2, 5, 10]
FEAT_BASE = ['hr_mean_z', 'hr_lag1', 'area_mean_global_z', 'w50_global_z',
             'alpha_womersley_global_z', 'ppg_crest_time_global_z',
             'ppg_decay_time_global_z', 'age', 'gender_bin']
SLOPE_COLS = ['hr_mean_z', 'area_mean_global_z', 'alpha_womersley_global_z']


def add_multi(df):
    gb = df.groupby('session_id')
    novos = []
    for c in SLOPE_COLS:
        for m in ESCALAS_MIN:
            df[f'd_{c}_{m}m'] = df[c] - gb[c].shift(int(m * 60 / BLOCO_SEG))
            novos.append(f'd_{c}_{m}m')
    W5 = int(5 * 60 / BLOCO_SEG)
    df['vol_hr_mean_z_5m'] = gb['hr_mean_z'].transform(
        lambda s: s.rolling(W5, min_periods=W5).std())
    novos.append('vol_hr_mean_z_5m')
    return df, novos


def preparar(ficheiro, direcao):
    usecols = list(dict.fromkeys(
        ['subject_id', 'session_id', 'bloco_id', 'sbp_target'] + FEAT_BASE + SLOPE_COLS))
    df = pd.read_csv(f"{RAIZ}/dados/{ficheiro}", usecols=lambda c: c in usecols)
    df = df.sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
    df, novos = add_multi(df)
    gb = df.groupby('session_id')
    df['delta'] = df['sbp_target'] - gb['sbp_target'].shift(int(W_MIN * 60 / BLOCO_SEG))
    df['y'] = ((df['delta'] <= -D) if direcao == 'descida'
               else (df['delta'] >= +D)).astype(float)
    feats = [c for c in FEAT_BASE if c in df.columns] + novos
    return df.dropna(subset=feats + ['y', 'delta']).reset_index(drop=True), feats


def main():
    for direcao in ('descida', 'subida'):
        bp = json.load(open(f"{RAIZ}/best_params_quant_{direcao}.json"))['best_params']
        q = 0.10 if direcao == 'descida' else 0.90
        sinal = -1.0 if direcao == 'descida' else 1.0

        dev, feats = preparar('dev_ppg.csv', direcao)
        hold, _ = preparar('holdout_ppg.csv', direcao)
        print(f"{direcao}: dev {len(dev)} | holdout {len(hold)} blocos", flush=True)

        m = XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=q,
                         n_jobs=8, tree_method='hist').fit(dev[feats].values,
                                                           dev['delta'].values)
        out = pd.DataFrame({
            'session_id': hold.session_id.values,
            'bloco_id': hold.bloco_id.values,
            'y': hold.y.values.astype(int),
            'p': sinal * m.predict(hold[feats].values),
        })
        dest = f"{RAIZ}/previsoes_causal_{direcao}.csv.gz"
        out.to_csv(dest, index=False, compression='gzip')
        print(f"  guardado {dest}  ({len(out)} linhas, "
              f"score p1={np.percentile(out.p,1):.2f} p99={np.percentile(out.p,99):.2f})")


if __name__ == "__main__":
    main()
