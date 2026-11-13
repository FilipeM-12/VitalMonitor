"""tempo_fixo.py — definição da tendência com a janela de 5 min ANCORADA NO TEMPO.

Na definição por posição, a variação a 5 min era PA(bloco i) - PA(bloco i-20) e os Δ das características a 2, 5 e 10 min eram shift(8), shift(20) e shift(40) sobre os blocos de cada
sessão. Como há blocos em falta dentro das sessões, isso cobria mais de 5 min em ~41 % das linhas do teste. Aqui cada sessão é posta numa
grade regular de 15 s (blocos em falta = linhas NaN) e shift(20) passa a ser exatamente 300 s: o alvo só existe se houver bloco a t-300 s,
e as variações das características só existem se houver bloco no instante certo (senão ficam NaN, que o XGBoost trata).

Uso:  df, familias = preparar_grade(pasta_dados, 'dev')      # df na grade completa, com delta, y_descida, y_subida e as características
      feats = familias['tudo']                                   # base, morf, apg, novas, tudo
      tem_alvo = df['delta'].notna()
"""
import os

import numpy as np
import pandas as pd

BLOCO_SEG, D, W_MIN = 15, 15, 5
FEAT_BASE = ['hr_mean_z', 'hr_lag1', 'area_mean_global_z', 'w50_global_z', 'alpha_womersley_global_z',
             'ppg_crest_time_global_z', 'ppg_decay_time_global_z', 'age', 'gender_bin']
SLOPE_COLS = ['hr_mean_z', 'area_mean_global_z', 'alpha_womersley_global_z']
MORF = ['w50_global_z', 'ppg_crest_time_global_z', 'ppg_decay_time_global_z']
APG = ['apg_b_a_global_z', 'apg_d_a_global_z']
NOVAS = ['w25_global_z', 'w75_global_z', 'ppg_max_slope_norm_global_z', 'reflection_index_global_z']


def grade(df):
    """Cada sessão numa grade regular de blocos: os blocos em falta entram como linhas NaN."""
    partes = []
    for s, g in df.groupby('session_id', sort=False):
        idx = np.arange(g.bloco_id.min(), g.bloco_id.max() + 1)
        partes.append(g.set_index('bloco_id').reindex(idx).assign(session_id=s).rename_axis('bloco_id').reset_index())
    out = pd.concat(partes, ignore_index=True)
    out['subject_id'] = out.groupby('session_id').subject_id.transform('first')
    return out


def preparar_grade(pasta, nome):
    cols = ['subject_id', 'session_id', 'bloco_id', 'sbp_target'] + FEAT_BASE + SLOPE_COLS + APG + NOVAS
    df = pd.read_csv(os.path.join(pasta, f'{nome}_ppg.csv'), usecols=lambda c: c in set(cols))
    df = df.sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
    df = grade(df)
    gb = df.groupby('session_id')
    sh = lambda c, m: gb[c].shift(int(m * 60 / BLOCO_SEG))
    base = []
    for c in SLOPE_COLS:
        for m in (2, 5, 10):
            df[f'd_{c}_{m}m'] = df[c] - sh(c, m); base.append(f'd_{c}_{m}m')
    W5 = int(5 * 60 / BLOCO_SEG)
    df['vol_hr_mean_z_5m'] = gb['hr_mean_z'].transform(lambda s: s.rolling(W5, min_periods=W5).std()); base.append('vol_hr_mean_z_5m')

    def familia(cols, niveis=True):
        out = list(cols) if niveis else []
        for c in cols:
            for m in (2, 5):
                df[f'd_{c}_{m}m'] = df[c] - sh(c, m); out.append(f'd_{c}_{m}m')
        return out
    f_morf, f_apg, f_novas = familia(MORF, False), familia(APG), familia(NOVAS)
    df['delta'] = df['sbp_target'] - sh('sbp_target', W_MIN)
    df['y_descida'] = np.where(df['delta'].isna(), np.nan, (df['delta'] <= -D).astype(float))
    df['y_subida'] = np.where(df['delta'].isna(), np.nan, (df['delta'] >= D).astype(float))
    f0 = FEAT_BASE + base
    tudo = f0 + f_morf + f_apg + f_novas
    return df, {'base': f0, 'morf': f0 + f_morf, 'apg': f0 + f_apg, 'novas': f0 + f_novas, 'tudo': tudo,
                'sem_ri': [c for c in tudo if 'reflection_index' not in c]}      # as 43 menos o reflection index e as suas duas variações (40)
