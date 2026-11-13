#!/usr/bin/env python3
"""Tabela única da tendência: a de todos os modelos de tendência e a que o servidor reproduz.

Cinco escolhas, todas para a tendência ficar coerente consigo mesma, com o absoluto e com o servidor:
  1. lê as janelas de features_*_ppg_apg.csv, as extraídas com o código atual (o do servidor): o ficheiro
     features_*_ppg_puro.csv veio de uma versão anterior da extração e não coincide com ela em parte das janelas;
  2. as âncoras normalizadas e exigidas são as cinco que os modelos usam (sem alpha_womersley_b, que nenhum usa);
  3. não se exige a APG: o rácio b/a da APG fica normalizado como coluna opcional (NaN onde não existe), só para a via
     APG do Apêndice A, e nenhuma linha sai por lhe faltar;
  4. o conjunto de teste é a lista do absoluto (1_dados_e_caracteristicas/selecao_e_particao_dos_doentes/pacientes_holdout.txt, 50 IDs);
  5. escreve dados/dev_ppg.csv e dados/holdout_ppg.csv nesta pasta.
O resto (colapso em blocos de 15 s, sessões, filtro 70 < PAS < 190, alpha recalculado no bloco, z-score causal pelos 10
primeiros blocos, sessões com 20 blocos ou mais) está no código abaixo.
  venv/bin/python construir_tabela_ppg.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

A = "/home/filipe/Desktop/Tese/algoritmo-ml"
sys.path.insert(0, f"{A}/03_ABSOLUTO_codigo_e_resultados/CODIGO")
from suavizacao_e_normalizacao_causal import normalizar_causal_calibracao  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
PATH_FEATURES = "/media/filipe/Dados/Features"
IDS_TESTE = f"{A}/RESULTADOS_RELATORIO/codigo_dependencias/particao_sujeitos/pacientes_holdout.txt"
OUT_DIR = f"{AQUI}/dados"

RHO_SANGUE = 1060
N_CALIB = 10
MIN_BLOCOS_SESSAO = 20
FEATURES_ANCORA = ['alpha_womersley', 'area_mean', 'w50', 'ppg_crest_time', 'ppg_decay_time']
OPCIONAIS = ['apg_b_a']                     # só para a via APG; NaN permitido
FORA = ['alpha_womersley_b', 'apg_c_a', 'apg_d_a', 'apg_aging']   # colunas das janelas que nenhum modelo usa


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dfs = []
    for g in ("homens", "mulheres"):
        d = pd.read_csv(f"{PATH_FEATURES}/features_{g}_ppg_apg.csv")
        dfs.append(d.drop(columns=[c for c in FORA if c in d.columns]))
        print(f"features_{g}_ppg_apg.csv: {len(d)} janelas", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    df['subject_id'] = df['subject_id'].astype(str)
    print(f"Total: {len(df)} janelas | {df.subject_id.nunique()} sujeitos", flush=True)

    # ── colapso temporal ─────────────────────────────────────────────────────
    df['record_id'] = df['segment_name'].astype(str).apply(lambda x: x.rsplit('_', 1)[0])
    df['bloco_id'] = df['window_idx'] // 3
    colunas_meta = ['subject_id', 'record_id', 'segment_order', 'bloco_id', 'gender', 'dataset', 'age']
    colunas_media = [c for c in df.columns if c not in colunas_meta and c not in ['segment_name', 'window_idx']]
    df = df.groupby(colunas_meta)[colunas_media].mean().reset_index()
    df = df.sort_values(['subject_id', 'record_id', 'segment_order', 'bloco_id']).reset_index(drop=True)
    print(f"-> {len(df)} blocos", flush=True)

    # ── sessões (critério do 11) ─────────────────────────────────────────────
    df['seg_diff'] = df.groupby('record_id')['segment_order'].diff()
    df['win_diff'] = df.groupby(['record_id', 'segment_order'])['bloco_id'].diff()
    df['is_new_session'] = ((df['seg_diff'] > 1) | (df['win_diff'] > 4)
                            | df['seg_diff'].isna() | df['win_diff'].isna())
    df['session_id'] = df['record_id'] + "_" + df.groupby('record_id')['is_new_session'].cumsum().astype(str)
    df = df.sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
    primeiro = df.groupby('session_id')['bloco_id'].transform('min')
    df['tempo_desde_calib_seg'] = (df['bloco_id'] - primeiro) * 15

    # ── filtro de alvo fisiológico ───────────────────────────────────────────
    n0 = len(df)
    df = df[(df['sbp_target'] > 70) & (df['sbp_target'] < 190)].reset_index(drop=True)
    print(f"Filtro PAS 70-190: {n0} -> {len(df)} blocos", flush=True)

    # ── alpha_womersley recalculado ao nível do bloco ────────────────────────
    df['omega_freq'] = df['hr_mean'] / 60.0
    df['alpha_womersley'] = (df['ppg_valley_intensity']
                             * np.sqrt((RHO_SANGUE * df['omega_freq']) / (df['ppg_ac_amplitude'] + 1e-6)))

    # ── z-score causal pela janela de calibração da sessão ───────────────────
    df = normalizar_causal_calibracao(df, ['hr_mean'], n_calib=N_CALIB, group_col='session_id', sufixo='_z')
    df['hr_lag1'] = df.groupby('session_id')['hr_mean_z'].shift(1)
    df['gender_bin'] = df['gender'].map({'M': 0, 'F': 1})
    df = normalizar_causal_calibracao(df, FEATURES_ANCORA + OPCIONAIS, n_calib=N_CALIB,
                                      group_col='session_id', sufixo='_global_z')

    # ── comprimento mínimo de sessão e limpeza ───────────────────────────────
    n0 = len(df)
    df = df.groupby('session_id').filter(lambda x: len(x) >= MIN_BLOCOS_SESSAO)
    print(f"Sessões com >= {MIN_BLOCOS_SESSAO} blocos: {n0} -> {len(df)} blocos", flush=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['hr_lag1', 'hr_mean_z'] + [f'{c}_global_z' for c in FEATURES_ANCORA]).reset_index(drop=True)
    print(f"Tabela final: {len(df)} blocos | {df.subject_id.nunique()} sujeitos | {df.session_id.nunique()} sessões")

    # ── divisão por sujeito: o teste é a lista do absoluto ───────────────────
    ids_teste = {l.strip() for l in open(IDS_TESTE) if l.strip()}
    no_teste = df.subject_id.isin(ids_teste)
    dev, hold = df[~no_teste].reset_index(drop=True), df[no_teste].reset_index(drop=True)
    resumo = dict(dev_blocos=len(dev), dev_sujeitos=int(dev.subject_id.nunique()), dev_sessoes=int(dev.session_id.nunique()),
                  teste_blocos=len(hold), teste_sujeitos=int(hold.subject_id.nunique()),
                  teste_sessoes=int(hold.session_id.nunique()), ids_teste_pedidos=len(ids_teste),
                  ids_teste_sem_dados=sorted(ids_teste - set(df.subject_id)),
                  apg_b_a_em_falta=round(float(df['apg_b_a_global_z'].isna().mean()), 4))
    print(json.dumps(resumo, indent=1, ensure_ascii=False))
    dev.to_csv(f"{OUT_DIR}/dev_ppg.csv", index=False)
    hold.to_csv(f"{OUT_DIR}/holdout_ppg.csv", index=False)
    json.dump(resumo, open(f"{OUT_DIR}/resumo_tabela_ppg.json", "w"), indent=1, ensure_ascii=False)
    print(f"Guardado em {OUT_DIR}/")


if __name__ == "__main__":
    main()
