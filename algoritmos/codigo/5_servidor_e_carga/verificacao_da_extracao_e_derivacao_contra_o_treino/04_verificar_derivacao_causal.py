#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derivação CAUSAL, 20 características: TREINO (vetorizada) contra PRODUÇÃO (incremental).

O treino e a produção normalizam as âncoras da mesma maneira (janela de calibração de 10 blocos),
por isso TODAS as características têm de bater.

  treino    colapso das janelas em blocos de 15 s (média de 3), alpha recalculado no
            bloco, normalizar_causal_calibracao (suavizacao_e_normalizacao_causal.py, a MESMA função do treino) sobre os 10 primeiros
            blocos, hr_lag1, declives a 2/5/10 min por shift posicional, volatilidade a 5 min (rolling.std, ddof 1)
  produção  servidor/workers/derivacao_ppg_classificador.py: bloco a bloco, com estado, sem ver o futuro

A extração das janelas (função pura) usa o código do servidor, já verificado idêntico ao do treino
(01_verificar_extracao.py). Compara, bloco a bloco (mesmo _bid), as características da lista do modelo instalado.

USO  python3 04_verificar_derivacao_causal.py [n_registos] [modelo.joblib]
"""
import importlib.util
import os
import sys

import joblib
import numpy as np
import pandas as pd
import wfdb

RAIZ = "/home/filipe/Desktop/Tese"
EXTR_PROD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "servidor", "workers", "extracao_ppg_classificador.py")
DERIV_PROD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "servidor", "workers", "derivacao_ppg_classificador.py")
MODELO = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "servidor", "modelos", "modelo_tendencia_ppg_quant.joblib")
DATASET = "/media/filipe/Dados/MIMIC_DATASET_CONSOLIDADO/dataset_para_extração_features.csv"
SAIDA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{RAIZ}/algoritmo-ml/03_ABSOLUTO_codigo_e_resultados/CODIGO")
from suavizacao_e_normalizacao_causal import normalizar_causal_calibracao  # noqa: E402

WINDOW_SEC, STRIDE_SEC, BLOCO_SEG = 15, 5, 15
JANELAS_POR_BLOCO, N_CALIB = 3, 10
ESCALAS_MIN = [2, 5, 10]
ANCORAS = ['alpha_womersley', 'area_mean', 'w50', 'ppg_crest_time', 'ppg_decay_time', 'alpha_womersley_b']
SLOPE_COLS = ['hr_mean_z', 'area_mean_global_z', 'alpha_womersley_global_z']
RHO = 1060
IDADE, GENERO = 60, 'M'
MIN_JANELAS = 3 * (int(10 * 60 / BLOCO_SEG) + N_CALIB + 5)


def carregar(caminho, nome):
    spec = importlib.util.spec_from_file_location(nome, caminho)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nome] = mod
    spec.loader.exec_module(mod)
    return mod


def janelas(base_path, n_max=900):
    hdr = wfdb.rdheader(base_path)
    fs = hdr.fs
    sig = [s.upper() for s in hdr.sig_name]
    i = next((k for k, s in enumerate(sig) if "PLETH" in s or "PPG" in s), None)
    if i is None:
        return None, []
    ppg = wfdb.rdrecord(base_path, channels=[i]).p_signal[:, 0].astype(np.float32)
    if np.isnan(ppg).any():
        ppg = pd.Series(ppg).interpolate().ffill().bfill().values
    n, p = int(WINDOW_SEC * fs), int(STRIDE_SEC * fs)
    return fs, [ppg[k * p: k * p + n] for k in range(min((len(ppg) - n) // p + 1, n_max))]


def derivacao_treino(linhas):
    """Replica 11_construir_causal_ppg_puro.py (um registo = uma sessão) + add_multi()."""
    df = pd.DataFrame(linhas)
    df['bloco_id'] = df.window_idx // JANELAS_POR_BLOCO
    num = [c for c in df.columns if c not in ('window_idx', 'bloco_id')]
    df = df.groupby('bloco_id', as_index=False)[num].mean().sort_values('bloco_id').reset_index(drop=True)
    df['alpha_womersley'] = df.ppg_valley_intensity * np.sqrt((RHO * (df.hr_mean / 60.0)) / (df.ppg_ac_amplitude + 1e-6))
    df['session_id'] = 'S'
    df = normalizar_causal_calibracao(df, ['hr_mean'], n_calib=N_CALIB, group_col='session_id', sufixo='_z')
    df['hr_lag1'] = df.groupby('session_id')['hr_mean_z'].shift(1)
    df = normalizar_causal_calibracao(df, ANCORAS, n_calib=N_CALIB, group_col='session_id', sufixo='_global_z')
    df['age'], df['gender_bin'] = IDADE, (0 if GENERO == 'M' else 1)
    gb = df.groupby('session_id')
    for c in SLOPE_COLS:
        for m in ESCALAS_MIN:
            df[f'd_{c}_{m}m'] = df[c] - gb[c].shift(int(m * 60 / BLOCO_SEG))
    W5 = int(5 * 60 / BLOCO_SEG)
    df['vol_hr_mean_z_5m'] = gb['hr_mean_z'].transform(lambda s: s.rolling(W5, min_periods=W5).std())
    return df


def main():
    n_reg = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for f in (EXTR_PROD, DERIV_PROD, DATASET, MODELO):
        if not os.path.exists(f):
            sys.exit(f"não encontrado: {f}")
    mod = joblib.load(MODELO)
    feats = list(mod["descida"]["feats"])
    assert feats == list(mod["subida"]["feats"])
    assert not any(f.startswith("acc_") for f in feats), "o modelo ainda tem acc_*"
    print(f"modelo: {MODELO}\n{len(feats)} características: {feats}\n")

    extr = carregar(EXTR_PROD, "extr_prod")
    dv = carregar(DERIV_PROD, "deriv_prod")
    df_reg = pd.read_csv(DATASET)

    pares = {f: ([], []) for f in feats}
    usados, vistos, n_blocos = 0, set(), 0
    for _, row in df_reg.iterrows():
        sujeito = str(row.get("subject_folder", ""))
        if sujeito in vistos or "3542049" in str(row.get("segment", "")) + sujeito:
            continue                                    # registo 3542049: canais desalinhados, assinalado na tese
        if usados >= n_reg:
            break
        base = str(row["absolute_path"]).replace(".dat", "").replace(".hea", "")
        if not os.path.exists(base + ".hea"):
            continue
        try:
            fs, ws = janelas(base)
        except Exception:
            continue
        if len(ws) < MIN_JANELAS:
            continue
        linhas, prod = [], []
        estado = dv.EstadoSessaoPPG()
        info = {'idade': IDADE, 'genero': GENERO}
        for k, w in enumerate(ws):
            raw = extr.processar_janela_so_ppg(w.copy(), fs)
            if raw is not None:
                linhas.append({**raw, 'window_idx': k})
            out = dv.derivar_features_ppg(raw, estado, info)       # TODAS as chamadas, como no treino
            if out is not None:
                prod.append(out)
        if len(prod) < 5:
            continue
        usados += 1
        vistos.add(sujeito)
        tre = derivacao_treino(linhas)
        idx_tre = {int(b): i for i, b in enumerate(tre.bloco_id)}
        casados = 0
        for o in prod:
            b = o.get('_bid')
            if b is None or b not in idx_tre:
                continue
            lin = tre.iloc[idx_tre[b]]
            if any(pd.isna(lin[f]) for f in feats):
                continue                                           # o treino descarta (sem histórico completo)
            casados += 1
            for f in feats:
                v = o.get(f)
                assert v is not None, f"produção sem {f} no bloco {b}"
                pares[f][0].append(float(lin[f]))
                pares[f][1].append(float(v))
        n_blocos += casados
        print(f"  registo {usados}: {os.path.basename(base)}  {len(ws)} janelas -> {len(prod)} blocos produção, "
              f"{casados} comparados", flush=True)

    if not n_blocos:
        sys.exit("nenhum bloco comparável")
    linhas_out = []
    print(f"\n  {'característica':<28}{'n':>7}{'dif abs máx':>14}{'dif rel máx':>13}{'r':>11}")
    for f in feats:
        x, y = np.array(pares[f][0]), np.array(pares[f][1])
        dif = float(np.abs(x - y).max())
        rel = float((np.abs(x - y) / np.maximum(np.abs(x), 1e-3)).max())
        r = 1.0 if (np.std(x) < 1e-15 and np.std(y) < 1e-15) else float(np.corrcoef(x, y)[0, 1])
        print(f"  {f:<28}{len(x):>7}{dif:>14.2e}{rel:>13.2e}{r:>11.7f}")
        linhas_out.append(dict(caracteristica=f, n=len(x), dif_max=dif, dif_rel_max=rel, r=r))
    pd.DataFrame(linhas_out).to_csv(f"{SAIDA}/resultado_derivacao_causal.csv", index=False)
    r_min = min(d['r'] for d in linhas_out)
    rel_max = max(d['dif_rel_max'] for d in linhas_out)
    print(f"\n  >>> {len(feats)} características, {usados} registos, {n_blocos} blocos: r mínimo = {r_min:.7f}, "
          f"diferença relativa máxima = {rel_max:.2e}")
    print(f"  guardado: {SAIDA}/resultado_derivacao_causal.csv")
    dif_max = max(d['dif_max'] for d in linhas_out)
    print(f"  diferença absoluta máxima = {dif_max:.2e}")
    sys.exit(0 if (r_min > 0.999999 and dif_max < 1e-6) else 1)


if __name__ == "__main__":
    main()
