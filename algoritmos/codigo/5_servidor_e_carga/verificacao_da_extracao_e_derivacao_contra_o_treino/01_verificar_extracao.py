#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Equivalência numérica entre a extração de TREINO e a de PRODUÇÃO (PPG puro).

O QUE ISTO VERIFICA
───────────────────
A cadeia de extração existe duas vezes:

  treino    algoritmo-ml/.../pipeline_ppg_puro/01_extrair_features_ppg_puro.py
  produção  servidor/workers/extracao_ppg_classificador.py

São implementações separadas do mesmo procedimento. Se divergirem, mesmo num
pormenor, nada do que foi medido no MIMIC se aplica ao protótipo — e a
divergência não dá erro nenhum, o worker limita-se a produzir números
ligeiramente diferentes.

A tese afirma r = 1,00000 entre as duas cadeias, mas essa verificação foi feita
para as características do modelo de PA ABSOLUTA. Este script faz o
equivalente para a cadeia PPG-pura, que é a da configuração proposta.

COMO
────
Ambas expõem `processar_janela_so_ppg(win_ppg, fs)` com a mesma assinatura.
Dá-se a MESMA janela de sinal real do MIMIC às duas e comparam-se todas as
características devolvidas, com três medidas:

  · diferença máxima absoluta
  · diferença máxima relativa
  · correlação de Pearson sobre todas as janelas

A correlação sozinha não bastaria: duas implementações podem produzir séries
perfeitamente correlacionadas e deslocadas por um fator de escala. Por isso se
reporta também a diferença absoluta.

USO
    python3 01_verificar_extracao.py [n_registos]
"""

import os
import sys
import importlib.util

import numpy as np
import pandas as pd
import wfdb

RAIZ = "/home/filipe/Desktop/Tese"
TREINO = f"{RAIZ}/algoritmo-ml/04_TENDENCIA_codigo_e_resultados/SBP_Tendencia_so_PPG/pipeline_ppg_puro/01_extrair_features_ppg_puro.py"
PRODUCAO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "servidor", "workers", "extracao_ppg_classificador.py")
DATASET = "/media/filipe/Dados/MIMIC_DATASET_CONSOLIDADO/dataset_para_extração_features.csv"

WINDOW_SEC, STRIDE_SEC = 15, 5
MAX_JANELAS = 400          # por registo, chega para a comparação


def carregar(caminho, nome):
    """Importa um ficheiro .py solto como módulo, sem o executar como script."""
    spec = importlib.util.spec_from_file_location(nome, caminho)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nome] = mod
    spec.loader.exec_module(mod)
    return mod


def janelas_do_registo(base_path, n_max):
    """Devolve janelas de PPG filtrado, como o treino as constrói."""
    hdr = wfdb.rdheader(base_path)
    fs = hdr.fs
    sig = [s.upper() for s in hdr.sig_name]
    idx = next((i for i, s in enumerate(sig) if "PLETH" in s or "PPG" in s), None)
    if idx is None:
        return None, []
    rec = wfdb.rdrecord(base_path, channels=[idx])
    ppg = rec.p_signal[:, 0].astype(np.float32)
    if np.isnan(ppg).any():
        ppg = pd.Series(ppg).interpolate().ffill().bfill().values

    n_win, passo = int(WINDOW_SEC * fs), int(STRIDE_SEC * fs)
    out = []
    for k in range(min((len(ppg) - n_win) // passo + 1, n_max)):
        out.append(ppg[k * passo: k * passo + n_win])
    return fs, out


def main():
    n_reg = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for f in (TREINO, PRODUCAO, DATASET):
        if not os.path.exists(f):
            sys.exit(f"não encontrado: {f}")

    tre = carregar(TREINO, "extr_treino")
    pro = carregar(PRODUCAO, "extr_producao")
    print(f"treino  : {TREINO}")
    print(f"produção: {PRODUCAO}\n")

    df = pd.read_csv(DATASET)
    if "absolute_path" not in df.columns:
        sys.exit("o dataset não tem a coluna absolute_path")

    vals = {}          # caracteristica -> (lista_treino, lista_producao)
    n_jan = n_ok = n_reg_usados = 0

    for _, row in df.iterrows():
        if n_reg_usados >= n_reg:
            break
        base = str(row["absolute_path"]).replace(".dat", "").replace(".hea", "")
        if not os.path.exists(base + ".hea"):
            continue
        try:
            fs, janelas = janelas_do_registo(base, MAX_JANELAS)
        except Exception:
            continue
        if not janelas:
            continue
        n_reg_usados += 1
        print(f"  registo {n_reg_usados}: {os.path.basename(base)}  "
              f"fs={fs}Hz  {len(janelas)} janelas", flush=True)

        for w in janelas:
            n_jan += 1
            a = tre.processar_janela_so_ppg(w.copy(), fs)
            b = pro.processar_janela_so_ppg(w.copy(), fs)
            if a is None or b is None:
                # as duas têm de rejeitar as MESMAS janelas
                if (a is None) != (b is None):
                    print(f"    DIVERGÊNCIA: uma rejeitou e a outra não "
                          f"(treino={'None' if a is None else 'ok'}, "
                          f"produção={'None' if b is None else 'ok'})")
                continue
            n_ok += 1
            if set(a) != set(b):
                print(f"    DIVERGÊNCIA nas chaves: só treino={set(a)-set(b)} "
                      f"| só produção={set(b)-set(a)}")
            for k in set(a) & set(b):
                if a[k] is None or b[k] is None:
                    continue
                vals.setdefault(k, ([], []))
                vals[k][0].append(float(a[k]))
                vals[k][1].append(float(b[k]))

    print(f"\n  janelas processadas: {n_jan}   aceites por ambas: {n_ok}\n")
    if not n_ok:
        sys.exit("nenhuma janela aceite — nada a comparar")

    print(f"  {'característica':<24}{'n':>7}{'dif. máx abs':>15}{'dif. máx rel':>15}{'r':>12}")
    tudo_igual = True
    for k in sorted(vals):
        x = np.array(vals[k][0]); y = np.array(vals[k][1])
        dif = np.abs(x - y)
        rel = dif / np.maximum(np.abs(x), 1e-12)
        r = 1.0 if np.std(x) < 1e-15 and np.std(y) < 1e-15 else float(np.corrcoef(x, y)[0, 1])
        if dif.max() > 1e-9:
            tudo_igual = False
        print(f"  {k:<24}{len(x):>7}{dif.max():>15.3e}{rel.max():>15.3e}{r:>12.5f}")

    print()
    if tudo_igual:
        print("  >>> EQUIVALENTES: diferença máxima abaixo de 1e-9 em todas as características.")
    else:
        print("  >>> DIVERGEM: ver as linhas com diferença acima de 1e-9.")


if __name__ == "__main__":
    main()
