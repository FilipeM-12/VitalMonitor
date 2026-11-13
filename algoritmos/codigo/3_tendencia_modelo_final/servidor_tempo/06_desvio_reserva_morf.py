#!/usr/bin/env python3
"""Igual a 5_servidor_e_carga/verificacao_da_extracao_e_derivacao_contra_o_treino/06_desvio_de_reserva_tendencia.py, para a tabela com as características novas
(3_tendencia_modelo_final/tendencia_tempo_fixo/construir_tabela_ppg_morf.py). Escreve sd_reserva_ppg_morf.json.  venv/bin/python 06_desvio_reserva_morf.py

06 — Desvio de reserva da normalização causal da tendência, para o servidor.

normalizar_causal_calibracao (suavizacao_e_normalizacao_causal.py) divide pelo desvio da janela de calibração da sessão mais 1e-6 e,
quando o desvio é nulo ou indefinido, usa a mediana desse desvio entre todas as sessões. O servidor não tem as outras
sessões, por isso guarda essas medianas como constantes (derivacao_ppg_classificador.TABELAS). Há duas tabelas de treino: a
ppg_apg (01_construir_causal.py), da regressão do delta, e a ppg_puro (11_construir_causal_ppg_puro.py), do
classificador. Aqui corre-se o construtor da tabela pedida a espiar cada chamada da normalização. A tabela vai para uma
pasta temporária e confere-se com a guardada (dados/holdout_<tabela>.csv); os ficheiros do treino não mudam.
Escreve sd_reserva_<tabela>.json.
  venv/bin/python 06_desvio_de_reserva_tendencia.py ppg_apg|ppg_puro|ppg
"""
import json
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = "/home/filipe/Desktop/Tese/algoritmo-ml"
TABELA = "ppg"
C09 = f"{RAIZ}/2026-09-09_normalizacao_causal_tendencia"
SA = f"{RAIZ}/RESULTADOS_RELATORIO/Sem_apg"
CONSTRUTOR = f"{RAIZ}/Tentativa_22_10/tendencia_tempo_fixo/construir_tabela_ppg_morf.py"
GUARDADA = f"{RAIZ}/Tentativa_22_10/tendencia_tempo_fixo/dados/holdout_ppg.csv"
sys.path.insert(0, f"{RAIZ}/03_ABSOLUTO_codigo_e_resultados/CODIGO")
import suavizacao_e_normalizacao_causal as mv  # noqa: E402

ORIGINAL = mv.normalizar_causal_calibracao
RESERVA, CASOS = {}, {}


def espiao(df, cols, n_calib=10, group_col="session_id", sufixo="_z", eps=1e-6):
    presentes = [c for c in cols if c in df.columns]
    sd = df.groupby(group_col, sort=False).head(n_calib).groupby(group_col)[presentes].std()
    for c in presentes:
        r = sd[c].median()
        RESERVA[c] = float(r) if np.isfinite(r) and r >= eps else 1.0
        CASOS[c] = dict(sessoes=int(len(sd)), indefinido=int(sd[c].isna().sum()), nulo=int((sd[c] == 0).sum()),
                        abaixo_de_1e6=int(((sd[c] > 0) & (sd[c] < eps)).sum()))
    return ORIGINAL(df, cols, n_calib=n_calib, group_col=group_col, sufixo=sufixo, eps=eps)


mv.normalizar_causal_calibracao = espiao
with tempfile.TemporaryDirectory(dir=os.environ.get("PASTA_TEMP")) as tmp:
    src = open(CONSTRUTOR, encoding="utf-8").read()
    src, n = re.subn(r"^OUT_DIR = .*$", f"OUT_DIR = {tmp!r}", src, flags=re.M)
    assert n == 1
    src = src.replace('if __name__ == "__main__":', 'if False:')
    ns = {"__name__": "construtor", "__file__": CONSTRUTOR}
    exec(compile(src, CONSTRUTOR, "exec"), ns)
    ns["main"]()
    nova = pd.read_csv(f"{tmp}/holdout_{TABELA}.csv")
guardada = pd.read_csv(GUARDADA)
num = guardada.select_dtypes("number").columns
igual = nova.shape == guardada.shape and np.allclose(nova[num].values, guardada[num].values, equal_nan=True, rtol=0, atol=1e-12)
print(f"tabela igual à guardada: {igual}")
assert igual
for c in RESERVA:
    print(f"  {c:22s} reserva {RESERVA[c]:.10g}   {CASOS[c]}")
json.dump(dict(data="2026-10-14", tabela=TABELA, construtor=CONSTRUTOR, reserva=RESERVA, casos=CASOS),
          open(f"{AQUI}/sd_reserva_ppg_morf.json", "w"), indent=1, ensure_ascii=False)
print("Guardado: sd_reserva_ppg_morf.json")
