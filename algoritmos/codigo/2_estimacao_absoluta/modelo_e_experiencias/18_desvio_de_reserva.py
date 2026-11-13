#!/usr/bin/env python3
"""18 — Desvio de reserva da normalização causal, para o servidor.

nucleo.dados.normalizar_por_calibracao divide pelo desvio da janela de calibração da sessão mais 1e-6. Quando esse
desvio é nulo ou indefinido, usa a mediana, entre todas as sessões da tabela, do mesmo desvio. O servidor não tem as
outras sessões, por isso precisa dessas medianas como constantes (comum.SD_RESERVA). Aqui corre-se a derivação do treino
(nucleo.dados.derivar) a espiar cada chamada de normalizar_por_calibracao, e guardam-se as medianas e quantas sessões
caem em cada caso (desvio indefinido, nulo, ou abaixo de 1e-6 sem ser nulo).
Escreve resultados/tabelas/sd_reserva_calibracao.json.
  venv/bin/python 18_desvio_de_reserva.py
"""
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from nucleo import config as C
from nucleo import dados as D

SAIDA = f"{C.DIR_RESULTADOS}/tabelas/sd_reserva_calibracao.json"
ORIGINAL = D.normalizar_por_calibracao
RESERVA, CASOS = {}, {}


def espiao(df, cols, grupo, n_calib=C.N_BLOCOS_CALIBRACAO, sufixo="_z", eps=1e-6):
    presentes = [c for c in cols if c in df.columns]
    sd = df.groupby(grupo, sort=False).head(n_calib).groupby(grupo)[presentes].std()
    for c in presentes:
        r = sd[c].median()
        RESERVA[c] = float(r) if np.isfinite(r) and r >= eps else 1.0
        CASOS[c] = dict(sessoes=int(len(sd)), indefinido=int(sd[c].isna().sum()), nulo=int((sd[c] == 0).sum()),
                        abaixo_de_1e6=int(((sd[c] > 0) & (sd[c] < eps)).sum()))
    return ORIGINAL(df, cols, grupo, n_calib, sufixo, eps)


D.normalizar_por_calibracao = espiao
tabela = D.derivar(C.VARIANTES["baseline"], "causal", C.MIN_BLOCOS_SESSAO)
guardada = D.carregar_tabela(C.VARIANTES["baseline"], "causal", C.MIN_BLOCOS_SESSAO)
igual = (tabela.shape == guardada.shape
         and np.allclose(tabela[list(C.FEATURES_V3)].values, guardada[list(C.FEATURES_V3)].values, equal_nan=True))
print(f"derivação igual à cache do treino: {igual}")
assert igual

for c in RESERVA:
    print(f"  {c:22s} reserva {RESERVA[c]:.10g}   {CASOS[c]}")
os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
json.dump(dict(data="2026-10-14", reserva=RESERVA, casos=CASOS), open(SAIDA, "w"), indent=1, ensure_ascii=False)
print(f"Guardado: {SAIDA}")
