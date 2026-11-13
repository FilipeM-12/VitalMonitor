#!/usr/bin/env python3
"""Quanto tempo de relógio cobrem os «20 blocos válidos» do rótulo (e os 8 e 40 das variações a 2 e 10 min) na tabela
única da tendência. O rótulo e as variações contam linhas da tabela (blocos válidos), não minutos: um bloco
rejeitado deixa um buraco em bloco_id e a janela cobre mais tempo. Escreve janela_em_blocos.json.
  python janela_em_blocos.py"""
import json
import os

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
D = "/home/filipe/Desktop/Tese/algoritmo-ml/2026-09-09_normalizacao_causal_tendencia/dados"
BLOCO_S = 15
d = pd.concat([pd.read_csv(f"{D}/{p}_ppg.csv", usecols=["session_id", "bloco_id"]).assign(parte=p) for p in ("dev", "holdout")])
d = d.sort_values(["session_id", "bloco_id"])
saida = {}
for n in (8, 20, 40):
    vol = d.groupby("session_id").bloco_id.diff(n).dropna() * BLOCO_S / 60          # minutos que cobrem n linhas
    saida[str(n)] = dict(linhas=int(len(vol)), minutos_nominais=n * BLOCO_S / 60, exatas_pct=round(100 * float((vol == n * BLOCO_S / 60).mean()), 1),
                         acima_pct=round(100 * float((vol > n * BLOCO_S / 60).mean()), 1), p95=round(float(vol.quantile(0.95)), 2),
                         max=round(float(vol.max()), 2))
json.dump(saida, open(f"{AQUI}/janela_em_blocos.json", "w"), indent=1)
print(json.dumps(saida, indent=1))
