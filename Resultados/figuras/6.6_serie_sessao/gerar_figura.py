"""Figura 6.6: a pressão sistólica de referência (linha arterial) e a estimativa do modelo absoluto numa sessão do conjunto de teste, com o valor da
calibração (braçadeira) no início.

Dados, nesta pasta:
  serie_sessao.csv   tempo_min (desde a calibração), sbp_referencia e sbp_estimada (mmHg), um valor por bloco de 15 s; o primeiro valor a seguir
                     a uma lacuna de mais de 1 min sem estimativa fica em branco, para a linha se partir aí.
  calibracao.json    o valor da calibração (mmHg), a linha tracejada.
Uso: python3 gerar_figura.py      (escreve bp_serie_sessao_final.png nesta pasta)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
d_ = pd.read_csv(f"{AQUI}/serie_sessao.csv")
tm, ref, est = d_.tempo_min.values, d_.sbp_referencia.values, d_.sbp_estimada.values
calib = json.load(open(f"{AQUI}/calibracao.json"))["sbp_calibracao_mmHg"]
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(5.35, 2.3))
ax.plot(tm, ref, color="#555555", lw=0.6, label="reference SBP (arterial line)")
ax.plot(tm, est, color="#1f5fa8", lw=1.1, label="model estimate")
ax.axhline(calib, color="#c0392b", ls="--", lw=0.9, label="calibration value (cuff)")
ax.plot([0], [calib], "D", color="#c0392b", ms=4, mec="white", mew=0.6, zorder=5)
ax.set_xlabel("Time since calibration (min)")
ax.set_ylabel("SBP (mmHg)")
ax.grid(alpha=0.3, ls=":")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.5, ncol=3, frameon=False, columnspacing=1.0, handlelength=1.6)
fig.tight_layout()
fig.savefig(f"{AQUI}/bp_serie_sessao_final.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
