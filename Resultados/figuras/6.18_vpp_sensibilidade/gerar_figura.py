"""Figura 6.18: valor preditivo positivo (VPP) em função da sensibilidade por episódio, no conjunto de teste, para a descida e a subida, com o
ponto adotado de cada direção.

Dados, nesta pasta: vpp_curva.json, com a curva de cada direção (pares sensibilidade e VPP, em fração) e o ponto adotado.
Uso: python3 gerar_figura.py      (escreve vpp_sensibilidade_tempo_final.png nesta pasta)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
VERM, AZUL = '#c0392b', '#1f5fa8'
J = json.load(open(f"{AQUI}/vpp_curva.json"))
curvas, adotado = J["curvas"], J["adotado"]
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6})
fig, ax = plt.subplots(figsize=(3.78, 2.55))
for d, cor, nome in (("descida", VERM, "downward"), ("subida", AZUL, "upward")):
    x, y = zip(*curvas[d])
    ax.plot(x, 100 * np.array(y), color=cor, lw=1.4, label=nome)
    ax.scatter([adotado[d][0]], [100 * adotado[d][1]], marker="*", s=70, color=cor, edgecolor="black", linewidth=0.6, zorder=5, label=f"{nome}, adopted point")
ax.axhline(50, color="#888888", ls=":", lw=0.9)
ax.set_ylim(0, 55); ax.set_xlim(-0.02, 1.02)
ax.set_xlabel("Episode sensitivity"); ax.set_ylabel("Positive predictive value (%)")
ax.legend(loc="upper right", fontsize=6)
fig.tight_layout(pad=0.4)
fig.savefig(f"{AQUI}/vpp_sensibilidade_tempo_final.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
