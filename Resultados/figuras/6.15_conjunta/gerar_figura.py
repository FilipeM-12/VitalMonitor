"""Figura 6.15: escolha conjunta dos dois limiares no conjunto de teste. Desenha os pares de limiares com ciclo de serviço até 30 %
(minutos de falso alarme por hora da união contra a pior das duas sensibilidades) e o ponto adotado (tau_d = 11,34 e tau_s = 10,92).

Dados, nesta pasta: grelha_teste.csv (um par de limiares por linha, com as sensibilidades, o tempo de falso alarme e o ciclo de serviço da união).
Uso: python3 gerar_figura.py      (escreve conjunta_tempo_final.png nesta pasta)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
g = pd.read_csv(f"{AQUI}/grelha_teste.csv")
q = g[g.u_duty <= 0.30].copy()
q["pior"] = np.minimum(q.sens_d, q.sens_s)
a = g[(g.tau_d == 11.34) & (g.tau_s == 10.92)].iloc[0]
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.titlesize": 7.5, "axes.linewidth": 0.6, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6})
fig, ax = plt.subplots(figsize=(3.66, 2.5))
ax.scatter(q.u_FA_min_h, q.pior, s=7, color="#b5b5b5", label="threshold pairs (duty $\\leq$ 30 %)", zorder=2)
ax.scatter([a.u_FA_min_h], [min(a.sens_d, a.sens_s)], marker="*", s=120, color="#0e7a5a", edgecolor="black", linewidth=0.7, label="adopted point", zorder=3)
ax.set_xlim(0, 14)
ax.set_xlabel("False-alarm minutes per hour (union of both detectors)")
ax.set_ylabel("Worse of the two sensitivities")
ax.set_title("Joint choice of the two thresholds, test set", loc="left")
ax.legend(loc="upper left", frameon=True)
fig.tight_layout(pad=0.4)
fig.savefig(f"{AQUI}/conjunta_tempo_final.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
print(len(q), "pares; adotado", round(a.u_FA_min_h, 3), round(min(a.sens_d, a.sens_s), 3))
