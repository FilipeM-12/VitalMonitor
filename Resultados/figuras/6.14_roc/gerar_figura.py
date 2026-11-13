"""Figura 6.14: curvas ROC por bloco do regressor quantílico no conjunto de teste, para a descida e a subida, com o AUC na legenda.

Dados, nesta pasta: roc_curvas.csv, com os pontos de cada curva (direcao, fpr, tpr). O AUC calcula-se desses pontos pela regra do trapézio, que é o que
o scikit-learn faz.
Uso: python3 gerar_figura.py      (escreve roc_tempo_larga.png nesta pasta)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
VERM, AZUL = '#c0392b', '#1f5fa8'
roc = pd.read_csv(f"{AQUI}/roc_curvas.csv")
plt.rcParams.update({"font.size": 7.5, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6})
fig, ax = plt.subplots(figsize=(2.7, 2.1))
for d, cor, nome in (("descida", VERM, "downward"), ("subida", AZUL, "upward")):
    c = roc[roc.direcao == d]
    fpr, tpr = c.fpr.values, c.tpr.values
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    print(d, "AUC", round(auc, 4), "pontos", len(c))
    ax.plot(fpr, tpr, color=cor, lw=1.3, label=f"{nome}, AUC = {auc:.3f}")
ax.plot([0, 1], [0, 1], ":", color="#888888", lw=0.9)
ax.set_xlabel("1 − specificity", fontsize=8, labelpad=1.5)
ax.set_ylabel("Sensitivity", fontsize=8, labelpad=1.5)
ax.set_title("Quantile regression on $\\Delta$, test set", fontsize=8, pad=2)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
ax.tick_params(labelsize=7.5, pad=1.5)
ax.legend(loc="lower right", fontsize=7, frameon=True, borderpad=0.4, labelspacing=0.3, handlelength=1.5)
fig.tight_layout(pad=0.3)
fig.savefig(f"{AQUI}/roc_tempo_larga.png", dpi=400)
print("roc_tempo_larga.png")
