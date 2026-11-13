"""Figura 6.19: importância das características (ganho total normalizado) dos dois regressores quantílicos, a descida e a subida. Mostra as 8 maiores de cada
direção, ordenadas pela média das duas.

Dados, nesta pasta: importancia_quant_descida.csv e importancia_quant_subida.csv (a importância de cada uma das 43 características).
Uso: python3 gerar_figura.py      (escreve importancia_compacta_tempo_final.png nesta pasta)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
IMP = {d: pd.read_csv(f"{AQUI}/importancia_quant_{d}.csv", index_col=0).iloc[:, 0] for d in ("descida", "subida")}
TOP = 8
d_, s_ = IMP["descida"], IMP["subida"]
feats = sorted(set(d_.nlargest(TOP).index) | set(s_.nlargest(TOP).index), key=lambda f: (d_[f] + s_[f]) / 2)
y = np.arange(len(feats)); hh = 0.38
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.46, 0.22 * len(feats) + 0.75))
ax.barh(y + hh / 2, [d_[f] for f in feats], height=hh, color="#c0392b", label="Downward")
ax.barh(y - hh / 2, [s_[f] for f in feats], height=hh, color="#2471a3", label="Upward")
ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=6.5)
ax.set_xlabel("Importance (normalised gain)", fontsize=7)
ax.legend(loc="lower right", frameon=False, fontsize=6.5)
ax.grid(axis="x", alpha=0.3, ls=":"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.4)
fig.savefig(f"{AQUI}/importancia_compacta_tempo_final.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
print(len(feats), "características:", ", ".join(reversed(feats)))
