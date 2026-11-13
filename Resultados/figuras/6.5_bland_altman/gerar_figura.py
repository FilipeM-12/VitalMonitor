"""Figura 6.5: análise de Bland-Altman da estimação absoluta no conjunto de teste. Diferença entre a estimativa e a referência contra a média das duas,
em mapa de densidade (escala logarítmica), com o viés e os limites de concordância (a vermelho) e a reta de regressão da diferença sobre a média.

Dados, nesta pasta:
  bland_altman_celulas.csv  as células do mapa (x e y são o centro da célula; contagem é o número de janelas nela);
  bland_altman.json         o viés, os limites, a reta de regressão, a forma das células e os limites dos eixos.
Uso: python3 gerar_figura.py      (escreve bland_altman_final.png nesta pasta)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

AQUI = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(f"{AQUI}/bland_altman.json"))
cel = pd.read_csv(f"{AQUI}/bland_altman_celulas.csv")

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(4.1, 2.55))
# o mapa de densidade (o que ax.hexbin desenha): um hexágono por célula, com a cor dada pela contagem em escala logarítmica
hb = PolyCollection([np.array(res["forma_da_celula"])], edgecolors="face", linewidths=0.2, offsets=cel[["x", "y"]].values,
                    offset_transform=mtransforms.AffineDeltaTransform(ax.transData))
hb.set_array(cel.contagem.values)
hb.set_cmap("Blues")
hb.set_norm(LogNorm())
ax.add_collection(hb, autolim=False)
ax.set_xlim(*res["xlim"])
ax.set_ylim(*res["ylim"])
xmax = ax.get_xlim()[1]
for y, txt, ls, va in ((res['vies'], f"bias {res['vies']:+.1f}", "-", "bottom"),
                       (res['limite_superior'], f"+1.96 SD: {res['limite_superior']:+.1f}", "--", "bottom"),
                       (res['limite_inferior'], f"-1.96 SD: {res['limite_inferior']:+.1f}", "--", "top")):
    ax.axhline(y, color="#c0392b", ls=ls, lw=0.9)
    ax.text(xmax - 2, y, txt, va=va, ha="right", fontsize=6.5, color="#c0392b")
ax.plot(res["regressao"]["x"], res["regressao"]["y"], color="#222222", lw=0.9, label=f"regression, slope {res['declive']:+.2f}")
ax.set_xlabel("Mean of estimate and reference SBP (mmHg)")
ax.set_ylabel("Estimate - reference (mmHg)")
ax.legend(loc="upper left", fontsize=6.5)
ax.grid(alpha=0.3, ls=":")
cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.045)
cb.set_label("windows (log scale)", fontsize=6.5); cb.ax.tick_params(labelsize=6.5)
fig.tight_layout(pad=0.4)
fig.savefig(f"{AQUI}/bland_altman_final.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
print("bland_altman_final.png")
