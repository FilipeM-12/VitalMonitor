"""Figura 6.13: AUC no conjunto de desenvolvimento, com previsões fora da dobra, em função da janela de tempo W (2, 3, 5 e 10 min), nas duas direções.

Dados, nesta pasta: resultados_janela_dev.csv (uma linha por janela e direção).
Uso: python3 gerar_figura.py      (escreve auc_janela_dev_larga.png nesta pasta)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
VERM, AZUL = '#c0392b', '#1f5fa8'
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6})
r = pd.read_csv(f"{AQUI}/resultados_janela_dev.csv")
d, s = r[r.direcao == 'descida'].sort_values('W'), r[r.direcao == 'subida'].sort_values('W')
fig, ax = plt.subplots(figsize=(3.9, 1.7))
ax.axvspan(4.6, 5.4, color="#f1c40f", alpha=0.25, lw=0, label="adopted window")
ax.plot(d.W, d.auc_dev, "o-", color=VERM, lw=1.5, ms=4, label="downward")
ax.plot(s.W, s.auc_dev, "s-", color=AZUL, lw=1.5, ms=4, label="upward")
# posição de cada valor (acima 'a' ou abaixo 'b' do ponto), escolhida para não tocar nas linhas
POS_D = {2: "a", 3: "b", 5: "b", 10: "b"}
POS_S = {2: "a", 3: "b", 5: "a", 10: "a"}
for x, a in zip(d.W, d.auc_dev):
    up = POS_D[int(x)] == "a"
    ax.text(x, a + (0.007 if up else -0.007), f"{a:.3f}", color=VERM, ha="center", va="bottom" if up else "top", fontsize=7.5)
for x, b in zip(s.W, s.auc_dev):
    up = POS_S[int(x)] == "a"
    ax.text(x, b + (0.007 if up else -0.007), f"{b:.3f}", color=AZUL, ha="center", va="bottom" if up else "top", fontsize=7.5)
ax.set_xticks(list(d.W))
ax.set_xlim(1.5, 10.5)
ax.set_xlabel("Time window W (min)", fontsize=8.5, labelpad=1)
ax.set_ylabel("AUC", fontsize=8.5, labelpad=1)
ax.set_ylim(0.73, 0.88)
ax.tick_params(labelsize=8, pad=1.5)
ax.legend(loc="upper right", fontsize=7, ncol=1, borderpad=0.4, labelspacing=0.3, handlelength=1.6)
fig.tight_layout(pad=0.3)
fig.savefig(f"{AQUI}/auc_janela_dev_larga.png", dpi=400)
print("auc_janela_dev_larga.png")
