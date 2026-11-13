"""Figura 6.8: autocorrelação do resíduo do modelo (estimativa menos referência) em função do desfasamento, até 60 minutos. A correlação calcula-se entre
blocos de 15 s da mesma sessão que distam o desfasamento indicado.

Dados, nesta pasta: acf_residuo.csv (lag_15s é o desfasamento em blocos de 15 s, minutos o desfasamento em minutos e autocorrelacao o valor).
Uso: python3 gerar_figura.py      (escreve acf_residuo_baixa.png nesta pasta)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.8})
AzulCor, Laranja = "#2b6cb0", "#c05621"
a = pd.read_csv(f"{AQUI}/acf_residuo.csv")
minutos, rho = a.minutos.values, a.autocorrelacao.values
fig, ax = plt.subplots(figsize=(3.1, 1.9))
ax.axhline(0, color="#888888", linewidth=0.7)
ax.plot(minutos, rho, color=AzulCor, linewidth=1.5)
ax.plot([minutos[-1]], [rho[-1]], "o", color=Laranja, markersize=4.5, zorder=3)
ax.annotate(f"{rho[-1]:.2f} at 60 min", xy=(minutos[-1], rho[-1]), xytext=(-5, 8), textcoords="offset points", ha="right", fontsize=7, color=Laranja)
ax.set_xlabel("Lag (minutes)")
ax.set_ylabel("Residual autocorrelation")
ax.set_xlim(0, 62); ax.set_ylim(-0.05, 1.0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{AQUI}/acf_residuo_baixa.png", dpi=300)
print("acf_residuo_baixa.png; autocorrelação a 60 min =", round(float(rho[-1]), 4))
