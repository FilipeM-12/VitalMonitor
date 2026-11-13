"""Figura 6.7: dispersão das 20 tentativas do Optuna contra a incerteza do MAE. Cada ponto é o MAE de uma tentativa (validação cruzada em 3 dobras por doente);
a linha tracejada é a mediana e a barra laranja o intervalo de confiança a 95 % por doente.

Dados, nesta pasta: trials.csv (uma linha por tentativa: number e value, o MAE) e planalto_incerteza.json (a largura do intervalo de confiança).
Uso: python3 gerar_figura.py      (escreve dispersao_trials_baixa.png nesta pasta)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
N_FOLDS = 3
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.8})
AzulCor, Laranja = "#2b6cb0", "#c05621"
trials = pd.read_csv(f"{AQUI}/trials.csv")
inc = json.load(open(f"{AQUI}/planalto_incerteza.json"))
y = trials["value"].values
med = float(np.median(y)); largura = inc["ic95_largura"]
fig, ax = plt.subplots(figsize=(3.1, 1.9))
ax.scatter(trials["number"], y, s=14, color=AzulCor, zorder=3, label=f"Trials (CV MAE, {len(y)})")
ax.axhline(med, color=AzulCor, linestyle="--", linewidth=0.9, alpha=0.6, label=f"Median ({med:.2f} mmHg)")
ax.errorbar([-2.5], [med], yerr=[[largura / 2], [largura / 2]], fmt="o", color=Laranja, capsize=4, markersize=4, linewidth=1.5,
            label=f"95% CI, subject level ({largura:.2f} mmHg)")
ax.set_xlim(-4, len(y))
ax.set_xlabel("Trial number (Optuna)")
ax.set_ylabel(f"MAE (mmHg, {N_FOLDS}-fold CV)")
ax.legend(loc="upper right", framealpha=0.9, borderpad=0.3, handletextpad=0.4, labelspacing=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{AQUI}/dispersao_trials_baixa.png", dpi=300)
print("dispersao_trials_baixa.png")
