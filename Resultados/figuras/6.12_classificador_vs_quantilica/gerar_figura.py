"""Figura 6.12: sensibilidade por episódio do classificador e do regressor quantílico a 4, 6, 8 e 10 minutos de falso alarme por hora da
união dos dois detetores, no conjunto de desenvolvimento, com previsões fora da dobra (a), e no de teste (b).

Dados, nesta pasta: comparacao_niveis_carga_dev.csv (desenvolvimento) e comparacao_niveis_carga.csv (teste).
Uso: python3 gerar_figura.py      (escreve classificador_vs_quantilica_tempo_larga.png nesta pasta)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
VERM, AZUL = '#c0392b', '#1f5fa8'
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6})
teste = pd.read_csv(f"{AQUI}/comparacao_niveis_carga.csv")
dev = pd.read_csv(f"{AQUI}/comparacao_niveis_carga_dev.csv")
fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.2), sharey=True)
for ax, d, pref, titulo in ((axs[0], dev, "dev_", "(a) Development set, out-of-fold"), (axs[1], teste, "", "(b) Test set")):
    for dire, cor, nome in (("d", VERM, "downward"), ("s", AZUL, "upward")):
        c = f"{pref}sens_{dire}"
        for modelo, estilo, mk, cheio, rot in (("quant", "-", "o", True, "quantile regression"), ("clf", "--", "s", False, "binary classifier")):
            x = d[d.modelo == modelo]
            ax.plot(x.alvo_min_h, x[c], estilo, marker=mk, color=cor, lw=1.3, ms=4.2, mfc=cor if cheio else "white", mew=1.1, label=f"{nome}, {rot}")
    ax.set_title(titulo, fontsize=8.5, pad=3)
    ax.set_xlabel("Minutes of false alarm per hour (union)", fontsize=8, labelpad=2)
    ax.set_xticks([4, 6, 8, 10])
    ax.tick_params(labelsize=7.5, pad=1.5)
axs[0].set_ylabel("Sensitivity per episode", fontsize=8, labelpad=2)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, frameon=False, handlelength=2.2, columnspacing=1.6, labelspacing=0.3)
fig.tight_layout(pad=0.3, rect=(0, 0.17, 1, 1))
fig.savefig(f"{AQUI}/classificador_vs_quantilica_tempo_larga.png", dpi=400)
print("classificador_vs_quantilica_tempo_larga.png")
