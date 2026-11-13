"""Figura F.1: sensibilidade por episódio, a custo igualado de falso alarme (4 a 10 minutos por hora da união), sem e com o portão de estabilidade,
nas duas direções, no conjunto de teste.

Dados, nesta pasta: portao_estabilidade.csv (direcao, serie, falso_alarme_min_por_h, sensibilidade).
Uso: python3 gerar_figura.py      (escreve portao_tempo_final.png nesta pasta)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
P = pd.read_csv(f"{AQUI}/portao_estabilidade.csv")
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.titlesize": 7.5, "axes.linewidth": 0.6})
fig, ax = plt.subplots(1, 2, figsize=(4.72, 2.1), sharey=True)
for a_, d, cor, nome in ((ax[0], "descida", "#c0392b", "Downward"), (ax[1], "subida", "#1f5fa8", "Upward")):
    sem, com = P[(P.direcao == d) & (P.serie == "sem_portao")], P[(P.direcao == d) & (P.serie == "com_portao")]
    a_.plot(sem.falso_alarme_min_por_h, sem.sensibilidade, "o-", color="#666666", lw=1.1, ms=2.8, label="no gate")
    a_.plot(com.falso_alarme_min_por_h, com.sensibilidade, "s-", color=cor, lw=1.2, ms=2.8, label="with gate")
    a_.set_title(nome, fontsize=7.5); a_.set_xlabel("False-alarm minutes per hour"); a_.grid(ls=":", alpha=0.6); a_.legend(loc="lower right", fontsize=6.5)
ax[0].set_ylabel("Sensitivity per episode")
fig.tight_layout(pad=0.4, w_pad=0.8); fig.savefig(f"{AQUI}/portao_tempo_final.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
print("portao_tempo_final.png")
