"""Figura 6.11: as regras da avaliação por episódio, num troço de 30 min do conjunto de teste: (a) a pressão de referência e a variação de pelo menos 15 mmHg
em 5 min, (b) os blocos de evento fundidos em episódios (lacunas até 1 min) e o crédito de 2 min de aviso prévio e (c) os alarmes dos detetores de descida e de subida,
verdadeiros ou falsos conforme toquem num episódio.

Dados, nesta pasta:
  blocos.csv   um bloco de 15 s por linha (120 blocos): sbp_referencia (mmHg; em branco onde a pressão foi rejeitada), evento_descida e evento_subida (1 se a
               variação em 5 min chega a 15 mmHg) e alarme_descida e alarme_subida (1 se a pontuação do modelo passa o limiar de decisão: 11,34 mmHg na descida e
               10,92 mmHg na subida).
Uso: python3 gerar_figura.py      (escreve regras_episodio_tempo_baixa2_final.png nesta pasta)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

AQUI = os.path.dirname(os.path.abspath(__file__))
BLOCO_SEG, GAP_MERGE, PRE_CREDITO = 15, 4, 8
VERM, AZUL, CINZA, ROXO = '#c0392b', '#1f5fa8', '#555555', '#8e44ad'


def runs(mask, gap=GAP_MERGE):
    # início e fim de cada corrida de True, fundindo corridas separadas por até `gap` blocos
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    cortes = np.where(np.diff(idx) > gap + 1)[0]
    return [(g[0], g[-1]) for g in np.split(idx, cortes + 1)]


B_ = pd.read_csv(f"{AQUI}/blocos.csv")
sbp = B_.sbp_referencia.values
ev_b = {"d": B_.evento_descida.values.astype(bool), "s": B_.evento_subida.values.astype(bool)}
al_b = {"d": B_.alarme_descida.values.astype(bool), "s": B_.alarme_subida.values.astype(bool)}
toca = lambda a, e: a[1] >= e[0] - PRE_CREDITO and a[0] <= e[1]          # um alarme conta se acaba depois do início do episódio menos o crédito e começa antes do fim
NB, W = len(B_), 20
ep = {k: runs(ev_b[k]) for k in "ds"}; Al = {k: runs(al_b[k]) for k in "ds"}
tm = np.arange(NB) * BLOCO_SEG / 60
COR_EV = {"d": VERM, "s": ROXO}
det_ep = [(k, e) for k in "ds" for e in ep[k] if any(toca(a, e) for a in Al[k])]
k_det, e_det = max(det_ep, key=lambda ke: ke[1][1] - ke[1][0])
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.titlesize": 7.5, "axes.linewidth": 0.6, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6})
fig, ax = plt.subplots(3, 1, figsize=(5.7, 2.5), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 0.38, 0.45]))
ax[0].plot(tm, sbp, color=CINZA, lw=0.9)
i1 = e_det[0]
ref = sbp[i1 - W]
t_ref = tm[i1] - 5
ax[0].annotate("", xy=(tm[i1], sbp[i1]), xytext=(tm[i1], ref), arrowprops=dict(arrowstyle="<->", color=COR_EV[k_det]))
ax[0].plot([max(t_ref, tm[0]), tm[i1]], [ref] * 2, ":", color=COR_EV[k_det], lw=1)
ax[0].set_ylim(np.nanmin(sbp) - 4, max(np.nanmax(sbp), ref) + 4)
if sbp[i1] > ref:      # subida: o texto fica à direita da seta, abaixo da curva
    ax[0].text(tm[i1] + 0.4, (sbp[i1] + ref) / 2, "$|\\Delta| \\geq 15$ mmHg\nover $W = 5$ min", color=COR_EV[k_det], fontsize=6.5, va="center", ha="left")
else:                  # descida: à esquerda da seta, abaixo da curva
    ax[0].text(tm[i1] - 0.3, sbp[i1] - 4.5, "$|\\Delta| \\geq 15$ mmHg\nover $W = 5$ min", color=COR_EV[k_det], fontsize=6.5, va="center", ha="right")
ax[0].set_ylabel("SBP (mmHg)")
ax[0].set_title("(a) Reference pressure (test set)", fontsize=7.5, loc="left")
for k, nome in (("d", "drop"), ("s", "rise")):
    ax[1].bar(tm[ev_b[k]], 0.5, width=0.25, bottom=0.25, color=COR_EV[k], alpha=0.35, align="edge")
    for n, (e0, e1) in enumerate(ep[k]):
        ax[1].axvspan(tm[e0], tm[e1] + 0.25, ymin=0.15, ymax=0.85, fill=False, ec=COR_EV[k], lw=1.0, label=f"{nome} episode" if n == 0 else None)
        if not any(toca(a, (e0, e1)) for a in Al[k]):
            ax[1].text(tm[e0] - 0.3 if tm[e1] > 26 else tm[e1] + 0.4, 0.5, "missed", fontsize=6.5, color=COR_EV[k], va="center", ha="right" if tm[e1] > 26 else "left")
ax[1].axvspan(tm[e_det[0]] - PRE_CREDITO * BLOCO_SEG / 60, tm[e_det[0]], ymin=0.15, ymax=0.85, color=COR_EV[k_det], alpha=0.12, hatch="///", ec=COR_EV[k_det], lw=0, label="2 min early-warning credit")
ax[1].set_yticks([]); ax[1].set_ylim(0, 1)
ax[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.5, ncol=1, frameon=False)
ax[1].set_title("(b) Event blocks merged into episodes (gaps up to 1 min)", fontsize=7.5, loc="left")
for k in "ds":
    for (a0, a1) in Al[k]:
        cor = AZUL if any(toca((a0, a1), e) for e in ep[k]) else "#e67e22"
        bl = np.where(al_b[k][a0:a1 + 1])[0] + a0
        ax[2].bar(tm[bl], 0.36, width=0.25, bottom=0.32, color=cor, alpha=0.55, align="edge")
        ax[2].axvspan(tm[a0], tm[a1] + 0.25, ymin=0.27, ymax=0.73, fill=False, ec=cor, lw=1.0)
ax[2].legend(handles=[Patch(fc=AZUL, alpha=0.55, ec=AZUL, label="true alarm"), Patch(fc="#e67e22", alpha=0.55, ec="#e67e22", label="false alarm")], loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.5, frameon=False)
for x in (tm[e_det[0]] - PRE_CREDITO * BLOCO_SEG / 60, tm[e_det[0]]):
    for a_ in (ax[1], ax[2]):
        a_.axvline(x, color=COR_EV[k_det], ls="--", lw=0.9, alpha=0.8)
ax[2].set_yticks([]); ax[2].set_ylim(0, 1)
ax[2].set_xlabel("Time (min)")
ax[2].set_title("(c) Alarms (drop and rise detectors)", fontsize=7.5, loc="left")
fig.tight_layout(pad=0.4, h_pad=0.5)
fig.savefig(f"{AQUI}/regras_episodio_tempo_baixa2_final.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("regras_episodio_tempo_baixa2_final.png")
