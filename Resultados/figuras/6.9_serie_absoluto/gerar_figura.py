"""Figura 6.9: o que o modelo absoluto faz em três situações do conjunto de teste, com a pressão de referência (linha arterial), a estimativa e o valor da
calibração: (a) uma sessão com uma só calibração, (b) a sessão em que o modelo mais ganha à calibração repetida e (c) um troço com várias sessões curtas
e próximas, cada uma com a sua calibração.

Dados, nesta pasta:
  series.csv   painel (a, b ou c), sessao (1, 2, ... dentro do painel), tempo_min, sbp_referencia, sbp_estimada e sbp_calibracao (mmHg), um valor por
               bloco de 15 s. O tempo é o real (minutos desde o início do painel) e a linha parte-se onde faltam mais de 5 minutos.
Uso: python3 gerar_figura.py      (escreve serie_absoluto_pequena.png nesta pasta)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
VERM, AZUL, CINZA = '#c0392b', '#1f5fa8', '#555555'
D_ = pd.read_csv(f"{AQUI}/series.csv")
a, b, c = (D_[D_.painel == k] for k in "abc")


def mae(x):
    """MAE do modelo e MAE de repetir a calibração, em mmHg."""
    return float((x.sbp_estimada - x.sbp_referencia).abs().mean()), float((x.sbp_calibracao - x.sbp_referencia).abs().mean())


(ma_m, ma_a), (mb_m, mb_a), (mc_m, mc_a) = mae(a), mae(b), mae(c)
plt.rcParams.update({"font.size": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.labelsize": 8})
fig = plt.figure(figsize=(5.67, 3.3))
gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1], hspace=0.62, wspace=0.34)
eixos = (fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))
titulos = (f"(a) One calibration: MAE model {ma_m:.1f}, anchor {ma_a:.1f} mmHg",
           f"(b) MAE: model {mb_m:.1f}, anchor {mb_a:.1f} mmHg",
           f"(c) {c.sessao.nunique()} sessions, MAE: model {mc_m:.1f}, anchor {mc_a:.1f}")
for ax, x, titulo in zip(eixos, (a, b, c), titulos):
    for _, g in x.groupby("sessao"):
        tm = g["tempo_min"].values
        corte = np.r_[False, np.diff(tm) > 5.0]           # parte a linha onde faltam mais de 5 minutos

        def partido(v):
            v = np.asarray(v, dtype=float).copy()
            v[corte] = np.nan
            return v
        ax.plot(tm, partido(g.sbp_referencia.values), color=CINZA, lw=0.6)
        ax.plot(tm, partido(g.sbp_estimada.values), color=AZUL, lw=1.1)
        ax.hlines(g.sbp_calibracao.iloc[0], tm.min(), tm.max(), colors=VERM, linestyles="--", lw=0.9)
        ax.plot(tm.min(), g.sbp_calibracao.iloc[0], "v", color=VERM, ms=4.5)
    ax.set_title(titulo, fontsize=7.2)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("SBP (mmHg)")
eixos[0].plot([], [], color=CINZA, lw=0.6, label="reference SBP (arterial line)")
eixos[0].plot([], [], color=AZUL, lw=1.1, label="model estimate")
eixos[0].plot([], [], "v--", color=VERM, lw=0.9, ms=4.5, label="calibration value")
eixos[0].legend(loc="lower left", fontsize=6.3, ncol=3, borderpad=0.3, handlelength=1.6)
fig.savefig(f"{AQUI}/serie_absoluto_pequena.png", dpi=300, bbox_inches="tight")
