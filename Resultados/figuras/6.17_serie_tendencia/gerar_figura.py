"""Figura 6.17: uma sessão do conjunto de teste com (a) a pressão de referência e os episódios de variação de pelo menos 15 mmHg em 5 min, (b) a variação a
5 min das três características mais importantes (cada uma dividida pelo seu desvio padrão na sessão) e (c) as pontuações dos dois detetores, os limiares de decisão
(tracejados) e os alarmes.

Dados, nesta pasta:
  blocos.csv   um bloco de 15 s por linha: tempo_min (desde o início da sessão), sbp_referencia (mmHg; em branco onde a pressão foi rejeitada),
               as três características, que são a variação a 5 min, normalizada, da frequência cardíaca (d_hr_mean_z_5m), da área do pulso (d_area_mean_global_z_5m)
               e do rácio d/a da segunda derivada (d_apg_d_a_global_z_5m) (em branco no início da sessão, até haver histórico suficiente),
               score_descida e score_subida (a pontuação de cada detetor, em mmHg, em branco enquanto não há score) e evento_descida e evento_subida (1 se a
               variação da pressão em 5 min chega a 15 mmHg).
Os alarmes são os blocos em que a pontuação passa o limiar de decisão (11,34 mmHg na descida, 10,92 mmHg na subida), fundidos em corridas (lacunas até 1 min).
Uso: python3 gerar_figura.py      (escreve serie_tendencia_tempo_final.png nesta pasta)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
VERM, AZUL, CINZA = '#c0392b', '#1f5fa8', '#7f8c8d'
BLOCO_SEG = 15
GAP_MERGE, PRE_CREDITO = 4, 8
TAU = {"descida": 11.34, "subida": 10.92}
PRE = PRE_CREDITO
toca = lambda a, e: a[1] >= e[0] - PRE and a[0] <= e[1]
TOP3 = ['d_hr_mean_z_5m', 'd_area_mean_global_z_5m', 'd_apg_d_a_global_z_5m']
NOME = {"d_hr_mean_z_5m": "change in heart rate", "d_area_mean_global_z_5m": "change in pulse area", "d_apg_d_a_global_z_5m": "change in d/a ratio", "d_reflection_index_global_z_5m": "change in reflection index",
        "d_ppg_max_slope_norm_global_z_5m": "change in maximum slope", "d_apg_b_a_global_z_5m": "change in b/a ratio", "vol_hr_mean_z_5m": "heart-rate volatility"}
COR3 = ["#16a085", "#8e44ad", "#d35400"]


def runs(mask, gap=GAP_MERGE):
    # início e fim de cada corrida de True, fundindo corridas separadas por até `gap` blocos
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    cortes = np.where(np.diff(idx) > gap + 1)[0]
    return [(g[0], g[-1]) for g in np.split(idx, cortes + 1)]


B_ = pd.read_csv(f"{AQUI}/blocos.csv")
t = B_.tempo_min.values
gg = B_.rename(columns={"sbp_referencia": "sbp_target"})
gs = B_.rename(columns={"score_descida": "p_descida", "score_subida": "p_subida", "evento_descida": "y_descida", "evento_subida": "y_subida"})
ev_d, ev_s = runs(gs.y_descida.values.astype(bool)), runs(gs.y_subida.values.astype(bool))
m_d, m_s = gs.p_descida.values >= TAU["descida"], gs.p_subida.values >= TAU["subida"]
al_d, al_s = runs(m_d), runs(m_s)
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.titlesize": 7.5, "axes.linewidth": 0.6, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6})
fig, ax = plt.subplots(3, 1, figsize=(5.35, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1, 1.2]))
ax[0].plot(t, gg.sbp_target, color=CINZA, lw=0.7)
for ev_, cor, nome in ((ev_d, VERM, "downward episode"), (ev_s, AZUL, "upward episode")):
    for k, (a0, a1) in enumerate(ev_):
        ax[0].axvspan(t[a0], t[a1] + 0.25, color=cor, alpha=0.18, lw=0, label=nome if k == 0 else None)
ax[0].set_ylabel("SBP (mmHg)"); ax[0].legend(loc="upper right", fontsize=6.5)
ax[0].set_title("(a) Reference pressure and events (change of at least 15 mmHg in 5 min)", fontsize=7.5, loc="left")
for col, cor in zip(TOP3, COR3):
    nome = NOME.get(col, col)
    v = gg[col].values
    ax[1].plot(t, v / np.nanstd(v), color=cor, lw=0.7, label=nome)       # cada variação dividida pelo seu desvio padrão na sessão
ax[1].set_ylabel("Change\n(session SDs)"); ax[1].set_ylim(-4.6, 6.2); ax[1].legend(loc="upper right", fontsize=6.5, ncol=3)
ax[1].set_title("(b) 5-minute change in the three most important features", fontsize=7.5, loc="left")
for dirn, col, al, cor, nome in (("descida", "p_descida", al_d, VERM, "downward"), ("subida", "p_subida", al_s, AZUL, "upward")):
    ax[2].plot(t, gs[col], color=cor, lw=0.7, label=f"{nome} score")
    ax[2].axhline(TAU[dirn], color=cor, ls="--", lw=0.7)
    for k, (a0, a1) in enumerate(al):
        ax[2].axvspan(t[a0], t[a1] + 0.25, ymin=0.92 if dirn == "descida" else 0.84, ymax=0.98 if dirn == "descida" else 0.90, color=cor, lw=0, label=f"{nome} alarm" if k == 0 else None)
off = float(t[int(np.argmax(gs.p_descida.notna().values))])
ax[2].axvspan(0, off, color="#dddddd", lw=0)
ax[2].text(off / 2, 15, "no score yet", ha="center", va="center", fontsize=6, color="#555555", rotation=90)
ax[2].set_ylabel("Score (mmHg)"); ax[2].set_xlabel("Time since the start of the session (min)")
ax[2].set_ylim(0, 30); ax[2].legend(loc="upper right", bbox_to_anchor=(1.0, 0.83), fontsize=6.5, ncol=2)      # logo abaixo dos alarmes, acima dos scores
ax[2].set_title("(c) Model scores, decision thresholds (dashed) and alarms", fontsize=7.5, loc="left")
fig.tight_layout(pad=0.4, h_pad=0.6); fig.savefig(f"{AQUI}/serie_tendencia_tempo_final.png", dpi=400, bbox_inches="tight", pad_inches=0.03); plt.close(fig)


