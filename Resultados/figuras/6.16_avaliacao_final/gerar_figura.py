"""Figura 6.16: avaliação da configuração final no conjunto de teste. (a) Sensibilidade por episódio, contra o alarme aleatório com a mesma
estrutura. (b) Tempo de falso alarme por hora da união dos dois detetores, contra o limite de 10 min/h. (c) Valor preditivo positivo. As barras de
erro são intervalos de confiança a 95 % por bootstrap sobre os doentes.

Dados, nesta pasta: avaliacao_tendencia.json.
Uso: python3 gerar_figura.py      (escreve avaliacao_tendencia_tempo_final.png nesta pasta)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
VERM, AZUL, CINZA = '#c0392b', '#1f5fa8', '#7f8c8d'
J = json.load(open(f"{AQUI}/avaliacao_tendencia.json", encoding="utf-8"))
pt, ic = J["ponto"], J["ic95"]
R = {"direcao": J["alarme_aleatorio"]}
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.titlesize": 7.5, "axes.linewidth": 0.6, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6})
fig, ax = plt.subplots(1, 3, figsize=(5.98, 2.4), gridspec_kw=dict(width_ratios=[1.1, 0.8, 1.1]))

def barras(a, xs, vv, cores, ics, fmt):
    for x, v, cc, (lo, hi_) in zip(xs, vv, cores, ics):
        a.bar(x, v, color=cc, width=0.6, alpha=0.9)
        a.errorbar(x, v, yerr=[[v - lo], [hi_ - v]], color="#222222", capsize=2, lw=0.9)
        a.text(x, hi_ + 1, fmt.format(v), fontsize=6.5, va="bottom", ha="center")


a = ax[0]
barras(a, [0, 1], [100 * pt["sens_d"], 100 * pt["sens_s"]], [VERM, AZUL], [[100 * x for x in ic["sens_d"]], [100 * x for x in ic["sens_s"]]], "{:.1f}")
for x, d in ((0, "descida"), (1, "subida")):
    a.plot([x - 0.33, x + 0.33], [100 * R["direcao"][d]["aleat_sens"]] * 2, color=CINZA, ls="--", lw=1.0)
    a.fill_between([x - 0.33, x + 0.33], 100 * R["direcao"][d]["aleat_lo"], 100 * R["direcao"][d]["aleat_hi"], color=CINZA, alpha=0.3, lw=0)
a.plot([], [], color=CINZA, ls="--", lw=1.5, label="random alarm,\nsame structure")
a.legend(loc="upper center", fontsize=6, frameon=False)
a.set_xticks([0, 1]); a.set_xticklabels(["Downward", "Upward"]); a.set_ylim(0, 112); a.set_yticks(range(0, 101, 20)); a.set_ylabel("Episodes detected (%)"); a.set_title("(a) Sensitivity", fontsize=7.5, loc="left")
a = ax[1]
a.bar([0], [pt["fa_min_h"]], color="#34495e", width=0.5, alpha=0.9)
a.errorbar([0], [pt["fa_min_h"]], yerr=[[pt["fa_min_h"] - ic["fa_min_h"][0]], [ic["fa_min_h"][1] - pt["fa_min_h"]]], color="#222222", capsize=2, lw=0.9)
a.axhline(10, color=VERM, ls="--", lw=1.3); a.text(0.42, 10.2, "limit 10 min/h", fontsize=6.5, color=VERM, ha="right", va="bottom")
a.text(0, ic["fa_min_h"][1] + 0.3, f"{pt['fa_min_h']:.1f}", fontsize=6.5, va="bottom", ha="center")
a.set_xlim(-0.5, 0.5); a.set_ylim(0, 14); a.set_ylabel("False-alarm time (min/h)")
a.set_xticks([0]); a.set_xticklabels(["Both\ndirections\n(union)"]); a.set_title("(b) Alarm burden", fontsize=7.5, loc="left")
a = ax[2]
barras(a, [0, 1], [pt["vpp_d"], pt["vpp_s"]], [VERM, AZUL], [ic["vpp_d"], ic["vpp_s"]], "{:.1f}")
a.set_xticks([0, 1]); a.set_xticklabels(["Downward", "Upward"]); a.set_ylim(0, 40); a.set_ylabel("Alarms that are true (%)"); a.set_title("(c) Positive predictive value", fontsize=7.5, loc="left")
fig.tight_layout(pad=0.4, w_pad=0.8); fig.savefig(f"{AQUI}/avaliacao_tendencia_tempo_final.png", dpi=400, bbox_inches="tight", pad_inches=0.03); plt.close(fig)
