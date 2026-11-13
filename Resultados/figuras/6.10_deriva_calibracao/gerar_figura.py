"""Figura 6.10: (a) erro do modelo e da âncora (repetir o último valor da braçadeira) em função do tempo desde a calibração; (b) MAE do modelo e da âncora de
cada variante contra a idade mediana da calibração.

Dados, nesta pasta:
  ancora_horizonte.csv        o MAE da âncora e do modelo por faixa de tempo desde a calibração (painel a);
  ganho_contra_ancora.json    o ganho do modelo sobre a âncora e o seu intervalo de confiança a 95 % (título do painel a);
  variantes.csv               por variante: doentes, janelas, idade mediana da calibração, MAE do modelo e MAE da âncora (painel b).
Uso: python3 gerar_figura.py      (escreve figura_deriva_calibracao_pequena.pdf e .png nesta pasta)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
g = pd.read_csv(f"{AQUI}/ancora_horizonte.csv")
ga = json.load(open(f"{AQUI}/ganho_contra_ancora.json"))["todos"]
pts = pd.read_csv(f"{AQUI}/variantes.csv").to_dict("records")

plt.rcParams.update({"font.size": 7.5})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.67, 2.3))
x = np.arange(len(g))
ax1.plot(x, g["ancora"], "o-", lw=1.4, ms=4.5, color="#555555", label="Anchor: repeat the last cuff value")
ax1.plot(x, g["modelo"], "s--", lw=1.4, ms=4, color="#c0392b", label="XGBoost model")
for xi, (a, m) in enumerate(zip(g["ancora"], g["modelo"])):
    ax1.annotate(f"{a - m:+.2f}", (xi, max(a, m) + 0.95), ha="center", fontsize=6, color="#c0392b")
ax1.set_xticks(x)
ax1.set_xticklabels(g["faixa_min"])
ax1.set_title(f"(a) Model against the anchor: gain {ga['ganho']:+.2f} mmHg\n(95% CI {ga['ic95'][0]:+.2f} to "
              f"{ga['ic95'][1]:+.2f}), labels: anchor - model", fontsize=7)
ax1.set_xlabel("Time since calibration (min)")
ax1.set_ylabel("MAE of systolic BP (mmHg)")
ax1.legend(fontsize=6, loc="upper left")
ax1.grid(alpha=0.3, ls=":")
ax1.set_ylim(0, 20)

xs = np.array([p["idade"] for p in pts]); ys = np.array([p["modelo"] for p in pts])
ANEL, vistos = {}, {}                                  # âncoras iguais (mesmas janelas e calibração) ficam concêntricas
for p in reversed(pts):
    chave = (round(p["idade"], 1), round(p["ancora"], 2))
    ANEL[p["v"]] = 90 + 70 * vistos.get(chave, 0); vistos[chave] = vistos.get(chave, 0) + 1
for p in pts:
    ax2.scatter(p["idade"], p["ancora"], s=ANEL[p["v"]], facecolor="none", edgecolor=p["cor"], lw=1.2, zorder=2)
    ax2.scatter(p["idade"], p["modelo"], s=26, color=p["cor"], zorder=3,
                label=f"{p['nome']}: {p['n_suj']} subj., {p['n'] // 1000}k windows, gain {p['ancora'] - p['modelo']:+.2f}")
k = np.polyfit(xs, ys, 1)
xx = np.linspace(xs.min() * 0.8, xs.max() * 1.25, 50)
ax2.plot(xx, np.polyval(k, xx), "--", color="#888888", lw=1.2, zorder=1)
h, l = ax2.get_legend_handles_labels()                 # legenda por baixo dos dois painéis, em duas colunas
fig.legend(h, l, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
           title="Panel (b): model (dot) and anchor (ring) of each variant", title_fontsize=6)
# ampliação das três variantes com a mesma coorte, que no eixo inteiro ficam umas por cima das outras
perto = [p for p in pts if p["idade"] > 60]
axz = ax2.inset_axes([0.60, 0.06, 0.37, 0.30])
for p in perto:
    axz.scatter(p["idade"], p["ancora"], s=ANEL[p["v"]], facecolor="none", edgecolor=p["cor"], lw=1.2, zorder=2)
    axz.scatter(p["idade"], p["modelo"], s=26, color=p["cor"], zorder=3)
xi = [p["idade"] for p in perto]; yi = [v for p in perto for v in (p["modelo"], p["ancora"])]
axz.set_xlim(min(xi) - 1.0, max(xi) + 1.0); axz.set_ylim(min(yi) - 0.08, max(yi) + 0.08)
axz.tick_params(labelsize=5.5)
axz.set_xticks([])                                  # só o eixo vertical (MAE): evita os números por cima do eixo do painel; axz.grid(alpha=0.3, ls=":")
ind = ax2.indicate_inset_zoom(axz, edgecolor="#888888")
for c in getattr(ind, "connectors", []):          # sem linhas de ligação: ficavam por cima dos anéis
    c.set_visible(False)
ax2.set_xlabel("Median calibration age (min)")
ax2.set_ylabel("MAE (mmHg)")
ax2.set_title("(b) Model and anchor follow the calibration age,\nnot the method", fontsize=7)
ax2.grid(alpha=0.3, ls=":")
ax2.set_xlim(30, 100); ax2.set_ylim(8.5, 14.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{AQUI}/figura_deriva_calibracao_pequena.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
for p in pts:
    print(f"{p['nome']:28s} idade {p['idade']:5.1f}  âncora {p['ancora']:6.2f}  modelo {p['modelo']:6.2f}  ganho {p['ancora'] - p['modelo']:+.2f}")
