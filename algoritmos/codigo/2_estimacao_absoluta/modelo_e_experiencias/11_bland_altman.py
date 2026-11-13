# Versão com a figura desenhada ao tamanho final na página (10,4 cm de largura) para as letras ficarem legíveis; escreve figuras/bland_altman_final.png e tabelas/bland_altman_final.json.
#!/usr/bin/env python3
"""Erro quadrático médio e análise de Bland-Altman do modelo de referência (XGBoost, variante baseline), no holdout.

Referência = PA sistólica invasiva (sbp_target); estimativa = preds_final. Bland-Altman: diferença (estimativa - referência) contra a
média das duas. Reporta o enviesamento (média das diferenças), os limites de concordância (média +- 1,96 desvios padrão),
o enviesamento proporcional (declive da regressão da diferença sobre a média, com IC95 por bootstrap de sujeitos) e o desvio
padrão das diferenças (critério AAMI: erro médio <= 5 e desvio padrão <= 8 mmHg). O MSE é dado também para a âncora
(repetir a última calibração), para situar o erro do modelo. Texto da figura em inglês."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nucleo import config as C

d = pd.read_csv(f"{C.DIR_RESULTADOS}/experiencias/XGB_baseline/holdout_predicoes.csv")
ref, est, ancora = d.sbp_target.values, d.preds_final.values, d.sbp_calib.values
dif, media = est - ref, (est + ref) / 2


def declive(x, y):
    return float(np.polyfit(x, y, 1)[0])


rng = np.random.default_rng(C.SEMENTE)
sujeitos = d.subject_id.values
ids = np.unique(sujeitos)
idx_por_suj = {s: np.where(sujeitos == s)[0] for s in ids}
boot = []
for _ in range(1000):
    amostra = np.concatenate([idx_por_suj[s] for s in rng.choice(ids, len(ids), replace=True)])
    boot.append(declive(media[amostra], dif[amostra]))

sd = float(dif.std(ddof=1))
res = dict(
    n_janelas=int(len(d)), n_sujeitos=int(len(ids)),
    mse_modelo=float(np.mean(dif ** 2)), rmse_modelo=float(np.sqrt(np.mean(dif ** 2))),
    mse_ancora=float(np.mean((ancora - ref) ** 2)), rmse_ancora=float(np.sqrt(np.mean((ancora - ref) ** 2))),
    mae_modelo=float(np.mean(np.abs(dif))),
    vies=float(dif.mean()), sd_diferencas=sd,
    limite_inferior=float(dif.mean() - 1.96 * sd), limite_superior=float(dif.mean() + 1.96 * sd),
    declive=declive(media, dif), declive_ic95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    corr_dif_media=float(np.corrcoef(media, dif)[0, 1]),
    fora_dos_limites_pct=float(100 * np.mean((dif < dif.mean() - 1.96 * sd) | (dif > dif.mean() + 1.96 * sd))),
    aami_vies_ok=bool(abs(dif.mean()) <= 5), aami_sd_ok=bool(sd <= 8))
os.makedirs(f"{C.DIR_RESULTADOS}/tabelas", exist_ok=True)
json.dump(res, open(f"{C.DIR_RESULTADOS}/tabelas/bland_altman_final.json", "w"), indent=1)

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(4.1, 2.55))
hb = ax.hexbin(media, dif, gridsize=55, bins="log", cmap="Blues", mincnt=1, linewidths=0.2)
xmax = ax.get_xlim()[1]
for y, txt, ls, va in ((res['vies'], f"bias {res['vies']:+.1f}", "-", "bottom"),
                       (res['limite_superior'], f"+1.96 SD: {res['limite_superior']:+.1f}", "--", "bottom"),
                       (res['limite_inferior'], f"-1.96 SD: {res['limite_inferior']:+.1f}", "--", "top")):
    ax.axhline(y, color="#c0392b", ls=ls, lw=0.9)
    ax.text(xmax - 2, y, txt, va=va, ha="right", fontsize=6.5, color="#c0392b")
xx = np.array([media.min(), media.max()])
k = np.polyfit(media, dif, 1)
ax.plot(xx, np.polyval(k, xx), color="#222222", lw=0.9, label=f"regression, slope {res['declive']:+.2f}")
ax.set_xlabel("Mean of estimate and reference SBP (mmHg)")
ax.set_ylabel("Estimate - reference (mmHg)")
ax.legend(loc="upper left", fontsize=6.5)
ax.grid(alpha=0.3, ls=":")
cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.045)
cb.set_label("windows (log scale)", fontsize=6.5); cb.ax.tick_params(labelsize=6.5)
fig.tight_layout(pad=0.4)
os.makedirs(f"{C.DIR_RESULTADOS}/figuras", exist_ok=True)
fig.savefig(f"{C.DIR_RESULTADOS}/figuras/bland_altman_final.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
print(json.dumps(res, indent=1))
