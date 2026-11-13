"""Figura 6.1: pulso do PPG e batimento do ECG do mesmo instante, com as grandezas extraídas (PAT, tempos de subida e de descida, W25/W50/W75, índice de
reflexão, segunda derivada, QRS, QT, TQ e RR).

Dados, nesta pasta:
  sinais.csv   PPG e ECG já filtrados (PPG: passa-baixo a 8 Hz e orientação corrigida; ECG: passa-alto a 0,5 Hz e polaridade corrigida), uma linha por amostra (100 Hz);
               ppg_savgol é a suavização (Savitzky-Golay, janela de 5, grau 3) do PPG entre o pé do pulso e 1,1 vezes o intervalo entre picos,
               de onde sai a segunda derivada.
  pontos.json  fs_hz e o índice (de `amostra`) de cada ponto característico: r e rn (picos R do batimento e do seguinte), q, s, t e qn (ECG),
               pk e pkn (picos sistólicos do PPG), v (pé do pulso), pe (fim da descida), v2 (pé do pulso seguinte), dmax (pico diastólico)
               e, relativos a v, ia, ib, ic e idd (pontos a, b, c e d da segunda derivada).
Uso: python3 gerar_figura.py      (escreve pulso_ecg_caracteristicas_morf_final.png nesta pasta)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
VERM, AZUL, CINZA = '#c0392b', '#1f5fa8', '#555555'
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6})
P_ = json.load(open(f"{AQUI}/pontos.json"))
FS, J = P_["fs_hz"], P_["amostras_da_janela"]
r, rn, q, s, t, qn, pk, pkn, v, pe, v2, dmax, ia, ib, ic, idd = (P_[k] for k in ("r", "rn", "q", "s", "t", "qn", "pk", "pkn", "v", "pe", "v2", "dmax", "ia", "ib", "ic", "idd"))
d_ = pd.read_csv(f"{AQUI}/sinais.csv")
p, e = np.full(J, np.nan), np.full(J, np.nan)          # só se desenha o troço dos dados; fora dele fica NaN
p[d_.amostra.values], e[d_.amostra.values] = d_.ppg.values, d_.ecg.values
suave = d_.ppg_savgol.dropna().values
VERDE, ROXO = "#1e8449", "#7d3c98"
amp_ = p[pk] - p[v]
ri = (p[dmax] - p[v]) / amp_
pl, ph = pk - 10, pk + 30
pul = p[pl:ph]; nrm = (pul - pul.min()) / np.ptp(pul); i50 = np.where(nrm >= 0.5)[0]; a50, b50 = pl + i50[0], pl + i50[-1]
y50 = pul.min() + 0.5 * np.ptp(pul); base = pul.min()
tx = (np.arange(J) - r) / FS
x0, x1 = r - 25, max(v2, rn, pkn) + 12
amp = p[pk] - p[v]
fig, (ap, ag, ae) = plt.subplots(3, 1, figsize=(7.8, 7.9), sharex=True, gridspec_kw=dict(hspace=0.12, height_ratios=[1.15, 0.8, 1]))
# ── PPG ──
ap.plot(tx[x0:x1], p[x0:x1], color=CINZA, lw=1.5)
ap.fill_between(tx[pl:ph], base, p[pl:ph], color=VERM, alpha=0.13, label="pulse area (window peak − 0.1 s to + 0.3 s)")
for k in (pl, ph - 1):
    ap.vlines(tx[k], base, p[k], colors=VERM, linestyles=":", lw=1.0)
    ap.plot(tx[k], p[k], "s", color=VERM, ms=4.5, mfc="white")
ap.text(tx[pl] + 0.06, base - 0.07 * amp, "peak − 0.1 s", color=VERM, ha="center", va="top", fontsize=9)
ap.text(tx[ph - 1], base - 0.07 * amp, "peak + 0.3 s", color=VERM, ha="center", va="top", fontsize=9)
ap.hlines(y50, tx[a50], tx[b50], colors="k", linestyles="--", lw=1.0)
ap.plot([tx[a50], tx[b50]], [y50, y50], "o", color="k", ms=4.5, mfc="white", label="W50 limits (50 % level)")
ap.text(tx[b50] + 0.012, y50, "W50", fontsize=9, ha="left", va="center")
ap.plot(tx[v], p[v], "o", color=AZUL, ms=7, label="valley = foot of the pulse ($P_f$)")
ap.plot(tx[pk], p[pk], "o", color=VERM, ms=7, label="systolic peak ($P_s$)")
ap.plot(tx[pe], p[pe], "o", color=ROXO, ms=6, label="end of decay")
ap.plot(tx[dmax], p[dmax], "^", color=VERDE, ms=7, label="diastolic peak ($P_d$)")
y_rt = p[v] - 0.14 * amp
ap.annotate("", xy=(tx[pk], y_rt), xytext=(tx[v], y_rt), arrowprops=dict(arrowstyle="<->", color=AZUL))
ap.text((tx[v] + tx[pk]) / 2, y_rt - 0.03 * amp, "rise time", color=AZUL, ha="center", va="top", fontsize=9)
y_dt = p[pk] + 0.10 * amp
ap.annotate("", xy=(tx[pe], y_dt), xytext=(tx[pk], y_dt), arrowprops=dict(arrowstyle="<->", color=ROXO))
ap.text((tx[pk] + tx[pe]) / 2, y_dt + 0.03 * amp, "decay time", color=ROXO, ha="center", va="bottom", fontsize=9)
ap.plot(tx[pkn], p[pkn], "o", color=VERM, ms=7)              # pico sistólico seguinte: intervalo entre picos (FC)
y_pp = max(p[pk], p[pkn]) + 0.30 * amp
ap.vlines([tx[pk], tx[pkn]], [p[pk], p[pkn]], y_pp, colors=VERM, linestyles=":", lw=0.8)
ap.annotate("", xy=(tx[pkn], y_pp), xytext=(tx[pk], y_pp), arrowprops=dict(arrowstyle="<->", color=VERM))
ap.text((tx[pk] + tx[pkn]) / 2, y_pp + 0.03 * amp, "interval between systolic peaks (HR)", color=VERM, ha="center",
        va="bottom", fontsize=9)
xa = tx[x0] + 0.04
ap.hlines([p[pk], p[v]], xa, [tx[pk], tx[v]], colors=VERM, linestyles=":", lw=0.8)
ap.annotate("", xy=(xa, p[pk]), xytext=(xa, p[v]), arrowprops=dict(arrowstyle="<->", color=VERM))
ap.text(xa + 0.015, p[v] + 0.80 * amp, "pulsatile amplitude\n$P_s - P_f$", color=VERM, ha="left", va="center", fontsize=9)
# ── grandezas novas: W25, W75 e índice de reflexão ──
ncor = "#b9770e"
for fr_, nome_, dy_ in ((0.25, "W25", -0.0), (0.75, "W75", 0.0)):
    ii = np.where(nrm >= fr_)[0]; a_, b_ = pl + ii[0], pl + ii[-1]
    yy = pul.min() + fr_ * np.ptp(pul)
    ap.hlines(yy, tx[a_], tx[b_], colors=ncor, linestyles="--", lw=1.0)
    ap.plot([tx[a_], tx[b_]], [yy, yy], "o", color=ncor, ms=4, mfc="white")
    ap.text(tx[b_] + 0.012, yy, nome_, color=ncor, fontsize=9, ha="left", va="center")
xr = tx[dmax] + 0.07
ap.hlines([p[v], p[dmax]], tx[dmax], xr + 0.01, colors=ncor, linestyles=":", lw=0.9)
ap.annotate("", xy=(xr, p[dmax]), xytext=(xr, p[v]), arrowprops=dict(arrowstyle="<->", color=ncor, lw=1.0))
ap.text(xr + 0.015, (p[dmax] + p[v]) / 2, "$P_d - P_f$", color=ncor, fontsize=9, ha="left", va="center")
ap.text(xr - 0.40, p[v] - 0.16 * amp, f"reflection index $= (P_d - P_f)\\,/\\,(P_s - P_f) = {ri:.2f}$", color=ncor, fontsize=9, ha="left", va="top")
for k_, nome_, dx_, dy_, ha_, va_ in ((pk, "$P_s$", -0.025, 0.03, "right", "bottom"), (v, "$P_f$", -0.03, -0.06, "right", "top"), (dmax, "$P_d$", 0.0, 0.07, "center", "bottom")):
    ap.text(tx[k_] + dx_, p[k_] + dy_ * amp, nome_, color=ncor, fontsize=11, fontweight="bold", ha=ha_, va=va_)
ap.set_ylim(p[v] - 0.62 * amp, y_pp + 0.18 * amp); ap.set_ylabel("PPG (ADC counts)")
h, l = ap.get_legend_handles_labels()
# ── 2.ª derivada do pulso (a, b, c, d, como em extrair_apg_features: mesmo troço, do pé a 1,1 x o intervalo entre picos) ──
fim_pulso = min(v + int(1.1 * (pkn - pk)), J)
apg = np.gradient(np.gradient(suave))
an = apg / apg[ia]
xs = tx[v:fim_pulso]
d1 = np.gradient(suave); d1n = d1 / d1.max()          # 1.ª derivada (declive do pulso), só para dar a noção: a 2.ª é o declive desta
ag.plot(xs, d1n, color=AZUL, lw=1.6, alpha=0.35, label="1st derivative, normalised to its maximum")
ag.plot(xs, an, color=CINZA, lw=1.5, label="2nd derivative, normalised to $a$"); ag.axhline(0, color="0.6", lw=0.8)
for k_, nome_, dx_, dy_, va_ in ((ia, "$a$", -0.025, 0.0, "center"), (ib, "$b$", 0.025, -0.08, "top"), (ic, "$c$", 0.0, 0.1, "bottom"), (idd, "$d$", 0.0, -0.1, "top")):
    ag.plot(xs[k_], an[k_], "o", color=VERM, ms=5); ag.text(xs[k_] + dx_, an[k_] + dy_, nome_, color=VERM, fontsize=11, fontweight="bold", ha="center", va=va_)
for k_ in (ia, ib, idd):
    for a_ in (ap, ag):
        a_.axvline(xs[k_], color=VERM, lw=0.7, ls=":", alpha=0.6)
ag.text(xs[idd] + 0.06, an.min() - 0.30, f"$b/a = {apg[ib] / apg[ia]:.2f}$\n$d/a = {apg[idd] / apg[ia]:.2f}$", color=VERM, fontsize=9, ha="left", va="bottom")
ag.set_ylabel("Derivatives\n(normalised)"); ag.legend(loc="upper right", fontsize=8.8, frameon=False); ag.set_ylim(an.min() - 0.35, 1.35)
# ── ECG ──
en = (e - np.median(e[x0:x1])) / (e[r] - np.median(e[x0:x1]))
ae.plot(tx[x0:x1], en[x0:x1], color=AZUL, lw=1.3)
for k, nome, dx, dy, va in ((q, "Q", -0.02, -0.05, "top"), (r, "R", -0.03, 0.0, "center"), (s, "S", 0.02, -0.05, "top"), (t, "T", 0, 0.07, "bottom")):
    ae.plot(tx[k], en[k], "o", color=AZUL, ms=4.5); ae.text(tx[k] + dx, en[k] + dy, nome, color=AZUL, ha="center", va=va, fontsize=9, fontweight="bold")
def seta(xa_, xb_, y, txt, cor, va="bottom", dy=0.04):
    ae.annotate("", xy=(xb_, y), xytext=(xa_, y), arrowprops=dict(arrowstyle="<->", color=cor, lw=1.0, shrinkA=0, shrinkB=0))
    ae.text((xa_ + xb_) / 2, y + (dy if va == "bottom" else -dy), txt, color=cor, ha="center", va=va, fontsize=9)
seta(tx[r], tx[rn], 1.35, "RR", CINZA); seta(tx[q], tx[t], 0.85, "QT", ROXO); seta(tx[t], tx[qn], 0.85, "TQ", ROXO)
seta(tx[q], tx[s], -0.45, "QRS", AZUL, va="top")
ae.set_ylim(-0.8, 1.65); ae.set_ylabel("ECG (normalised to R)"); ae.set_xlabel("Time from R peak (s)")
ae.set_xlim(tx[x0], tx[x1 - 1])
# ── PAT: R (ECG) -> pico sistólico (PPG) ──
for k in (r, pk):
    for a_ in (ap, ag, ae):
        a_.axvline(tx[k], color="0.6", lw=0.8, ls=":")
ypat = p[v] - 0.42 * amp
ap.annotate("", xy=(tx[pk], ypat), xytext=(tx[r], ypat), arrowprops=dict(arrowstyle="<->", color="k", lw=1.1))
ap.text((tx[r] + tx[pk]) / 2, ypat - 0.03 * amp, "PAT", ha="center", va="top", fontsize=9, fontweight="bold")
for a_, letra in ((ap, "(a)"), (ag, "(b)"), (ae, "(c)")):
    a_.text(-0.09, 1.0, letra, transform=a_.transAxes, fontsize=10, fontweight="bold", va="top")
fig.align_ylabels([ap, ag, ae]); fig.legend(h, l, loc='lower center', ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.53, 0.0), columnspacing=1.2); fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.165)
fig.savefig(f"{AQUI}/pulso_ecg_caracteristicas_morf_final.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("pulso_ecg_caracteristicas_morf_final.png")
