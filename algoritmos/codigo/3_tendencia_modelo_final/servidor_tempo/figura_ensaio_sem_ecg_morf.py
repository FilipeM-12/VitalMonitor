"""Figura do ensaio sem ECG com o worker da tendência (um só detetor, 43 caracteristicas).
Painel de cima: janelas aceites (sonda_extracao_ecg_morf.py, 20 min de sinal real de bancada).
Painel de baixo: previsoes gravadas pelo worker no ensaio ponta a ponta com ECG a zeros (ensaio_worker_600s)."""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q = os.path.dirname(os.path.abspath(__file__))
VERM, AZUL, CINZA = '#c0392b', '#1f5fa8', '#555555'
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6})
j = json.load(open(f"{Q}/janelas_sonda_ecg_morf.json"))
d = pd.read_csv(f"{Q}/ensaio_worker_600s/medicao_tendencia_ensaio.csv", parse_dates=["timestamp"])

fig, ax = plt.subplots(2, 1, figsize=(8.5, 4.4), gridspec_kw=dict(height_ratios=[3, 1.6]))
linhas = (("ecg_real", "extracao_com_ecg", "R-peak extraction, real ECG", VERM),
          ("ecg_a_zeros", "extracao_com_ecg", "R-peak extraction, ECG at zero", VERM),
          ("ecg_a_zeros", "extracao_so_ppg", "PPG-only extraction, ECG at zero", AZUL))
contagens = {}
for y, (c, ex, nome, cor) in enumerate(linhas):
    t = np.array(j[c]["fim_s"]) / 60
    ok = np.array([bool(v) for v in j[c][ex]])
    ax[0].plot(t[ok], np.full(ok.sum(), y), "|", ms=14, mew=1.2, color=cor)
    contagens[nome] = int(ok.sum())
    ax[0].text(20.7, y, f"{ok.sum()}/{len(ok)}", fontsize=8, color=CINZA, va="center")
ax[0].set_yticks(range(len(linhas))); ax[0].set_yticklabels([n for *_, n, _ in linhas], fontsize=8.5)
ax[0].set_ylim(-0.6, len(linhas) - 0.4); ax[0].set_xlim(0, 22.5)
ax[0].set_xlabel("Time (min)"); ax[0].grid(axis="y", visible=False)
ax[0].set_title("Windows accepted by each extraction", fontsize=9.5, loc="left")

t0 = d.timestamp.min()
tm = (d.timestamp - t0).dt.total_seconds() / 60
ax[1].plot(tm, np.zeros(len(tm)), "|", ms=14, mew=2, color=AZUL)
ax[1].set_yticks([0]); ax[1].set_yticklabels(["Worker, ECG at zero"], fontsize=8.5)
ax[1].set_ylim(-0.6, 0.6); ax[1].set_xlim(-0.2, 10.5)
ax[1].text(10.6, 0, f"{len(d)} predictions", fontsize=8, color=CINZA, va="center", clip_on=False)
ax[1].set_xlabel("Time since the first prediction (min)"); ax[1].grid(axis="y", visible=False)
ax[1].set_title("Predictions recorded in the end-to-end test", fontsize=9.5, loc="left")
fig.tight_layout()
fig.savefig(f"{Q}/ensaio_sem_ecg_morf.png", dpi=200)
json.dump(dict(aceites=contagens, previsoes=int(len(d))), open(f"{Q}/ensaio_sem_ecg_morf.json", "w"), indent=1)
print(contagens, len(d))
