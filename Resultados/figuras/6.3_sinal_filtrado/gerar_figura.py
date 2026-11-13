"""Figura 6.3: o PPG (infravermelho) e o ECG antes e depois do filtro, em dois troços de 8 s de uma gravação com o dispositivo.

Dados, nesta pasta (um troço de 8 s de cada sinal, a 100 Hz):
  ppg.csv   tempo_s, ppg_bruto (contagens do ADC) e ppg_filtrado (passa-baixo a 8 Hz, orientação corrigida e deslocado para valores positivos).
  ecg.csv   tempo_s, ecg_bruto e ecg_filtrado (passa-alto a 0,5 Hz, de fase nula).
Uso: python3 gerar_figura.py      (escreve sinal_filtrado_final.png nesta pasta)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
VERM, AZUL, CINZA = '#c0392b', '#1f5fa8', '#555555'
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.titlesize": 7.5, "axes.linewidth": 0.6, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6})
dp, de = pd.read_csv(f"{AQUI}/ppg.csv"), pd.read_csv(f"{AQUI}/ecg.csv")
tp, ir_m, ppg_m = dp.tempo_s.values, dp.ppg_bruto.values, dp.ppg_filtrado.values
te, ecg_m, ecg_f_m = de.tempo_s.values, de.ecg_bruto.values, de.ecg_filtrado.values
fig, ax = plt.subplots(1, 2, figsize=(5.67, 2.35))
inv = np.sign(np.corrcoef(ir_m, ppg_m)[0, 1])        # -1 se a orientação do PPG foi invertida pelo pré-processamento
for a_, x, bruto, filt, cor, passo, tit in ((ax[0], tp, (ir_m if inv > 0 else ir_m.max() + ir_m.min() - ir_m), ppg_m, VERM, 100, "PPG (infrared)" + (", raw inverted" if inv < 0 else "")),
                                            (ax[1], te, ecg_m, ecg_f_m, AZUL, 250, "ECG")):
    a2 = a_.twinx()
    a_.plot(x, bruto, color="#777777", alpha=0.45, lw=1.3, label="raw", zorder=1)
    a2.plot(x, filt, color=cor, lw=0.8, label="filtered", zorder=3)
    ext = max(np.ptp(bruto), np.ptp(filt)) * 1.15
    ext = np.ceil(ext / passo) * passo
    for ej in (a_, a2):
        ej.yaxis.set_major_locator(plt.MultipleLocator(passo))
    c1, c2 = (bruto.max() + bruto.min()) / 2, (filt.max() + filt.min()) / 2
    a_.set_ylim(c1 - ext / 2, c1 + ext / 2); a2.set_ylim(c2 - ext / 2, c2 + ext / 2)
    a_.set_ylabel("Raw (counts)", color="#555555"); a2.set_ylabel("Filtered (counts)", color=cor)
    a_.tick_params(axis="y", colors="#555555"); a2.tick_params(axis="y", colors=cor)
    a2.grid(False)
    a_.set_xlabel("Time (s)"); a_.set_title(tit)
    h1, l1 = a_.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a_.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.5, frameon=False)
fig.tight_layout(pad=0.4, w_pad=0.6)
fig.savefig(f"{AQUI}/sinal_filtrado_final.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("sinal_filtrado_final.png")
