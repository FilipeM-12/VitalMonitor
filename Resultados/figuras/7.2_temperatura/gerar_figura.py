"""Figura 7.2: temperatura medida pelo sensor MLX90614 e leituras do termómetro, nos dois ensaios (o sensor debaixo de uma botija de água quente, a arrefecer).

Dados, nesta pasta: ensaio_1_sensor.csv e ensaio_2_sensor.csv (time_ms, o relógio da pulseira, e Temp, a temperatura do sensor em °C, uma leitura a cada 0,5 s) e
termometro.csv (as duas leituras do termómetro de cada ensaio: tempo_s e termometro_C).
Uso: python3 gerar_figura.py      (escreve temperatura_ensaio_final.png nesta pasta)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
termometro = pd.read_csv(f"{AQUI}/termometro.csv")
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.titlesize": 7.5, "axes.linewidth": 0.6})
linhas, fig = [], plt.figure(figsize=(5.98, 2.3))
for k in (1, 2):
    d = pd.read_csv(f"{AQUI}/ensaio_{k}_sensor.csv")
    t = (d.time_ms - d.time_ms.iloc[0]) / 1000
    ax = fig.add_subplot(1, 2, k)
    ax.plot(t / 60, d.Temp, color="tab:blue", lw=0.9, label="MLX90614")
    for _, r in termometro[termometro.ensaio == k].iterrows():
        i = (t - r.tempo_s).abs().idxmin()
        linhas.append({"ensaio": k, "momento": r.momento, "t_s": round(r.tempo_s, 1), "termometro": r.termometro_C, "sensor": d.Temp[i], "erro": round(d.Temp[i] - r.termometro_C, 2)})
        ax.plot(r.tempo_s / 60, r.termometro_C, "o", color="tab:red", ms=4.5, label="Thermometer" if r.momento == "inicio" else None)
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Temperature (°C)"); ax.legend(frameon=False)
    ax.set_title(f"Trial {k}")
fig.tight_layout(pad=0.4, w_pad=1.0)
fig.savefig(os.path.join(AQUI, "temperatura_ensaio_final.png"), dpi=400, bbox_inches="tight", pad_inches=0.03)

r = pd.DataFrame(linhas)
print(r.to_string(index=False))
print("erro médio %+.2f   erro absoluto médio %.2f   máximo %.2f" % (r.erro.mean(), r.erro.abs().mean(), r.erro.abs().max()))
