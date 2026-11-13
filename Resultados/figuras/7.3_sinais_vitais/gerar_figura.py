"""Figura 7.3: frequência cardíaca (FC), SpO2 e frequência respiratória (FR) calculadas pelo servidor e valores de referência, no ensaio de uma pessoa em repouso.
A referência da FC e da SpO2 é a leitura de uma pulseira de atividade e a da FR a contagem manual das respirações durante um minuto (um segmento horizontal de um minuto).
O traço da SpO2 não mostra os primeiros e os últimos 30 s da gravação.

Dados, nesta pasta:
  serie_servidor.csv   os valores do servidor ao longo do ensaio (grandeza, minutos desde o início da gravação, valor);
  referencias.csv      as leituras de referência (grandeza, minutos desde o início, valor); na FR, os minutos são o início do minuto contado.
Uso: python3 gerar_figura.py      (escreve ensaio_ppg.png nesta pasta)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
S = pd.read_csv(f"{AQUI}/serie_servidor.csv")
R = pd.read_csv(f"{AQUI}/referencias.csv")
xm = S[S.grandeza == "FC"].minutos
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.6})
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.45))
for k, (g, rot) in enumerate((("FC", "HR (bpm)"), ("SpO2", "SpO$_2$ (%)"), ("FR", "RR (breaths/min)"))):
    s, r = S[S.grandeza == g], R[R.grandeza == g]
    ax[k].plot(s.minutos, s.valor, color="#555555", lw=0.7, label="device (server)")
    if g == "FR":           # cada contagem cobre um minuto: segmento horizontal
        for i, (_, l) in enumerate(r.iterrows()):
            ax[k].plot([l.minutos, l.minutos + 1], [l.valor] * 2, color="#c0392b", lw=1.8, label="manual count (1 min)" if i == 0 else None)
    else:
        ax[k].plot(r.minutos, r.valor, "o", color="#c0392b", ms=3, label="smartband")
    ax[k].set_ylabel(rot)
    if g == "FR":           # escala com folga: com valores parecidos, o eixo automático exagera as diferenças
        v = pd.concat([s.valor, r.valor]).dropna()
        if len(v):
            ax[k].set_ylim(max(0, v.min() - 5), v.max() + 5)
    if g == "SpO2":         # folga por baixo (legenda e pontos de referência) e o 100 % visível
        v = pd.concat([s.valor, r.valor]).dropna()
        ax[k].set_ylim(v.min() - 2.5, 100.6)
    if g == "FC":           # folga no topo, para a legenda não tapar os picos
        lo, hi = ax[k].get_ylim()
        ax[k].set_ylim(lo, hi + (hi - lo) * 0.2)
    ax[k].legend(loc={"FR": "lower right", "SpO2": "lower left"}.get(g, "upper right"), fontsize=6.5)
    ax[k].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[k].grid(ls=":", alpha=0.6)
rr = R[R.grandeza == "FR"]
xlim = (xm.min() - 0.35, max(xm.max(), (rr.minutos.max() + 1) if len(rr) else 0) + 0.35)   # o mesmo eixo do tempo nos três painéis
for x_ in ax:
    x_.set_xlim(*xlim)
    x_.set_xlabel("Time since start of recording (min)", fontsize=6.5)
fig.tight_layout(pad=0.4, w_pad=0.8)
fig.savefig(os.path.join(AQUI, "ensaio_ppg.png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
print("ensaio_ppg.png")
