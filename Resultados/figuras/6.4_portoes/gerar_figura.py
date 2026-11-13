"""Figura 6.4: os portões de qualidade, com uma janela de 15 s aceite e uma rejeitada pelo critério do artefacto (desvio máximo acima de 4 amplitudes
do pulso), cada uma com os valores dos critérios de janela e de batimento.

Dados, nesta pasta:
  janela_aceite.csv, janela_rejeitada.csv   PPG filtrado da janela de 15 s (125 Hz): tempo_s, ppg, pico_sistolico (1 nos picos detetados) e
                                            pico_retido (1 nos picos que passam os critérios de batimento).
  criterios.json                            os limites dos portões (limites) e, por janela, o veredito e o valor de cada critério, mais a mediana da
                                            janela, a amplitude do pulso e a amostra do desvio máximo.
Uso: python3 gerar_figura.py      (escreve portoes_aceite.png e portoes_rejeitada.png nesta pasta)
"""
import json
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
VERM, AZUL, CINZA, VERDE = '#c0392b', '#1f5fa8', '#555555', '#1e8449'
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6})
CRIT = json.load(open(f"{AQUI}/criterios.json"))
T = SimpleNamespace(**CRIT["limites"])


def desenhar(janela, nome, titulo):
    d = pd.read_csv(f"{AQUI}/janela_{janela}.csv")
    c = CRIT[janela]
    t, sinal = d.tempo_s.values, d.ppg.values
    v = dict(picos=np.flatnonzero(d.pico_sistolico.values), retidos=list(np.flatnonzero(d.pico_retido.values)), n=c["batimentos"], fc=c["fc_mediana_bpm"],
             cv=c["cv_ritmo"], dev=c["desvio_maximo_em_amplitudes"], n_retidos=c["batimentos_retidos"], disp=c["dispersao_tempo_de_subida_s"])
    m, med, amp = c["veredito"], c["mediana"], c["amplitude_do_pulso"]
    # cada imagem fica com metade da largura do texto: gráfico em cima, critérios por baixo
    fig = plt.figure(figsize=(3.5, 4.5))
    ax = fig.add_axes([0.19, 0.50, 0.78, 0.42])
    ax.axhspan(med - 4 * amp, med + 4 * amp, color=VERDE, alpha=0.07, lw=0)
    for s in (-1, 1):
        ax.axhline(med + s * 4 * amp, color=VERDE, ls="--", lw=0.8, label="median ± 4 × pulse amp." if s > 0 else None)
    ax.axhline(med, color=CINZA, ls=":", lw=0.9, label="window median")
    ax.plot(t, sinal, color=CINZA, lw=0.9)
    falhou_janela = m in ("batimentos", "fc", "ritmo", "artefacto")
    if falhou_janela:                                  # os critérios por batimento não chegam a ser avaliados
        ax.plot(t[v["picos"]], sinal[v["picos"]], "o", color=VERM, ms=3.5, label="systolic peak")
    else:
        ret = set(v["retidos"])
        out = [p for p in v["picos"] if p not in ret]
        ax.plot(t[sorted(ret)], sinal[sorted(ret)], "o", color=VERM, ms=3.5, label="systolic peak (kept)")
        if out:
            ax.plot(t[out], sinal[out], "x", color=AZUL, ms=5, mew=1.4, label="systolic peak (discarded)")
    if v["dev"] > 4:
        i = c["amostra_do_desvio_maximo"]
        ax.annotate("max. deviation", xy=(t[i], sinal[i]), xytext=(-6, 0), textcoords="offset points", fontsize=8,
                    color=VERM, va="center", ha="right")
    ax.set_xlim(0, 15)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("PPG (a.u.)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(titulo, fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, fontsize=7, frameon=False,
              handlelength=1.6, columnspacing=0.8)

    falhou = False
    L = []
    for txt, val, ok in (("beats", f"{v['n']}  (≥ {T.MIN_BATIMENTOS})", v["n"] >= T.MIN_BATIMENTOS),
                         ("median HR", f"{v['fc']:.0f} bpm  ({T.HR_MIN}–{T.HR_MAX})", T.HR_MIN <= v["fc"] <= T.HR_MAX),
                         ("rhythm CV", f"{v['cv']:.2f}  (≤ {T.CV_RITMO_MAX})", v["cv"] <= T.CV_RITMO_MAX),
                         ("max. deviation", f"{v['dev']:.1f} × pulse amp.  (≤ 4)", v["dev"] <= 4),
                         ("beats kept", f"{v['n_retidos']}  (≥ {T.MIN_BATIMENTOS_KEPT})", v["n_retidos"] >= T.MIN_BATIMENTOS_KEPT),
                         ("rise time SD", f"{1000 * v['disp']:.0f} ms  (< {1000 * T.DISPERSAO_MAX:.0f})", v["disp"] < T.DISPERSAO_MAX)):
        L.append((txt, val, None if falhou else ok))
        falhou = falhou or not ok
    y = 0.30
    for txt, val, ok in L:
        cor = CINZA if ok is None else (VERDE if ok else VERM)
        marca = "–" if ok is None else ("✓" if ok else "✗")
        fig.text(0.06, y, f"{marca} {txt}", fontsize=8, color=cor, va="top", fontweight="bold")
        fig.text(0.48, y, val if ok is not None else "not evaluated", fontsize=8, color=cor, va="top")
        y -= 0.034
    fig.text(0.06, y - 0.012, "ACCEPTED" if m == "aceite" else "REJECTED", fontsize=9.5, fontweight="bold",
             color=VERDE if m == "aceite" else VERM, va="top")
    fig.savefig(f"{AQUI}/{nome}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ", nome, m)


desenhar("aceite", "portoes_aceite.png", "(a) Accepted window")
desenhar("rejeitada", "portoes_rejeitada.png", "(b) Rejected window")
