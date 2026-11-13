"""Figura 7.1: latência (percentil 95) e CPU em função do número de pulseiras por processo, para o XGBoost e para os dois híbridos. Níveis de 10 a 200 pulseiras:
a corrida grossa; de 300 a 500: a mediana das três repetições. A linha a tracejado é o limiar de fila (1 s).

Dados, nesta pasta: carga_<modelo>.json (corrida grossa) e carga_<modelo>_rep1.json a _rep3.json (repetições), para xgb, bilstm e bitcn.
Uso: python3 gerar_figura.py      (escreve teste_carga_tempo_final.png e resumo_carga_tempo_final.json nesta pasta)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q = os.path.dirname(os.path.abspath(__file__))
VERM, AZUL, VERDE, CINZA = '#c0392b', '#1f5fa8', '#16a085', '#555555'
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6})
NOMES = {"xgb": ("XGBoost", VERM), "bilstm": ("BiLSTM + XGBoost", AZUL), "bitcn": ("BiTCN + XGBoost", VERDE)}
fig, ax = plt.subplots(1, 2, figsize=(5.98, 2.3))
R = {}
for m, (nome, cor) in NOMES.items():
    grosso = {n["n_pulseiras"]: [n] for n in json.load(open(f"{Q}/carga_{m}.json"))["niveis"] if n["n_pulseiras"] <= 200}
    reps = {}
    for r in (1, 2, 3):
        for n in json.load(open(f"{Q}/carga_{m}_rep{r}.json"))["niveis"]:
            reps.setdefault(n["n_pulseiras"], []).append(n)
    niv = {**grosso, **reps}
    x = sorted(niv)
    p95 = [float(np.median([n["lat"]["p95"] for n in niv[k]])) for k in x]
    cpu = [float(np.median([n["cpu_nucleos"] for n in niv[k]])) for k in x]
    ax[0].plot(x, p95, "o-", color=cor, lw=1.1, ms=2.8, label=nome)
    ax[1].plot(x, cpu, "o-", color=cor, lw=1.1, ms=2.8, label=nome)
    sem_fila = [k for k in x if all(n["lat"]["p95"] < 1.0 for n in niv[k])]
    g = json.load(open(f"{Q}/carga_{m}.json"))
    R[m] = dict(capacidade_pulseiras=max(sem_fila), p95_ms={k: [round(n["lat"]["p95"] * 1e3, 1) for n in niv[k]] for k in x if k >= 300},
                cpu_nucleos={k: round(float(np.median([n["cpu_nucleos"] for n in niv[k]])), 2) for k in x if k >= 300},
                memoria_modelo_mb=round(g["rss_modelo_mb"] - g["rss_base_mb"]), ficheiros_modelo_mb=round(g["artefacto_bytes"] / 1e6, 2),
                extracao_ms_p50=round(g["micro"]["extracao_ms"]["p50"], 1), inferencia_isolada_ms_p50=round(g["micro"]["inferencia_isolada_ms"]["p50"], 2))
ax[0].axhline(1.0, color=CINZA, ls="--", lw=1)
ax[0].text(12, 1.25, "queue threshold (1 s)", fontsize=6.5, color=CINZA)
ax[0].set_yscale("log"); ax[0].set_xscale("log"); ax[0].set_ylabel("Latency, 95th percentile (s)")
ax[1].set_xscale("log"); ax[1].set_ylabel("CPU (cores)")
for a in ax:
    a.set_xlabel("Wearable devices per process")
ax[0].legend(loc="upper left", fontsize=6.5)
fig.tight_layout(pad=0.4, w_pad=1.0)
fig.savefig(f"{Q}/teste_carga_tempo_final.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
json.dump(R, open(f"{Q}/resumo_carga_tempo_final.json", "w"), indent=1)      # o resumo por modelo
print(json.dumps({m: {k: v for k, v in r.items() if k not in ("p95_ms", "cpu_nucleos")} for m, r in R.items()}, indent=1))
