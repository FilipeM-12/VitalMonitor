#!/usr/bin/env python3
"""tab:modelos_absoluto e figura do planalto (dispersão das tentativas face à incerteza da métrica).

As cinco famílias vêm de 01_correr_experiencia.py (variante baseline, 20 tentativas cada, 3 dobras). As redes não
preveem os primeiros JANELA_SEQ-1 blocos de cada sessão, por isso o MAE de cada modelo é dado nas suas linhas e
também nas linhas que TODOS os modelos têm (comparação equitativa)."""
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
from nucleo.avaliacao import bootstrap_sujeitos

FAMILIAS = ["XGB", "BILSTM", "BITCN", "HIBRIDO_BILSTM_XGB", "HIBRIDO_BITCN_XGB"]
NOMES = {"XGB": "XGBoost", "BILSTM": "BiLSTM", "BITCN": "BiTCN", "HIBRIDO_BILSTM_XGB": "Hybrid BiLSTM-XGBoost",
         "HIBRIDO_BITCN_XGB": "Hybrid BiTCN-XGBoost"}
CHAVE = ['subject_id', 'record_id', 'segment_order', 'bloco_id']   # chave física do bloco
pasta = lambda f: f"{C.DIR_RESULTADOS}/experiencias/{f}_baseline"

prev, met = {}, {}
for f in FAMILIAS:
    met[f] = json.load(open(f"{pasta(f)}/metricas.json"))
    d = pd.read_csv(f"{pasta(f)}/holdout_predicoes.csv").set_index(CHAVE)
    prev[f] = d
comum = None
for d in prev.values():
    comum = d.index if comum is None else comum.intersection(d.index)
erros = {f: (prev[f].loc[comum, 'sbp_target'] - prev[f].loc[comum, 'preds_final']).abs() for f in FAMILIAS}
linhas = []
for f in FAMILIAS:
    linhas.append(dict(modelo=NOMES[f], mae_cv=round(met[f]['mae_cv'], 4), mae_holdout=round(met[f]['mae_holdout'], 4),
                       n=met[f]['n_holdout'], mae_holdout_linhas_comuns=round(float(erros[f].mean()), 4), n_comum=len(comum),
                       tempo_busca_h=round(met[f]['tempo_busca_s'] / 3600, 2)))
tab = pd.DataFrame(linhas)
os.makedirs(f"{C.DIR_RESULTADOS}/tabelas", exist_ok=True)
tab.to_csv(f"{C.DIR_RESULTADOS}/tabelas/modelos_absoluto.csv", index=False)
print(tab.to_string(index=False))

# ── planalto: tentativas do XGBoost contra a incerteza de amostragem do MAE ───────────────────────────────────────
trials = pd.read_csv(f"{pasta('XGB')}/trials.csv")
x = prev['XGB']
sujeito = pd.Series(x.index.get_level_values('subject_id'), index=x.index)
boot = bootstrap_sujeitos({"XGB": (x['sbp_target'] - x['preds_final']).abs()}, sujeito)
ic = boot['mae']['XGB']['ic95']
largura = ic[1] - ic[0]
y = trials['value'].values
inc = dict(mae_holdout=met['XGB']['mae_holdout'], n_sujeitos_bootstrap=boot['n_sujeitos'], n_bootstrap=1000,
           ic95_baixo=ic[0], ic95_alto=ic[1], ic95_largura=largura,
           trial_melhor=float(y.min()), trial_mediana=float(np.median(y)), trial_diferenca=float(np.median(y) - y.min()),
           trial_pior=float(y.max()), n_trials=int(len(y)))
json.dump(inc, open(f"{C.DIR_RESULTADOS}/tabelas/planalto_incerteza.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.scatter(trials['number'], y, s=28, color="#2b6cb0", zorder=3, label=f"Optuna trials (CV MAE, {len(y)})")
med = float(np.median(y))
ax.axhline(med, color="#2b6cb0", linestyle="--", linewidth=1, alpha=0.6, label=f"Median of the trials ({med:.2f} mmHg)")
ax.errorbar([-2.5], [med], yerr=[[largura / 2], [largura / 2]], fmt="o", color="#c05621", capsize=6, markersize=6, linewidth=2,
            label=f"95% bootstrap CI (subject level, width={largura:.2f} mmHg)")
ax.set_xlim(-4, len(y))                                  # o intervalo de confiança fica à esquerda, longe da legenda
ax.set_xlabel("Trial number (Optuna)")
ax.set_ylabel(f"MAE (mmHg, inner {C.N_FOLDS}-fold CV)")
ax.set_title(f"Spread of the Optuna trials against the metric's uncertainty ({len(y)} trials)")
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
os.makedirs(f"{C.DIR_RESULTADOS}/figuras", exist_ok=True)
fig.savefig(f"{C.DIR_RESULTADOS}/figuras/dispersao_trials.png", dpi=200)
print(json.dumps(inc, indent=1))
