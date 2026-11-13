#!/usr/bin/env python3
"""Acerto na direção da variação desde a calibração (janelas com |variação|>5 mmHg) do modelo de referência, no conjunto de teste.
Compara com 50 %, com um previsor que diz sempre a classe mais frequente e dá o IC95 por sujeito (bootstrap, 1000 reamostragens).
Resultado: 55,6 % contra 50 % não prova informação direcional, porque o IC contém os 50 % e o previsor trivial acerta mais.
Resultado em resultados/tabelas/direcao_variacao.json."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

AQUI = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(f"{AQUI}/resultados/experiencias/XGB_baseline/holdout_predicoes.csv")
d["dreal"] = d.sbp_target - d.sbp_calib
g = d[d.dreal.abs() > 5].copy()
g["ok"] = (np.sign(g.preds_delta) == np.sign(g.dreal)).astype(float)
y = (g.dreal > 0).astype(int)
sens, esp = g.ok[y == 1].mean(), g.ok[y == 0].mean()
soma, n = g.groupby("subject_id").ok.sum(), g.groupby("subject_id").size()
rng = np.random.default_rng(42)
bs = [soma.values[i].sum() / n.values[i].sum() for i in (rng.integers(0, len(n), len(n)) for _ in range(1000))]
res = dict(janelas=len(g), sujeitos=int(len(n)), acerto=round(g.ok.mean(), 3), ic95_sujeito=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
           fracao_subidas=round(float(y.mean()), 3), acerto_sempre_classe_mais_frequente=round(float(max(y.mean(), 1 - y.mean())), 3),
           acerto_subidas=round(float(sens), 3), acerto_descidas=round(float(esp), 3), acuidade_equilibrada=round(float((sens + esp) / 2), 3), auc_direcao=round(float(roc_auc_score(y, g.preds_delta)), 3))
json.dump(res, open(f"{AQUI}/resultados/tabelas/direcao_variacao.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
