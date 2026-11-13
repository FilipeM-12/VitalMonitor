#!/usr/bin/env python3
"""Tabela do AUC no teste, por semente, do XGBoost e dos híbridos com regressão quantílica (a formulação adotada), nas duas direções (o que varia entre sementes e a maior variabilidade dos híbridos). Lê resumo_hibrido_quantilico_sementes.csv (colunas direcao, semente,
modelo, auc_xgb, auc_hibrido), por omissão o da pasta ORIG e, se não existir, o de ORIG/antes_reafinar_22_10.
Escreve tabela_sementes_hibridos.json. Uso: python tabela_sementes_hibridos.py"""
import json, os
import pandas as pd
A = "/home/filipe/Desktop/Tese/algoritmo-ml"
ORIG = f"{A}/2026-09-13_modelo_temporal_causal"
f = f"{ORIG}/resumo_hibrido_quantilico_sementes.csv"
if not os.path.exists(f):
    f = f"{ORIG}/antes_reafinar_22_10/resumo_hibrido_quantilico_sementes.csv"
d = pd.read_csv(f)
res = {"ficheiro": f}
for dr in ("descida", "subida"):
    x = d[d.direcao == dr]
    xgb = x.drop_duplicates("semente").sort_values("semente").auc_xgb.tolist()
    linhas = {"XGBoost": xgb}
    for m in ("BiLSTM+XGB", "BiTCN+XGB"):
        linhas[m] = x[x.modelo == m].sort_values("semente").auc_hibrido.tolist()
    res[dr] = {k: dict(auc=v, amplitude=max(v) - min(v)) for k, v in linhas.items()}
json.dump(res, open(os.path.dirname(os.path.abspath(__file__)) + "/tabela_sementes_hibridos.json", "w"), indent=1)

print(json.dumps({dr: {m: [round(a, 3) for a in v["auc"]] + [round(v["amplitude"], 3)] for m, v in res[dr].items()} for dr in ("descida", "subida")}))
