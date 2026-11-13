#!/usr/bin/env python3
"""A mesma semente dá o mesmo resultado? Treina cada modelo várias vezes com os MESMOS hiperparâmetros, os mesmos dados e a MESMA semente
(a semente 42 da corrida principal) e mede o MAE de holdout de cada corrida. Se o MAE variar, o treino não é reprodutível e essa variação
não se deve à semente. Para o XGBoost passa-se também a semente do booster (como em 02_sementes_bootstrap.py).
Resultado em resultados/tabelas/reprodutibilidade.{csv,json}.

Com o argumento `determinista` corre só os dois híbridos com as operações deterministas do TensorFlow ligadas (TF_DETERMINISTIC_OPS=1,
TF_CUDNN_DETERMINISTIC=1) e grava reprodutibilidade_determinista.{csv,json}: mostra que a variação vem dessas operações."""
import json
import os
import sys

DETERMINISTA = len(sys.argv) > 1 and sys.argv[1] == "determinista"
if DETERMINISTA:                                          # tem de ficar antes de importar o TensorFlow
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
SUFIXO = "_determinista" if DETERMINISTA else ""

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nucleo import config as C
from nucleo.avaliacao import reconstruir
from nucleo.busca import melhores_parametros
from nucleo.dados import carregar
from nucleo.modelos import ajustar_e_prever

CORRIDAS = {"HIBRIDO_BILSTM_XGB": 3, "HIBRIDO_BITCN_XGB": 2} if DETERMINISTA else \
    {"XGB": 2, "BILSTM": 2, "BITCN": 2, "HIBRIDO_BILSTM_XGB": 3, "HIBRIDO_BITCN_XGB": 3}
dev, teste, _ = carregar("baseline")
linhas = []
for familia, n in CORRIDAS.items():
    p = melhores_parametros(familia, "baseline")
    pars = dict(p, seed=C.SEMENTE) if familia == "XGB" else p
    previsoes = []
    for k in range(n):
        s = ajustar_e_prever(familia, pars, dev, teste, C.FEATURES_V3, semente=C.SEMENTE)
        d = reconstruir(teste, s)
        previsoes.append(d['preds_final'].to_numpy())
        mae = float((d['sbp_target'] - d['preds_final']).abs().mean())
        linhas.append(dict(familia=familia, corrida=k + 1, mae_holdout=round(mae, 4), n=len(d)))
        print(linhas[-1], flush=True)
    dif = np.abs(previsoes[0] - previsoes[1])
    linhas[-1]["dif_media_previsoes_corrida1_vs_2"] = round(float(dif.mean()), 4)
    print(f"  {familia}: diferença média entre as previsões da corrida 1 e 2 = {dif.mean():.4f} mmHg", flush=True)

tab = pd.DataFrame(linhas)
os.makedirs(f"{C.DIR_RESULTADOS}/tabelas", exist_ok=True)
tab.to_csv(f"{C.DIR_RESULTADOS}/tabelas/reprodutibilidade{SUFIXO}.csv", index=False)
res = {f: dict(maes=[l["mae_holdout"] for l in linhas if l["familia"] == f],
               amplitude=round(max(l["mae_holdout"] for l in linhas if l["familia"] == f) - min(l["mae_holdout"] for l in linhas if l["familia"] == f), 4))
       for f in CORRIDAS}
json.dump(res, open(f"{C.DIR_RESULTADOS}/tabelas/reprodutibilidade{SUFIXO}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
