#!/usr/bin/env python3
"""tab:versoes_absoluto. As quatro versões incrementais das características, todas com os MESMOS hiperparâmetros
(os do XGBoost de referência) e a mesma divisão de sujeitos:
  v1 PAT, idade e FC (+ características anti-deriva)   v2 = v1 + morfologia do PPG
  v3 = v2 + morfologia do ECG (a referência)           v4 = v3 + grupo nosológico (is_cardiac, is_respiratory)
Os hiperparâmetros são os do modelo de referência (v3) em todas, para que a diferença entre versões se deva só às características."""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nucleo import config as C
from nucleo.avaliacao import avaliar_holdout
from nucleo.busca import melhores_parametros
from nucleo.dados import carregar

params = melhores_parametros("XGB", "baseline")
linhas = []
for versao, features in C.VERSOES.items():
    variante = "diagnostico" if versao == "v4" else "baseline"        # a v4 precisa das duas colunas do diagnóstico
    dev, teste, info = carregar(variante, features=features)
    _, m = avaliar_holdout("XGB", params, dev, teste, features)
    linhas.append(dict(versao=versao, n_features=len(features), mae_holdout=round(m['mae_sbp'], 4), n=m['n'], n_sujeitos=m['n_sujeitos']))
    print(linhas[-1], flush=True)
os.makedirs(f"{C.DIR_RESULTADOS}/tabelas", exist_ok=True)
pd.DataFrame(linhas).to_csv(f"{C.DIR_RESULTADOS}/tabelas/versoes_features.csv", index=False)
json.dump(dict(hiperparametros=params, linhas=linhas), open(f"{C.DIR_RESULTADOS}/tabelas/versoes_features.json", "w"), indent=1)
