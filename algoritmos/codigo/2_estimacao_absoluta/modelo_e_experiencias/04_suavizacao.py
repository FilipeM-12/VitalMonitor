#!/usr/bin/env python3
"""tab:suavizacao_absoluto. A mesma configuração de referência (XGBoost da variante baseline, mesmos hiperparâmetros,
mesma divisão de sujeitos) com três suavizações das previsões: Kalman, EWMA e nenhuma. Só muda o pós-processamento."""
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
dev, teste, info = carregar("baseline")
linhas = []
for metodo in ("kalman", "ewma", "none"):
    _, m = avaliar_holdout("XGB", params, dev, teste, C.FEATURES_V3, metodo)
    linhas.append(dict(suavizacao=metodo, mae_holdout=round(m['mae_sbp'], 4), n=m['n'], n_sujeitos=m['n_sujeitos']))
    print(linhas[-1], flush=True)
os.makedirs(f"{C.DIR_RESULTADOS}/tabelas", exist_ok=True)
pd.DataFrame(linhas).to_csv(f"{C.DIR_RESULTADOS}/tabelas/suavizacao.csv", index=False)
json.dump(dict(coorte=info, hiperparametros=params, linhas=linhas), open(f"{C.DIR_RESULTADOS}/tabelas/suavizacao.json", "w"), indent=1)
