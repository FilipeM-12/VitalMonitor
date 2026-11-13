#!/usr/bin/env python3
"""Planalto (Cap. 6): que hiperparâmetros acompanham o erro nas 20 tentativas do Optuna do XGBoost de referência.

Correlação de Spearman entre o valor de cada hiperparâmetro e o MAE de validação cruzada da tentativa, nas primeiras 20
tentativas completas (as que a escolha considera), na pesquisa com as 27 características. Escreve resultados/tabelas/efeito_hiperparametros.json."""
import json
import os
import sys

import numpy as np
import optuna
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nucleo import config as C
from nucleo.busca import primeiras_completas

optuna.logging.set_verbosity(optuna.logging.WARNING)
saida = {}
for nome, db in (("27", f"{C.DIR_OPTUNA}/XGB_baseline.db"),):
    estudo = optuna.load_study(study_name="XGB_baseline", storage=f"sqlite:///{db}")
    t = primeiras_completas(estudo, C.N_TRIALS)
    v = np.array([x.value for x in t])
    linhas = {}
    for p in t[0].params:
        rho, pv = spearmanr([x.params[p] for x in t], v)
        linhas[p] = dict(rho=round(float(rho), 3), p=round(float(pv), 4))
    saida[nome] = dict(n_tentativas=len(t), melhor=float(v.min()), mediana=float(np.median(v)), pior=float(v.max()),
                       spearman=linhas, significativos_5pc=sorted(k for k, x in linhas.items() if x["p"] < 0.05))
    print(nome, json.dumps(saida[nome], indent=1))
os.makedirs(f"{C.DIR_RESULTADOS}/tabelas", exist_ok=True)
json.dump(saida, open(f"{C.DIR_RESULTADOS}/tabelas/efeito_hiperparametros.json", "w"), indent=1)
