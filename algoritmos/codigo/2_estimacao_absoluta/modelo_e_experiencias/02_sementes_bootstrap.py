#!/usr/bin/env python3
"""

Ganho dos híbridos sobre o XGBoost no absoluto, com a incerteza de sementes e de amostragem. As redes sozinhas também se treinam com as 3 sementes
(para a tabela dos modelos, que mostra o MAE de cada arquitetura por semente).

Cada semente é UM modelo (nunca se faz a média entre sementes, que seria um ensemble). O XGBoost também se treina com
as 3 sementes (semente do booster 0, 1, 2), para o híbrido de cada semente se comparar com o XGBoost da mesma semente.
Com operações deterministas do TensorFlow (ver o topo do ficheiro), a variação entre sementes é só a da semente.
Bootstrap ao nível do sujeito (1000 reamostragens do holdout) do MAE e da diferença emparelhada
MAE_híbrido - MAE_XGBoost (negativo = o híbrido é melhor). Usa os hiperparâmetros das buscas de 01_correr_experiencia.py.
"""

# Operações deterministas do TensorFlow: a mesma semente dá o mesmo resultado. Sem isto, o híbrido BiLSTM-XGB varia até 0,3 mmHg
# entre corridas com a mesma semente (ver 14_reprodutibilidade.py). Os processos dos trials herdam estas variáveis.
import os as _os
_os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
_os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import json
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nucleo import config as C
from nucleo.avaliacao import bootstrap_sujeitos, reconstruir
from nucleo.busca import melhores_parametros
from nucleo.dados import carregar
from nucleo.modelos import ajustar_e_prever

N_SEMENTES, N_BOOT = 3, 1000
dev, teste, info = carregar("baseline")
features = C.FEATURES_V3

erros, mae = {}, {}
for familia in ("XGB", "BILSTM", "BITCN", "HIBRIDO_BITCN_XGB", "HIBRIDO_BILSTM_XGB"):
    p = melhores_parametros(familia, "baseline")
    for s in range(N_SEMENTES):
        pars = dict(p, seed=s) if familia == "XGB" else p            # XGB: semente do booster; híbridos: semente da rede
        previsoes = ajustar_e_prever(familia, pars, dev, teste, features, semente=C.SEMENTE + s)
        d = reconstruir(teste, previsoes)
        nome = f"{familia} s{s}"
        erros[nome] = (d['sbp_target'] - d['preds_final']).abs()
        mae[nome] = float(erros[nome].mean())
        print(f"{nome}: MAE {mae[nome]:.4f} (n={len(d)})", flush=True)

pares = {k: f"XGB s{k[-1]}" for k in erros if k.startswith("HIBRIDO")}          # só os híbridos se comparam com o XGBoost
res = bootstrap_sujeitos(erros, teste['subject_id'], pares, n_boot=N_BOOT)
res['mae_por_semente'] = mae
res['amplitude_xgb'] = max(v for k, v in mae.items() if k.startswith("XGB")) - min(v for k, v in mae.items() if k.startswith("XGB"))
for fam in ("BILSTM", "BITCN", "HIBRIDO_BITCN_XGB", "HIBRIDO_BILSTM_XGB"):
    v = [mae[f"{fam} s{s}"] for s in range(N_SEMENTES)]
    res[f'amplitude_{fam}'] = max(v) - min(v)
os.makedirs(f"{C.DIR_RESULTADOS}/tabelas", exist_ok=True)
json.dump(res, open(f"{C.DIR_RESULTADOS}/tabelas/sementes_bootstrap.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in res if k.startswith('amplitude')}, indent=1))
print(json.dumps(res['diferenca'], indent=1))
