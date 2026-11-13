#!/usr/bin/env python3
"""tab:variantes_absoluto. As variantes de extração e de covariáveis, cada uma com a sua busca (01_correr_experiencia.py).
Como as variantes retêm sujeitos diferentes, dá-se também o MAE só nas linhas que a variante tem em comum com a referência."""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nucleo import config as C

ORDEM = ["nk", "nk2", "fft", "clinicas", "baseline"]
CHAVE = ['subject_id', 'record_id', 'segment_order', 'bloco_id']   # chave física (as sessões diferem entre variantes)
pasta = lambda v: f"{C.DIR_RESULTADOS}/experiencias/XGB_{v}"
base = pd.read_csv(f"{pasta('baseline')}/holdout_predicoes.csv").set_index(CHAVE)
linhas = []
for v in ORDEM:
    m = json.load(open(f"{pasta(v)}/metricas.json"))
    d = pd.read_csv(f"{pasta(v)}/holdout_predicoes.csv").set_index(CHAVE)
    comum = d.index.intersection(base.index)
    e_v = (d.loc[comum, 'sbp_target'] - d.loc[comum, 'preds_final']).abs().mean()
    e_b = (base.loc[comum, 'sbp_target'] - base.loc[comum, 'preds_final']).abs().mean()
    linhas.append(dict(variante=v, descricao=C.VARIANTES[v].descricao, sujeitos=m['n_sujeitos_holdout'], janelas=m['n_holdout'],
                       mae=round(m['mae_holdout'], 4), mae_cv=round(m['mae_cv'], 4),
                       janelas_comuns_com_baseline=len(comum), mae_nas_comuns=round(float(e_v), 4),
                       mae_baseline_nas_comuns=round(float(e_b), 4), ganho_nas_comuns=round(float(e_b - e_v), 4)))
tab = pd.DataFrame(linhas)
os.makedirs(f"{C.DIR_RESULTADOS}/tabelas", exist_ok=True)
tab.to_csv(f"{C.DIR_RESULTADOS}/tabelas/variantes_absoluto.csv", index=False)
print(tab.to_string(index=False))
