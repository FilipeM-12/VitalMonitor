#!/usr/bin/env python3
"""O modelo bate a âncora (repetir a última calibração)? Diferença de erro absoluto (âncora - modelo; positivo = o modelo ganha), no conjunto todo e
por faixa de tempo desde a calibração, com IC95 por bootstrap ao nível do sujeito (1000 reamostragens). Variante de referência, holdout."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nucleo import config as C

d = pd.read_csv(f"{C.DIR_RESULTADOS}/experiencias/XGB_baseline/holdout_predicoes.csv")
d["err_modelo"] = (d.preds_final - d.sbp_target).abs()
d["err_ancora"] = (d.sbp_calib - d.sbp_target).abs()
d["ganho"] = d.err_ancora - d.err_modelo
d["idade_min"] = d.tempo_desde_calib_seg / 60.0
FAIXAS, ROT = [0, 5, 15, 30, 60, 120, 1e5], ["<5", "5-15", "15-30", "30-60", "60-120", ">120"]
d["faixa"] = pd.cut(d.idade_min, FAIXAS, labels=ROT)
rng = np.random.default_rng(C.SEMENTE)
ids = d.subject_id.unique()
por_suj = {s: g for s, g in d.groupby("subject_id")}


def ic(sel):
    base = {s: g[sel(g)] for s, g in por_suj.items()}
    base = {s: g for s, g in base.items() if len(g)}
    chaves = list(base)
    vals = []
    for _ in range(1000):
        a = pd.concat([base[k] for k in rng.choice(chaves, len(chaves), replace=True)])
        vals.append(a.ganho.mean())
    sub = pd.concat(base.values())
    return float(sub.ganho.mean()), [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], int(len(sub)), int(len(chaves))


out = {"todos": ic(lambda g: np.ones(len(g), bool))}
for r in ROT:
    out[r] = ic(lambda g, r=r: (g.faixa == r).values)
res = {k: dict(ganho=v[0], ic95=v[1], n_janelas=v[2], n_sujeitos=v[3], exclui_zero=bool(v[1][0] > 0 or v[1][1] < 0)) for k, v in out.items()}
os.makedirs(f"{C.DIR_RESULTADOS}/tabelas", exist_ok=True)
json.dump(res, open(f"{C.DIR_RESULTADOS}/tabelas/ganho_contra_ancora.json", "w"), indent=1)
for k, v in res.items():
    print(f"{k:7s} ganho {v['ganho']:+.3f}  IC95 [{v['ic95'][0]:+.3f}, {v['ic95'][1]:+.3f}]  n={v['n_janelas']} suj={v['n_sujeitos']}  {'EXCLUI 0' if v['exclui_zero'] else 'inclui 0'}")
