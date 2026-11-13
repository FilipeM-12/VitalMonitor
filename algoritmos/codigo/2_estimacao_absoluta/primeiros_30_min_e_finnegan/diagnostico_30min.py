#!/usr/bin/env python3
"""Dois diagnósticos nas previsões do holdout (XGB baseline, 27 caract.), primeiros 30 min após a calibração, para perceber porque o modelo não bate a âncora:
(a) a correlação entre a variação prevista e a real sobe se se suavizar o ruído (mediana móvel de 2 e 5 min dentro da sessão)?
(b) o modelo ajuda quando a variação real é grande? ganho face à âncora por tamanho da variação real. Não reajusta nada."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/filipe/Desktop/Tese/algoritmo-ml/RESULTADOS_RELATORIO/Absoluto_27")
from nucleo import config as C  # noqa: E402

d = pd.read_csv(f"{C.DIR_RESULTADOS}/experiencias/XGB_baseline/holdout_predicoes.csv").sort_values(["session_id", "bloco_id"]).reset_index(drop=True)
d["idade_min"] = d.tempo_desde_calib_seg / 60
d["dv_real"] = d.sbp_target - d.sbp_calib
d["dv_prev"] = d.preds_final - d.sbp_calib
d["e_mod"] = (d.preds_final - d.sbp_target).abs()
d["e_anc"] = (d.sbp_calib - d.sbp_target).abs()
print("(a) correlação variação prevista vs real, 0-30 min, após mediana móvel centrada dentro da sessão")
for nome, w in (("sem suavizar", 1), ("2 min (9 blocos)", 9), ("5 min (21 blocos)", 21)):
    g = d.groupby("session_id", sort=False)
    r = g.dv_real.transform(lambda s: s.rolling(w, center=True, min_periods=1).median()) if w > 1 else d.dv_real
    p = g.dv_prev.transform(lambda s: s.rolling(w, center=True, min_periods=1).median()) if w > 1 else d.dv_prev
    m = d.idade_min <= 30
    por = pd.DataFrame({"s": d.subject_id[m], "r": r[m], "p": p[m]}).groupby("s").apply(lambda x: np.corrcoef(x.r, x.p)[0, 1] if len(x) >= 20 and x.r.std() > 0.5 and x.p.std() > 1e-6 else np.nan).dropna()
    print(f"  {nome:18s} agregada {np.corrcoef(r[m], p[m])[0,1]:.2f} | mediana por doente {por.median():.2f} (n={len(por)})")
print("(b) 0-30 min, MAE do modelo e da âncora por |variação real|")
s = d[d.idade_min <= 30].copy()
s["tam"] = pd.cut(s.dv_real.abs(), [0, 5, 10, 20, 1e3], labels=["<5", "5-10", "10-20", ">20"], right=False)
for k, g in s.groupby("tam", observed=True):
    print(f"  |Δ real| {k:6s} n={len(g):5d} ({100*len(g)/len(s):4.1f} %) | MAE âncora {g.e_anc.mean():5.2f} modelo {g.e_mod.mean():5.2f} ganho {g.e_anc.mean()-g.e_mod.mean():+.2f} | direção certa {100*(np.sign(g.dv_prev)==np.sign(g.dv_real)).mean():.0f} %")
