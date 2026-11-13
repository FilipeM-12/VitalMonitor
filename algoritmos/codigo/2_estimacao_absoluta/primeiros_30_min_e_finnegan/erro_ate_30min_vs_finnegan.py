#!/usr/bin/env python3
"""Erro do modelo absoluto (XGB baseline, 27 caract., holdout) nos primeiros 30 min depois da calibração, para comparar com Finnegan et al. 2023
(Sci Rep 13:986), cujas sessões duram ~30 min. Não reajusta nada: usa as previsões do holdout já guardadas.
Medidas: MAE do modelo e da âncora (repetir a calibração), com IC95 por bootstrap de doentes; mediana (IQR) por doente, como na Tab. 5 deles;
correlação entre a variação prevista e a real (agregada e por doente) e razão de amplitudes. Escreve erro_ate_30min_vs_finnegan.json/.csv."""
import json
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/filipe/Desktop/Tese/algoritmo-ml/RESULTADOS_RELATORIO/Absoluto_27")
from nucleo import config as C  # noqa: E402

d = pd.read_csv(f"{C.DIR_RESULTADOS}/experiencias/XGB_baseline/holdout_predicoes.csv")
d["idade_min"] = d.tempo_desde_calib_seg / 60.0
d["e_mod"] = (d.preds_final - d.sbp_target).abs()
d["e_anc"] = (d.sbp_calib - d.sbp_target).abs()
d["dv_real"] = d.sbp_target - d.sbp_calib      # variação real desde a calibração
d["dv_prev"] = d.preds_final - d.sbp_calib     # variação prevista
FAIXAS = {"<5": (0, 5), "25-35": (25, 35), "5-15": (5, 15), "15-30": (15, 30), "<=30": (0, 30), "30-60": (30, 60), ">60": (60, 1e5), "todos": (0, 1e5)}
rng = np.random.default_rng(0)
B = 2000


def resumo(sub):
    por = sub.groupby("subject_id")
    pm = por.e_mod.mean()
    pa = por.e_anc.mean()
    q = lambda s: [float(np.median(s)), float(np.percentile(s, 25)), float(np.percentile(s, 75))]
    cor_suj = []
    for _, g in por:
        if len(g) >= 20 and g.dv_real.std() > 0.5 and g.dv_prev.std() > 1e-6:
            cor_suj.append(np.corrcoef(g.dv_real, g.dv_prev)[0, 1])
    cor_ag = float(np.corrcoef(sub.dv_real, sub.dv_prev)[0, 1]) if len(sub) > 2 else np.nan
    sd_r, sd_p = float(sub.dv_real.std()), float(sub.dv_prev.std())
    rmse_m = float(np.sqrt(((sub.preds_final - sub.sbp_target) ** 2).mean()))
    rmse_a = float(np.sqrt(((sub.sbp_calib - sub.sbp_target) ** 2).mean()))
    return dict(n_janelas=int(len(sub)), n_doentes=int(sub.subject_id.nunique()), mae_modelo=float(sub.e_mod.mean()), mae_ancora=float(sub.e_anc.mean()),
                rmse_modelo=rmse_m, rmse_ancora=rmse_a, mediana_iqr_por_doente_modelo=q(pm), mediana_iqr_por_doente_ancora=q(pa),
                corr_agregada=cor_ag, corr_por_doente_mediana=float(np.median(cor_suj)) if cor_suj else None, n_doentes_corr=len(cor_suj),
                dp_variacao_real=sd_r, dp_variacao_prevista=sd_p, razao_amplitude=sd_p / sd_r if sd_r else None)


res, linhas = {}, []
for nome, (a, b) in FAIXAS.items():
    sub = d[(d.idade_min >= a) & (d.idade_min < b if b < 1e5 else d.idade_min >= a)] if nome != "<=30" else d[d.idade_min <= 30]
    if nome == "<5":
        sub = d[d.idade_min < 5]
    r = resumo(sub)
    # bootstrap de doentes: MAE modelo, MAE âncora, diferença
    por = {s: g for s, g in sub.groupby("subject_id")}
    ks = list(por)
    mm, aa = [], []
    for _ in range(B):
        x = pd.concat([por[k] for k in rng.choice(ks, len(ks), replace=True)])
        mm.append(x.e_mod.mean())
        aa.append(x.e_anc.mean())
    mm, aa = np.array(mm), np.array(aa)
    r["ic95_mae_modelo"] = [float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))]
    r["ic95_mae_ancora"] = [float(np.percentile(aa, 2.5)), float(np.percentile(aa, 97.5))]
    r["ic95_ganho_ancora_menos_modelo"] = [float(np.percentile(aa - mm, 2.5)), float(np.percentile(aa - mm, 97.5))]
    res[nome] = r
    linhas.append(dict(faixa=nome, **{k: v for k, v in r.items() if not isinstance(v, list)}))
json.dump(res, open(f"{AQUI}/erro_ate_30min_vs_finnegan.json", "w"), indent=1)
pd.DataFrame(linhas).to_csv(f"{AQUI}/erro_ate_30min_vs_finnegan.csv", index=False)
for k, r in res.items():
    print(f"{k:6s} n={r['n_janelas']:6d} suj={r['n_doentes']:2d} | MAE modelo {r['mae_modelo']:.2f} {np.round(r['ic95_mae_modelo'],2)} | âncora {r['mae_ancora']:.2f} {np.round(r['ic95_mae_ancora'],2)} | "
          f"ganho IC {np.round(r['ic95_ganho_ancora_menos_modelo'],2)} | por doente mediana(IQR) modelo {r['mediana_iqr_por_doente_modelo'][0]:.2f} ({r['mediana_iqr_por_doente_modelo'][1]:.2f}-{r['mediana_iqr_por_doente_modelo'][2]:.2f}) âncora {r['mediana_iqr_por_doente_ancora'][0]:.2f} | "
          f"DP Δ real {r['dp_variacao_real']:.2f} prev {r['dp_variacao_prevista']:.2f} (razão {r['razao_amplitude']:.2f}) | corr ag {r['corr_agregada']:.2f}, por doente mediana {r['corr_por_doente_mediana']} (n={r['n_doentes_corr']})")
