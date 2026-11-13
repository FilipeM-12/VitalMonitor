"""sens_ecg_todas.py — o ganho de AUC do ECG (ablacao_ecg_todas.py) traduz-se em sensibilidade ao mesmo custo de falsos alarmes? Só XGBoost, só desenvolvimento para decidir.

Reconstrói as características do ECG como em ablacao_ecg_todas.py (mesmo código), volta a calcular as previsões fora da dobra do braço B (PPG + ECG) com os hiperparâmetros já afinados
(best_params_B_<direção>.json) e as do braço A (só PPG) lê de resultados_deteccao/previsoes.npz (o modelo adotado). Depois repete, para A e para B, a escolha do ponto de operação da UNIÃO
dos dois detetores como no resto do Capítulo 6 (funções do notebook 03; maximin das duas sensibilidades, falso alarme da união até 10 min/h, ciclo de serviço até 30 %, escolhido no
desenvolvimento) e mede o teste uma só vez. O braço A tem de reproduzir 0,668/0,657 (Tabela 6.8).     Uso: python3 sens_ecg_todas.py [n_threads]
"""
import ast
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

AQUI = os.path.dirname(os.path.abspath(__file__))
A = "/home/filipe/Desktop/Tese/algoritmo-ml"
TF = f"{A}/Tentativa_22_10/tendencia_tempo_fixo"
RD = f"{A}/Tentativa_22_10/resultados_deteccao"
NB = f"{A}/ENTREGA_novo/03_tendencia_regressao_quantilica.ipynb"
NJ = int(sys.argv[1]) if len(sys.argv) > 1 else 15
sys.path.insert(0, TF)
sys.path.insert(0, f"{A}/03_ABSOLUTO_codigo_e_resultados/CODIGO")
import tempo_fixo as tf  # noqa: E402
from suavizacao_e_normalizacao_causal import normalizar_causal_calibracao  # noqa: E402

DIRS = ("descida", "subida")
TAU = {'descida': 0.10, 'subida': 0.90}
SINAL = {'descida': -1.0, 'subida': 1.0}
BLOCO_SEG, GAP_MERGE, PRE_CREDITO, DUTY_MAX = 15, 4, 8, 0.30
NS = dict(np=np, pd=pd, DIRS=DIRS, BLOCO_SEG=BLOCO_SEG, GAP_MERGE=GAP_MERGE, PRE_CREDITO=PRE_CREDITO, DUTY_MAX=DUTY_MAX)
cells = json.load(open(NB))["cells"]
for i in (17, 19):
    for no in ast.parse("".join(cells[i]["source"])).body:
        if isinstance(no, ast.FunctionDef) or (isinstance(no, ast.Assign) and getattr(no.targets[0], "id", "") in ("SEP",)):
            exec(compile(ast.Module([no], []), NB, "exec"), NS)
runs_np, mascara_eventos, falsos, concatenar, preencher = (NS[k] for k in ("runs_np", "mascara_eventos", "falsos", "concatenar", "preencher"))


def detecta(al, ev):
    if al.size == 0 or ev.size == 0:
        return 0
    i = np.searchsorted(al[:, 1], ev[:, 0] - PRE_CREDITO, side="left")
    ok = i < len(al)
    return int((al[np.minimum(i, len(al) - 1), 0][ok] <= ev[ok, 1]).sum())


def uniao(cat, taus, n_real):
    n = len(cat["p_descida"])
    ev = {d: runs_np(cat[f"y_{d}"].astype(bool)) for d in DIRS}
    cs_any = mascara_eventos(runs_np(cat["y_descida"].astype(bool) | cat["y_subida"].astype(bool)), n)
    M = {d: {} for d in DIRS}
    for d in DIRS:
        for t in taus[d]:
            al = runs_np(cat[f"p_{d}"] >= t)
            M[d][float(t)] = (al, preencher(np.zeros(n, bool), al), detecta(al, ev[d]))
    horas = n_real * BLOCO_SEG / 3600
    linhas = []
    for td in taus["descida"]:
        for ts in taus["subida"]:
            m = M["descida"][float(td)][1] | M["subida"][float(ts)][1]
            al_u = runs_np(m)
            fa_n, fa_b = falsos(al_u, cs_any)
            linhas.append(dict(tau_d=float(td), tau_s=float(ts), sens_d=M["descida"][float(td)][2] / max(len(ev["descida"]), 1),
                               sens_s=M["subida"][float(ts)][2] / max(len(ev["subida"]), 1), fa_min_h=fa_b * BLOCO_SEG / 60 / horas, duty=m.sum() / n_real,
                               alarmes_h=len(al_u) / horas, falsos_h=fa_n / horas, vpp=1 - fa_n / max(len(al_u), 1)))
    return pd.DataFrame(linhas)


def montar(g, ok, P):
    x = g[["session_id", "bloco_id"]].copy()
    for d in DIRS:
        p = np.full(len(g), np.nan); p[ok] = P[d]
        x[f"p_{d}"] = p
        x[f"y_{d}"] = g[f"y_{d}"].fillna(0).values
    return x


def ponto(dev_g, okd, Pd, hold_g, okh, Pt):
    cols = ["p_descida", "p_subida", "y_descida", "y_subida"]
    cd = concatenar(montar(dev_g, okd, Pd), cols); ct = concatenar(montar(hold_g, okh, Pt), cols)
    nd, nt = int(np.isfinite(cd["p_descida"]).sum()), int(np.isfinite(ct["p_descida"]).sum())
    taus = {d: np.round(np.linspace(*np.nanpercentile(cd[f"p_{d}"], [2, 98]), 18), 2) for d in DIRS}
    gd, gt = uniao(cd, taus, nd), uniao(ct, taus, nt)
    c = gd[(gd.fa_min_h <= 10) & (gd.duty <= DUTY_MAX)]
    e = c.loc[np.minimum(c.sens_d, c.sens_s).idxmax()]
    h = gt[(gt.tau_d == e.tau_d) & (gt.tau_s == e.tau_s)].iloc[0]
    return {**{f"dev_{k}": float(e[k]) for k in ("tau_d", "tau_s", "sens_d", "sens_s", "fa_min_h", "duty", "falsos_h", "vpp")},
            **{f"teste_{k}": float(h[k]) for k in ("sens_d", "sens_s", "fa_min_h", "duty", "alarmes_h", "falsos_h", "vpp")}}


# ── dados: as 43 características e o ECG, como em ablacao_ecg_todas.py ──────────────────────────────────────────────────────────────
ECG_BRUTO = ['qrs_duration', 'r_amplitude', 'qrs_amplitude', 'qrs_area', 'qt_interval', 'tq_interval', 'qtc_interval', 'sdi_interaction', 'pat_mean']
partes = []
for g in ("homens", "mulheres"):
    d = pd.read_csv(f"/media/filipe/Dados/Features/features_{g}_baseline.csv", usecols=['subject_id', 'segment_name', 'segment_order', 'window_idx'] + ECG_BRUTO)
    d.loc[~d.pat_mean.between(0.15, 0.60), 'pat_mean'] = np.nan
    d['record_id'] = d.segment_name.astype(str).apply(lambda x: x.rsplit('_', 1)[0]); d['bloco_id'] = d.window_idx // 3; d['subject_id'] = d.subject_id.astype(str)
    partes.append(d.groupby(['subject_id', 'record_id', 'segment_order', 'bloco_id'])[ECG_BRUTO].mean().reset_index())
ECG = pd.concat(partes, ignore_index=True)


def carregar(nome):
    g, V = tf.preparar_grade(TF + "/dados", nome)
    tab = pd.read_csv(f"{TF}/dados/{nome}_ppg.csv", usecols=['subject_id', 'session_id', 'bloco_id', 'record_id', 'segment_order'])
    tab['subject_id'] = tab.subject_id.astype(str); tab['record_id'] = tab.record_id.astype(str)
    tab = tab.sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
    tab = tab.merge(ECG, on=['subject_id', 'record_id', 'segment_order', 'bloco_id'], how='left')
    tab = normalizar_causal_calibracao(tab, ECG_BRUTO, n_calib=10, group_col='session_id', sufixo='_z')
    zc = [c + '_z' for c in ECG_BRUTO]
    g = g.merge(tab[['session_id', 'bloco_id'] + zc], on=['session_id', 'bloco_id'], how='left')
    g = g.sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
    gb = g.groupby('session_id'); novas = list(zc)
    for c in zc:
        for m in (2, 5, 10):
            g[f'd_{c}_{m}m'] = g[c] - gb[c].shift(int(m * 60 / BLOCO_SEG)); novas.append(f'd_{c}_{m}m')
    for c in zc:
        g[f'vol_{c}_5m'] = gb[c].transform(lambda s: s.rolling(20, min_periods=20).std()); novas.append(f'vol_{c}_5m')
    return g.replace([np.inf, -np.inf], np.nan), V['tudo'], novas


dev_g, FEATS, FEATS_ECG = carregar('dev'); hold_g, _, _ = carregar('holdout')
okd, okh = dev_g.delta.notna().values, hold_g.delta.notna().values
z = np.load(f"{RD}/previsoes.npz")
OOF_A = {d: z[f"oof_{d}"] for d in DIRS}; TESTE_A = {d: z[f"teste_{d}"] for d in DIRS}
assert len(OOF_A["descida"]) == okd.sum() and len(TESTE_A["descida"]) == okh.sum()
for d in DIRS:   # verificação: a ordem das linhas é a das previsões guardadas
    y = dev_g.loc[okd, f"y_{d}"].values
    print(d, "AUC dev do braço A pelas previsões guardadas:", round(roc_auc_score(y, OOF_A[d]), 4), flush=True)
RES = {}
t0 = time.time()
RES["A_so_PPG"] = ponto(dev_g, okd, OOF_A, hold_g, okh, TESTE_A)
print("A só PPG (modelo adotado):", {k: round(v, 3) for k, v in RES["A_so_PPG"].items()}, f"[{time.time() - t0:.0f} s]", flush=True)
json.dump(RES, open(f"{AQUI}/sens_ecg_todas.json", "w"), indent=1)

# ── braço B: previsões fora da dobra e no teste ─────────────────────────────────────────────────────────────────────────────────────
OOF_B, TESTE_B = {}, {}
di, hi = dev_g[okd], hold_g[okh]
for d in DIRS:
    bp = json.load(open(f"{AQUI}/best_params_B_{d}.json"))
    for k in ("n_estimators", "max_depth", "min_child_weight"):
        bp[k] = int(bp[k])
    X = di[FEATS + FEATS_ECG].values.astype(np.float32); yb, yr, suj = di[f"y_{d}"].values, di["delta"].values, di["subject_id"].values
    oof = np.zeros(len(yr))
    for tr, va in GroupKFold(n_splits=3).split(X, yb, suj):
        oof[va] = SINAL[d] * XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=TAU[d], n_jobs=NJ, tree_method='hist').fit(X[tr], yr[tr]).predict(X[va])
    OOF_B[d] = oof
    m = XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=TAU[d], n_jobs=NJ, tree_method='hist').fit(X, yr)
    TESTE_B[d] = SINAL[d] * m.predict(hi[FEATS + FEATS_ECG].values.astype(np.float32))
    print(d, "AUC dev braço B:", round(roc_auc_score(yb, oof), 4), "teste:", round(roc_auc_score(hi[f'y_{d}'].values, TESTE_B[d]), 4), f"[{time.time() - t0:.0f} s]", flush=True)
np.savez(f"{AQUI}/previsoes_B.npz", **{f"oof_{d}": OOF_B[d] for d in DIRS}, **{f"teste_{d}": TESTE_B[d] for d in DIRS})
RES["B_PPG_ECG"] = ponto(dev_g, okd, OOF_B, hold_g, okh, TESTE_B)
print("B PPG + ECG:", {k: round(v, 3) for k, v in RES["B_PPG_ECG"].items()}, flush=True)
json.dump(RES, open(f"{AQUI}/sens_ecg_todas.json", "w"), indent=1)
print("FIM")
