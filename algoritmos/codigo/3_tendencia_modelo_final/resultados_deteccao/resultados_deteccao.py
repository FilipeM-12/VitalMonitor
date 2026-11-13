"""resultados_deteccao.py — números e figuras dos «resultados da deteção» do Capítulo 6, para o modelo final (43 características, janela de 5 min por tempo exato).

Refaz, com o modelo e o ponto de operação finais, o que a tese tinha para o modelo anterior (causal09/17_numeros_capitulo5.py, 19_auc_por_tempo_de_sessao.py,
25_alarmes_sessoes_sem_evento.py, 15_robustez_ciclo_servico.py, 18_importancia_quantilica.py e figuras_cap6/figura_avaliacao_tendencia.py, gerar_figuras_cap6.py),
sem alterar nenhum desses ficheiros. As funções de alarmes, episódios e carga são as do notebook ENTREGA_novo/03_tendencia_regressao_quantilica.ipynb (extraídas
dele, sem cópia), por isso os números coincidem com os que o notebook verifica.

Treina os dois regressores (hiperparâmetros de tendencia_tempo_fixo/best_params_quant_<direção>_tudo.json) em todo o desenvolvimento, e as previsões fora da dobra
(3 dobras por doente) servem para escolher o ponto de operação, como na tese. As previsões ficam em cache (previsoes.npz).

Escreve nesta pasta: numeros_deteccao.json, importancia_quant_{descida,subida}.csv, avaliacao_tendencia.json e figuras/ (avaliacao_tendencia_tempo.png,
importancia_compacta_tempo.png, serie_tendencia_tempo.png).     Uso: python3 resultados_deteccao.py [n_threads]
"""
import ast
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

AQUI = os.path.dirname(os.path.abspath(__file__))
A = "/home/filipe/Desktop/Tese/algoritmo-ml"
TF = f"{A}/Tentativa_22_10/tendencia_tempo_fixo"
NB = f"{A}/ENTREGA_novo/03_tendencia_regressao_quantilica.ipynb"
NJ = int(sys.argv[1]) if len(sys.argv) > 1 else 14
sys.path.insert(0, TF)
import tempo_fixo as tf  # noqa: E402

os.makedirs(f"{AQUI}/figuras", exist_ok=True)
DIRS = ("descida", "subida")
Q = {"descida": 0.10, "subida": 0.90}
SINAL = {"descida": -1.0, "subida": 1.0}
BLOCO_SEG, D = 15, 15
GAP_MERGE, PRE_CREDITO, DUTY_MAX = 4, 8, 0.30
VERM, AZUL, CINZA = '#c0392b', '#1f5fa8', '#7f8c8d'

# ── funções do notebook 03 (passos 7 a 9), extraídas por AST ─────────────────────────────────────────────
NS = dict(np=np, pd=pd, DIRS=DIRS, BLOCO_SEG=BLOCO_SEG, GAP_MERGE=GAP_MERGE, PRE_CREDITO=PRE_CREDITO, DUTY_MAX=DUTY_MAX)
cells = json.load(open(NB))["cells"]
for i in (17, 19, 21):
    for no in ast.parse("".join(cells[i]["source"])).body:
        if isinstance(no, ast.FunctionDef) or (isinstance(no, ast.Assign) and getattr(no.targets[0], "id", "") in ("SEP", "ALVOS_UNIAO", "SENS_ALVO", "RNG")):
            exec(compile(ast.Module([no], []), NB, "exec"), NS)
runs, runs_np, medir, sessoes, varrer, concatenar = (NS[k] for k in ("runs", "runs_np", "medir", "sessoes", "varrer", "concatenar"))
tau_para_sens, aleatorio, carga_uniao = NS["tau_para_sens"], NS["aleatorio_igual_estrutura"], NS["carga_uniao"]
SEP = NS["SEP"]

# ── dados e previsões ────────────────────────────────────────────────────────────────────────────────────
dev_g, V = tf.preparar_grade(f"{TF}/dados", "dev")
hold_g, _ = tf.preparar_grade(f"{TF}/dados", "holdout")
F = V["tudo"]
okd, okh = dev_g.delta.notna().values, hold_g.delta.notna().values
DEV_A, HOLD_A = dev_g[okd].reset_index(drop=True), hold_g[okh].reset_index(drop=True)
HP = {d: json.load(open(f"{TF}/best_params_quant_{d}_tudo.json"))["best_params"] for d in DIRS}
mk = lambda d: XGBRegressor(**HP[d], objective="reg:quantileerror", quantile_alpha=Q[d], n_jobs=NJ, tree_method="hist")
cache = f"{AQUI}/previsoes.npz"
IMP = {}
if os.path.exists(cache):
    z = np.load(cache)
    OOF = {d: z[f"oof_{d}"] for d in DIRS}
    TESTE_P = {d: z[f"teste_{d}"] for d in DIRS}
    IMP = {d: pd.read_csv(f"{AQUI}/importancia_quant_{d}.csv", index_col=0).iloc[:, 0] for d in DIRS}
else:
    OOF, TESTE_P, sv = {}, {}, {}
    for d in DIRS:
        t0 = time.time()
        X, yr, yb = DEV_A[F].values, DEV_A["delta"].values, DEV_A[f"y_{d}"].values
        oof = np.zeros(len(X))
        for tr, va in GroupKFold(n_splits=3).split(X, yb, DEV_A["subject_id"].values):
            oof[va] = SINAL[d] * mk(d).fit(X[tr], yr[tr]).predict(X[va])
        m = mk(d).fit(X, yr)
        OOF[d], TESTE_P[d] = oof, SINAL[d] * m.predict(HOLD_A[F].values)
        g = m.get_booster().get_score(importance_type="total_gain")
        s = pd.Series({f: g.get(f, g.get(f"f{i}", 0.0)) for i, f in enumerate(F)}, dtype=float)
        IMP[d] = (s / s.sum()).sort_values(ascending=False)
        IMP[d].to_csv(f"{AQUI}/importancia_quant_{d}.csv")
        print(f"{d}: previsões em {time.time() - t0:.0f} s | AUC dev {roc_auc_score(yb, oof):.4f} teste {roc_auc_score(HOLD_A[f'y_{d}'], TESTE_P[d]):.4f}", flush=True)
    np.savez_compressed(cache, **{f"oof_{d}": OOF[d] for d in DIRS}, **{f"teste_{d}": TESTE_P[d] for d in DIRS})


def montar(g, ok, P):
    x = g[["session_id", "bloco_id", "subject_id"]].copy()
    for d in DIRS:
        p = np.full(len(g), np.nan)
        p[ok] = P[d]
        x[f"p_{d}"] = p
        x[f"y_{d}"] = g[f"y_{d}"].fillna(0).values
    return x


dev, hold = montar(dev_g, okd, OOF), montar(hold_g, okh, TESTE_P)
cols = ["p_descida", "p_subida", "y_descida", "y_subida"]
cat_dev, cat_hold = concatenar(dev, cols), concatenar(hold, cols)
TAUS = {d: np.round(np.linspace(*np.nanpercentile(dev[f"p_{d}"], [2, 98]), 18), 2) for d in DIRS}
grelha_dev, *_ = varrer(cat_dev, TAUS)
grelha_teste, M_TESTE, EV_TESTE, CS_TESTE, N_REAL = varrer(cat_hold, TAUS)
c = grelha_dev[(grelha_dev.u_FA_min_h <= 10) & (grelha_dev.u_duty <= DUTY_MAX)]
e = c.loc[np.minimum(c.sens_d, c.sens_s).idxmax()]
TAU = {"descida": float(e.tau_d), "subida": float(e.tau_s)}
assert (TAU["descida"], TAU["subida"]) == (11.34, 10.92), TAU
R = {"tau": TAU}
grelha_dev.to_csv(f"{AQUI}/grelha_dev.csv", index=False)
grelha_teste.to_csv(f"{AQUI}/grelha_teste.csv", index=False)
rob = []
for alvo in (6, 10, 12):
    cc = grelha_dev[(grelha_dev.u_FA_min_h <= alvo) & (grelha_dev.u_duty <= DUTY_MAX)]
    ee = cc.loc[np.minimum(cc.sens_d, cc.sens_s).idxmax()]
    hh_ = grelha_teste[(grelha_teste.tau_d == ee.tau_d) & (grelha_teste.tau_s == ee.tau_s)].iloc[0]
    rob.append(dict(alvo=alvo, tau_d=float(ee.tau_d), tau_s=float(ee.tau_s), sens_d=float(hh_.sens_d), sens_s=float(hh_.sens_s), fa_min_h=float(hh_.u_FA_min_h), duty=float(hh_.u_duty)))
R["robustez"] = rob
ate10 = grelha_teste[grelha_teste.u_FA_min_h <= 10]
R["duty_max_teste_ate_10"] = float(ate10.u_duty.max())
R["pares_teste_acima_30"] = int((ate10.u_duty > DUTY_MAX).sum())

# ── discriminação ────────────────────────────────────────────────────────────────────────────────────────
R["auc_teste"] = {d: float(roc_auc_score(HOLD_A[f"y_{d}"], TESTE_P[d])) for d in DIRS}
R["auc_dev"] = {d: float(roc_auc_score(DEV_A[f"y_{d}"], OOF[d])) for d in DIRS}
R["prevalencia_teste"] = {d: float(HOLD_A[f"y_{d}"].mean()) for d in DIRS}
R["prevalencia_dev"] = {d: float(DEV_A[f"y_{d}"].mean()) for d in DIRS}

# ── por direção: sensibilidade, VPP, aleatório, sempre ligado, latência, episódios falhados ──────────────
SENS_ALVO = NS["SENS_ALVO"]
PRE = PRE_CREDITO
R["direcao"] = {}
for d in DIRS:
    t = pd.DataFrame({"session_id": hold.session_id.values, "y": hold[f"y_{d}"].values, "p": hold[f"p_{d}"].values})
    g = sessoes(t)
    pto = medir(g, TAU[d])
    alea = aleatorio(g, TAU[d])
    ligado = medir(g, -1e9)
    vpp = []
    for s in SENS_ALVO:
        m = medir(g, tau_para_sens(g, s, np.nanmin(t.p), np.nanmax(t.p)))
        vpp.append([s, round(m["vpp"], 4), m["n_true"], m["n_al"]])
    lat, falh_dur, det_dur, falh_pico, det_pico, carga = [], [], [], [], [], []
    for (y, p, ok) in g:
        al = runs((p >= TAU[d]) & ok)
        ev = runs(y)
        for (e0, e1) in ev:
            dur, pico = (e1 - e0 + 1) * BLOCO_SEG / 60.0, float(np.nanmax(p[e0:e1 + 1]))
            toca = [a for a in al if a[1] >= e0 - PRE and a[0] <= e1]
            if toca:
                det_dur.append(dur); det_pico.append(pico); lat.append((toca[0][0] - e0) * BLOCO_SEG)
            else:
                falh_dur.append(dur); falh_pico.append(pico)
        carga.append(sum((a1 - a0 + 1) * BLOCO_SEG / 60.0 for (a0, a1) in al if not any(a1 >= e0 - PRE and a0 <= e1 for (e0, e1) in ev)))
    carga = np.sort(np.array(carga))[::-1]
    R["direcao"][d] = dict(
        sens=pto["sens"], vpp=pto["vpp"], duty=pto["duty"], alarmes_h=pto["alarmes_h"], n_true=pto["n_true"], n_al=pto["n_al"], n_ev=pto["n_ev"], det=pto["det"],
        aleat_sens=float(alea.mean()), aleat_lo=float(np.percentile(alea, 2.5)), aleat_hi=float(np.percentile(alea, 97.5)),
        sempre_ligado=dict(sens=ligado["sens"], duty=1.0, alarmes_h=ligado["alarmes_h"]),     # ciclo de serviço 1 por definição (medir() soma os intervalos entre blocos com score, que contam a mais)
        vpp_por_sens=vpp, fator_falsos_vpp50=(vpp[0][3] - vpp[0][2]) / vpp[0][2],
        latencia_mediana_s=float(np.median(lat)), episodios=len(falh_dur) + len(det_dur), falhados=len(falh_dur),
        dur_falhados_min=float(np.median(falh_dur)), dur_detetados_min=float(np.median(det_dur)),
        pico_falhados=float(np.median(falh_pico)), pico_detetados=float(np.median(det_pico)),
        carga_top10=float(carga[:max(1, len(carga) // 10)].sum() / max(carga.sum(), 1e-9)), carga_zero=float((carga == 0).mean()))
    print(d, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R["direcao"][d].items() if k in ("sens", "vpp", "duty", "alarmes_h", "latencia_mediana_s", "fator_falsos_vpp50")}, flush=True)

# ── união no ponto adotado, soma contra união e detetores escolhidos isoladamente ──────────────────────────
h = grelha_teste[(grelha_teste.tau_d == TAU["descida"]) & (grelha_teste.tau_s == TAU["subida"])].iloc[0]
R["uniao"] = dict(sens_d=float(h.sens_d), sens_s=float(h.sens_s), fa_min_h=float(h.u_FA_min_h), fa_h=float(h.u_FA_h), duty=float(h.u_duty),
                  **carga_uniao(cat_hold, M_TESTE, CS_TESTE, N_REAL, TAU["descida"], TAU["subida"]))
R["soma_alarmes_h"] = R["direcao"]["descida"]["alarmes_h"] + R["direcao"]["subida"]["alarmes_h"]
iso = {}
for d in DIRS:
    tdev = pd.DataFrame({"session_id": dev.session_id.values, "y": dev[f"y_{d}"].values, "p": dev[f"p_{d}"].values})
    gd = sessoes(tdev)
    cand = []
    for tau in TAUS[d]:
        m = medir(gd, tau)
        if m["fa_min_h"] <= 10 and m["duty"] <= DUTY_MAX:
            cand.append((m["sens"], float(tau)))
    iso[d] = max(cand)[1]
hi = grelha_teste[(grelha_teste.tau_d == iso["descida"]) & (grelha_teste.tau_s == iso["subida"])].iloc[0]
R["isolada_10"] = dict(tau=iso, uniao_fa_min_h=float(hi.u_FA_min_h), uniao_duty=float(hi.u_duty), sens_d=float(hi.sens_d), sens_s=float(hi.sens_s))
print("união", {k: round(v, 3) for k, v in R["uniao"].items()}, "| isoladas", R["isolada_10"], flush=True)

# ── sessões sem nenhum episódio ──────────────────────────────────────────────────────────────────────────
n_al = blocos = n_ses = 0
for sid, gs in hold.groupby("session_id", sort=False):
    if gs.y_descida.sum() == 0 and gs.y_subida.sum() == 0:
        m = ((gs.p_descida.values >= TAU["descida"]) | (gs.p_subida.values >= TAU["subida"]))
        n_al += len(runs(m)); blocos += int(gs.p_descida.notna().sum()); n_ses += 1
R["sessoes_sem_evento"] = dict(sessoes=n_ses, horas=blocos * BLOCO_SEG / 3600, alarmes=n_al, alarmes_por_hora=n_al / (blocos * BLOCO_SEG / 3600))

# ── AUC por tempo desde o início da sessão, IC por bootstrap de doentes ───────────────────────────────────
FAIXAS, ROT = [0, 30, 60, 120, 240, 1e9], ["0-30", "30-60", "60-120", "120-240", ">240"]
ini = hold_g.groupby("session_id").bloco_id.transform("min")
hold["min"] = (hold_g.bloco_id - ini).values * BLOCO_SEG / 60.0
rng = np.random.default_rng(42)
suj = hold.subject_id.unique()
R["auc_faixa"] = {}
for d in DIRS:
    x = hold[hold[f"p_{d}"].notna()][["subject_id", "min", f"y_{d}", f"p_{d}"]]
    x = x.assign(faixa=pd.cut(x["min"], FAIXAS, right=False, labels=ROT))
    R["auc_faixa"][d] = {}
    por_suj = {s: gs for s, gs in x.groupby("subject_id")}
    for f in ROT:
        xf = x[x.faixa == f]
        auc = roc_auc_score(xf[f"y_{d}"], xf[f"p_{d}"])
        sub = {s: (gs[f"y_{d}"].values, gs[f"p_{d}"].values) for s, gs in xf.groupby("subject_id")}
        chaves = list(sub)
        bs = []
        for _ in range(1000):
            esc = rng.choice(chaves, len(chaves))
            yy = np.concatenate([sub[s][0] for s in esc]); pp = np.concatenate([sub[s][1] for s in esc])
            if yy.min() != yy.max():
                bs.append(roc_auc_score(yy, pp))
        R["auc_faixa"][d][f] = dict(auc=float(auc), ic=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], sessoes=int(hold.loc[xf.index, "session_id"].nunique()),
                                   blocos=int(len(xf)), blocos_evento=int(xf[f"y_{d}"].sum()), doentes=int(xf.subject_id.nunique()))
    print(d, {k: round(v["auc"], 3) for k, v in R["auc_faixa"][d].items()}, flush=True)

# ── bootstrap de doentes das medidas do ponto de operação (figura de avaliação) ───────────────────────────
toca = lambda a, e: a[1] >= e[0] - PRE and a[0] <= e[1]
COLS = ["blocos", "ev_d", "det_d", "ev_s", "det_s", "al_d", "ver_d", "al_s", "ver_s", "fa_blocos", "uniao_blocos", "al_u", "fa_u"]
lin = {}
for sid, gs in hold.groupby("session_id", sort=False):
    r = lin.setdefault(gs.subject_id.iloc[0], dict.fromkeys(COLS, 0))
    r["blocos"] += int(gs.p_descida.notna().sum())
    ev_d, ev_s = runs(gs.y_descida.values.astype(bool)), runs(gs.y_subida.values.astype(bool))
    m_d, m_s = gs.p_descida.values >= TAU["descida"], gs.p_subida.values >= TAU["subida"]
    al_d, al_s = runs(m_d), runs(m_s)
    r["ev_d"] += len(ev_d); r["ev_s"] += len(ev_s)
    r["det_d"] += sum(any(toca(a, e_) for a in al_d) for e_ in ev_d); r["det_s"] += sum(any(toca(a, e_) for a in al_s) for e_ in ev_s)
    r["al_d"] += len(al_d); r["al_s"] += len(al_s)
    r["ver_d"] += sum(any(toca(a, e_) for e_ in ev_d) for a in al_d); r["ver_s"] += sum(any(toca(a, e_) for e_ in ev_s) for a in al_s)
    fd, fs = np.zeros(len(m_d), bool), np.zeros(len(m_d), bool)         # alarmes já fundidos, como em varrer() do notebook
    for a0, a1 in al_d:
        fd[a0:a1 + 1] = True
    for a0, a1 in al_s:
        fs[a0:a1 + 1] = True
    r["uniao_blocos"] += int((fd | fs).sum())
    for a in runs(fd | fs):
        r["al_u"] += 1
        if not any(toca(a, e_) for e_ in ev_d + ev_s):
            r["fa_blocos"] += a[1] - a[0] + 1
            r["fa_u"] += 1
dfb = pd.DataFrame(lin).T.astype(float)


def metricas(t):
    hh = t.blocos * BLOCO_SEG / 3600
    return dict(sens_d=t.det_d / t.ev_d, sens_s=t.det_s / t.ev_s, fa_min_h=t.fa_blocos * BLOCO_SEG / 60 / hh, duty=100 * t.uniao_blocos / t.blocos,
                vpp_d=100 * t.ver_d / t.al_d, vpp_s=100 * t.ver_s / t.al_s, vpp_u=100 * (1 - t.fa_u / t.al_u), alarmes_h_u=t.al_u / hh)


pt = metricas(dfb.sum())
rng = np.random.default_rng(42)
vals = dfb.values
bt = pd.DataFrame([metricas(pd.Series(vals[rng.integers(0, len(dfb), len(dfb))].sum(axis=0), index=dfb.columns)) for _ in range(1000)])
ic = {k: [float(np.percentile(bt[k], 2.5)), float(np.percentile(bt[k], 97.5))] for k in pt}
R["avaliacao"] = dict(ponto={k: float(v) for k, v in pt.items()}, ic95=ic, doentes=int(len(dfb)))
print({k: (round(float(v), 3), [round(x, 3) for x in ic[k]]) for k, v in pt.items()}, flush=True)

# ── importância por grupo ────────────────────────────────────────────────────────────────────────────────
def grupo(f):
    if f.startswith("d_"):
        return "variações (2, 5 e 10 min)"
    if f == "vol_hr_mean_z_5m":
        return "volatilidade da FC"
    if f in ("age", "gender_bin"):
        return "idade e sexo"
    return "níveis (valores do bloco)"


R["importancia_grupos"] = {d: {g: float(v) for g, v in IMP[d].groupby(grupo).sum().items()} for d in DIRS}
R["importancia_top"] = {d: {k: float(v) for k, v in IMP[d].head(11).items()} for d in DIRS}

# ── sessão de exemplo (regra fixa, como em gerar_figuras_cap6.fig_serie_tendencia) ─────────────────────────
alvo_fa = R["uniao"]["fa_min_h"]
melhor, dist = None, np.inf
cand = hold.assign(t0=hold_g.groupby("session_id").bloco_id.transform("min").values)
for sid, gs in hold.groupby("session_id", sort=False):
    dur = len(gs) * BLOCO_SEG / 60
    if not (90 <= dur <= 240):
        continue
    ev_d, ev_s = runs(gs.y_descida.values.astype(bool)), runs(gs.y_subida.values.astype(bool))
    if not ev_d or not ev_s or len(ev_d) + len(ev_s) < 8:
        continue
    zz = hold_g.loc[hold_g.session_id == sid, ["hr_mean_z", "area_mean_global_z", "alpha_womersley_global_z"]].abs().max().max()
    if not zz <= 30:        # as três características desenhadas no painel (b) ficam dentro de 30 desvios padrão, para a figura se ler
        continue
    m_d, m_s = gs.p_descida.values >= TAU["descida"], gs.p_subida.values >= TAU["subida"]
    al_d, al_s = runs(m_d), runs(m_s)
    det = sum(any(toca(a, e_) for a in al_d) for e_ in ev_d) + sum(any(toca(a, e_) for a in al_s) for e_ in ev_s)
    if not (0.5 <= det / (len(ev_d) + len(ev_s)) <= 0.8):
        continue
    ev = ev_d + ev_s
    fa = sum((a1 - a0 + 1) for a0, a1 in runs(m_d | m_s) if not any(toca((a0, a1), e_) for e_ in ev)) * BLOCO_SEG / 60
    n_sc = int(gs.p_descida.notna().sum())
    fa_h = fa / (n_sc * BLOCO_SEG / 3600)
    if abs(fa_h - alvo_fa) < dist:
        melhor, dist, fa_sel = sid, abs(fa_h - alvo_fa), fa_h
gs = hold[hold.session_id == melhor].reset_index(drop=True)
gg = hold_g[hold_g.session_id == melhor].reset_index(drop=True)
t = (gg.bloco_id.values - gg.bloco_id.values[0]) * BLOCO_SEG / 60.0
ev_d, ev_s = runs(gs.y_descida.values.astype(bool)), runs(gs.y_subida.values.astype(bool))
m_d, m_s = gs.p_descida.values >= TAU["descida"], gs.p_subida.values >= TAU["subida"]
al_d, al_s = runs(m_d), runs(m_s)
al_u = runs(m_d | m_s)
ev_all = ev_d + ev_s
primeiro = min(a0 for a0, _ in ev_all)
ultimo = max(a1 for _, a1 in ev_all)
falhados = [(("descida" if e_ in ev_d else "subida"), (e_[1] - e_[0] + 1) * BLOCO_SEG / 60) for e_ in ev_all if not any(toca(a, e_) for a in (al_d if e_ in ev_d else al_s))]
antes = [a for a in al_u if a[1] < primeiro - PRE]
R["sessao_exemplo"] = dict(
    session_id=str(melhor), duracao_min=float(len(gs) * BLOCO_SEG / 60), fa_min_h=float(fa_sel), ep_descida=len(ev_d), ep_subida=len(ev_s),
    primeiro_episodio_min=float(t[primeiro]), ultimo_episodio_min=float(t[ultimo]), falhados=falhados,
    alarmes_uniao=len(al_u), alarmes_verdadeiros=sum(any(toca(a, e_) for e_ in ev_all) for a in al_u),
    alarmes_antes_do_primeiro=len(antes), pa_media_antes=float(np.nanmean(gg.sbp_target.values[: primeiro + 1])), score_inicia_min=float(t[int(np.argmax(gs.p_descida.notna().values))]))
print("sessão exemplo", R["sessao_exemplo"], flush=True)

fig, ax = plt.subplots(3, 1, figsize=(9.5, 7.2), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1, 1.2]))
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6})
ax[0].plot(t, gg.sbp_target, color=CINZA, lw=1.0)
for ev_, cor, nome in ((ev_d, VERM, "downward episode"), (ev_s, AZUL, "upward episode")):
    for k, (a0, a1) in enumerate(ev_):
        ax[0].axvspan(t[a0], t[a1] + 0.25, color=cor, alpha=0.18, lw=0, label=nome if k == 0 else None)
ax[0].set_ylabel("SBP (mmHg)"); ax[0].legend(loc="upper right", fontsize=8)
ax[0].set_title("(a) Reference pressure and events (drop or rise of at least 15 mmHg in 5 min)", fontsize=9, loc="left")
for col, cor, nome in (("hr_mean_z", "#16a085", "heart rate"), ("area_mean_global_z", "#8e44ad", "pulse area"), ("alpha_womersley_global_z", "#d35400", "Womersley index")):
    ax[1].plot(t, hold_g.loc[hold_g.session_id == melhor, col].values, color=cor, lw=1.0, label=nome)
ax[1].set_ylabel("z-score"); ax[1].legend(loc="upper right", fontsize=8, ncol=3)
ax[1].set_title("(b) Features normalised by the calibration window", fontsize=9, loc="left")
for dirn, col, al, cor, nome in (("descida", "p_descida", al_d, VERM, "downward"), ("subida", "p_subida", al_s, AZUL, "upward")):
    ax[2].plot(t, gs[col], color=cor, lw=1.0, label=f"{nome} score")
    ax[2].axhline(TAU[dirn], color=cor, ls="--", lw=1.0)
    for k, (a0, a1) in enumerate(al):
        ax[2].axvspan(t[a0], t[a1] + 0.25, ymin=0.92 if dirn == "descida" else 0.84, ymax=0.98 if dirn == "descida" else 0.90, color=cor, lw=0, label=f"{nome} alarm" if k == 0 else None)
off = R["sessao_exemplo"]["score_inicia_min"]
ax[2].axvspan(0, off, color="#dddddd", lw=0)
ax[2].text(off / 2, 15, "no score yet\n(5 min of history)", ha="center", va="center", fontsize=8, color="#555555")
ax[2].set_ylabel("Score (mmHg)"); ax[2].set_xlabel("Time since the start of the session (min)")
ax[2].set_ylim(0, 30); ax[2].legend(loc="center right", fontsize=8, ncol=2)
ax[2].set_title("(c) Model scores, decision thresholds (dashed) and alarms", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(f"{AQUI}/figuras/serie_tendencia_tempo.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ── figura de avaliação (3 painéis) ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(10, 3.3), gridspec_kw=dict(width_ratios=[1.1, 0.8, 1.1]))


def barras(a, xs, vv, cores, ics, fmt):
    for x, v, cc, (lo, hi_) in zip(xs, vv, cores, ics):
        a.bar(x, v, color=cc, width=0.6, alpha=0.9)
        a.errorbar(x, v, yerr=[[v - lo], [hi_ - v]], color="#222222", capsize=3, lw=1.2)
        a.text(x, hi_ + 1, fmt.format(v), fontsize=8, va="bottom", ha="center")


a = ax[0]
barras(a, [0, 1], [100 * pt["sens_d"], 100 * pt["sens_s"]], [VERM, AZUL], [[100 * x for x in ic["sens_d"]], [100 * x for x in ic["sens_s"]]], "{:.1f}")
for x, d in ((0, "descida"), (1, "subida")):
    a.plot([x - 0.33, x + 0.33], [100 * R["direcao"][d]["aleat_sens"]] * 2, color=CINZA, ls="--", lw=1.5)
    a.fill_between([x - 0.33, x + 0.33], 100 * R["direcao"][d]["aleat_lo"], 100 * R["direcao"][d]["aleat_hi"], color=CINZA, alpha=0.3, lw=0)
a.plot([], [], color=CINZA, ls="--", lw=1.5, label="random alarm, same structure")
a.legend(loc="upper center", fontsize=8, frameon=False)
a.set_xticks([0, 1]); a.set_xticklabels(["Downward", "Upward"]); a.set_ylim(0, 112); a.set_yticks(range(0, 101, 20)); a.set_ylabel("Episodes detected (%)"); a.set_title("(a) Sensitivity", fontsize=9, loc="left")
a = ax[1]
a.bar([0], [pt["fa_min_h"]], color="#34495e", width=0.5, alpha=0.9)
a.errorbar([0], [pt["fa_min_h"]], yerr=[[pt["fa_min_h"] - ic["fa_min_h"][0]], [ic["fa_min_h"][1] - pt["fa_min_h"]]], color="#222222", capsize=3, lw=1.2)
a.axhline(10, color=VERM, ls="--", lw=1.3); a.text(0.42, 10.2, "limit 10 min/h", fontsize=8, color=VERM, ha="right", va="bottom")
a.text(0, ic["fa_min_h"][1] + 0.3, f"{pt['fa_min_h']:.1f}", fontsize=8, va="bottom", ha="center")
a.set_xlim(-0.5, 0.5); a.set_ylim(0, 14); a.set_ylabel("False-alarm time (min/h)")
a.set_xticks([0]); a.set_xticklabels(["Both directions\n(union)"]); a.set_title("(b) Alarm burden", fontsize=9, loc="left")
a = ax[2]
barras(a, [0, 1], [pt["vpp_d"], pt["vpp_s"]], [VERM, AZUL], [ic["vpp_d"], ic["vpp_s"]], "{:.1f}")
a.set_xticks([0, 1]); a.set_xticklabels(["Downward", "Upward"]); a.set_ylim(0, 40); a.set_ylabel("Alarms that are true (%)"); a.set_title("(c) Positive predictive value", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(f"{AQUI}/figuras/avaliacao_tendencia_tempo.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ── importância compacta ─────────────────────────────────────────────────────────────────────────────────
TOP = 8
d_, s_ = IMP["descida"], IMP["subida"]
feats = sorted(set(d_.nlargest(TOP).index) | set(s_.nlargest(TOP).index), key=lambda f: (d_[f] + s_[f]) / 2)
y = np.arange(len(feats)); hh = 0.38
fig, ax = plt.subplots(figsize=(5.8, 0.36 * len(feats) + 0.9))
ax.barh(y + hh / 2, [d_[f] for f in feats], height=hh, color=VERM, label="Downward (hypotension)")
ax.barh(y - hh / 2, [s_[f] for f in feats], height=hh, color="#2471a3", label="Upward (hypertension)")
ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=8.5)
ax.set_xlabel("Importance (total gain, normalised)"); ax.legend(loc="lower right", frameon=False)
ax.grid(axis="x", alpha=0.3, ls=":"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{AQUI}/figuras/importancia_compacta_tempo.png", dpi=250); plt.close(fig)
R["importancia_figura"] = list(reversed(feats))

json.dump(R, open(f"{AQUI}/numeros_deteccao.json", "w"), indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("FIM")
