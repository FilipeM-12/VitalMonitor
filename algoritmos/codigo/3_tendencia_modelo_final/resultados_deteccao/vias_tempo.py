"""vias_tempo.py — vias exploradas para melhorar a deteção, calculadas com o modelo final (43 características, janela de 5 min por tempo exato).

Calcula, sobre as previsões de resultados_deteccao/previsoes.npz (regressor quantílico: fora da dobra no desenvolvimento, treino completo no teste) e sobre as séries
na grade de 15 s (blocos sem score = vazio), as vias do Apêndice «Vias Exploradas» e o portão de estabilidade:
  · regressão quantílica contra classificador (o classificador é treinado aqui, com best_params_clf_<direção>_tudo.json);
  · portão de estabilidade (std, amp, dif e rel; 4, 8, 12 e 20 blocos; limites nos percentis 50, 75, 90, 95 e 99 do desenvolvimento), escolhido no desenvolvimento;
  · persistência (alarmes com menos de M blocos são cortados), histerese (liga em tau, desliga abaixo de r*tau), suavização do score (média móvel dos k últimos blocos)
    e limite de duração do alarme (corta cada alarme a N blocos); o parâmetro de cada via escolhe-se no desenvolvimento.
Cada via compara-se ao limiar simples A CUSTO IGUALADO: a fronteira (sensibilidade crescente com os minutos de falso alarme por hora, ciclo de serviço <= 30 %) interpola-se a
4, 5, ..., 10 min/h, por direção, como em causal09/07_portao_causal.py. Escolhas no desenvolvimento (previsões fora da dobra), medida no teste uma só vez.
O ponto de operação conjunto com e sem portão repete a escolha da união (maximin, 10 min/h, ciclo <= 30 %) com os alarmes bloqueados pelo portão.
As funções de alarme, episódio e carga são as do notebook 03 (extraídas por AST).
Escreve vias_tempo.json e vias_tempo.csv nesta pasta.     Uso: python3 vias_tempo.py [n_threads]
"""
import ast
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

AQUI = os.path.dirname(os.path.abspath(__file__))
A = "/home/filipe/Desktop/Tese/algoritmo-ml"
TF = f"{A}/Tentativa_22_10/tendencia_tempo_fixo"
NB = f"{A}/ENTREGA_novo/03_tendencia_regressao_quantilica.ipynb"
NJ = int(sys.argv[1]) if len(sys.argv) > 1 else 14
sys.path.insert(0, TF)
import tempo_fixo as tf  # noqa: E402

DIRS = ("descida", "subida")
BLOCO_SEG, GAP_MERGE, PRE_CREDITO, DUTY_MAX = 15, 4, 8, 0.30
ALVOS = np.array([4, 5, 6, 7, 8, 9, 10])
NS = dict(np=np, pd=pd, DIRS=DIRS, BLOCO_SEG=BLOCO_SEG, GAP_MERGE=GAP_MERGE, PRE_CREDITO=PRE_CREDITO, DUTY_MAX=DUTY_MAX)
cells = json.load(open(NB))["cells"]
for i in (17, 19):
    for no in ast.parse("".join(cells[i]["source"])).body:
        if isinstance(no, ast.FunctionDef) or (isinstance(no, ast.Assign) and getattr(no.targets[0], "id", "") in ("SEP",)):
            exec(compile(ast.Module([no], []), NB, "exec"), NS)
runs_np, mascara_eventos, falsos, concatenar, preencher = (NS[k] for k in ("runs_np", "mascara_eventos", "falsos", "concatenar", "preencher"))


def detecta(al, ev):
    """Mesmo critério do notebook (um alarme que toque o episódio ou os 2 min antes), vetorizado: os alarmes são disjuntos e ordenados."""
    if al.size == 0 or ev.size == 0:
        return 0
    i = np.searchsorted(al[:, 1], ev[:, 0] - PRE_CREDITO, side="left")
    ok = i < len(al)
    return int((al[np.minimum(i, len(al) - 1), 0][ok] <= ev[ok, 1]).sum())

# ── dados ─────────────────────────────────────────────────────────────────────────────────────────
dev_g, V = tf.preparar_grade(f"{TF}/dados", "dev")
hold_g, _ = tf.preparar_grade(f"{TF}/dados", "holdout")
F = V["tudo"]
okd, okh = dev_g.delta.notna().values, hold_g.delta.notna().values
z = np.load(f"{AQUI}/previsoes.npz")


def montar(g, ok, P):
    x = g[["session_id", "bloco_id"]].copy()
    for d in DIRS:
        p = np.full(len(g), np.nan); p[ok] = P[d]
        x[f"p_{d}"] = p
        x[f"y_{d}"] = g[f"y_{d}"].fillna(0).values
    return x


# previsões do classificador no teste (treino completo no desenvolvimento)
t0 = time.time()
DEV_A, HOLD_A = dev_g[okd].reset_index(drop=True), hold_g[okh].reset_index(drop=True)
PC = {}
for d in DIRS:
    bp = json.load(open(f"{TF}/best_params_clf_{d}_tudo.json"))["best_params"]
    y = DEV_A[f"y_{d}"].values
    m = XGBClassifier(**bp, scale_pos_weight=(len(y) - y.sum()) / max(y.sum(), 1), eval_metric="auc", n_jobs=NJ, tree_method="hist").fit(DEV_A[F].values, y)
    PC[d] = m.predict_proba(HOLD_A[F].values)[:, 1]
print(f"classificador treinado em {time.time() - t0:.0f} s", flush=True)

cols = ["p_descida", "p_subida", "y_descida", "y_subida"]
CAT = {"dev": concatenar(montar(dev_g, okd, {d: z[f"oof_{d}"] for d in DIRS}), cols),
       "teste": concatenar(montar(hold_g, okh, {d: z[f"teste_{d}"] for d in DIRS}), cols)}
CAT_CLF = concatenar(montar(hold_g, okh, PC), cols)


def prepara(cat, d):
    p, y = cat[f"p_{d}"], cat[f"y_{d}"].astype(bool)
    n = len(p)
    ev = runs_np(y)
    return dict(p=p, ev=ev, cs=mascara_eventos(ev, n), n=n, n_real=int(np.isfinite(p).sum()), ev_len=len(ev),
                cs_any=mascara_eventos(runs_np(cat["y_descida"].astype(bool) | cat["y_subida"].astype(bool)), n))


S = {(c, d): prepara(CAT[c], d) for c in ("dev", "teste") for d in DIRS}


# ── alarmes de cada via ───────────────────────────────────────────────────────────────────────────────
def estatistica(p, tipo, n):
    s = pd.Series(p).shift(1)
    if tipo == "std":
        return s.rolling(n, min_periods=n).std().values
    if tipo == "amp":
        r = s.rolling(n, min_periods=n)
        return (r.max() - r.min()).values
    if tipo == "dif":
        return s.diff().rolling(n, min_periods=n).std().values
    v = s.rolling(n, min_periods=n).std()                      # rel
    return (v / (s.expanding(min_periods=n).std() + 1e-6)).values


def alarmes(p, tau, ok=None, M=1, r_hist=None, cap=None):
    mask = p >= tau
    if ok is not None:
        mask = mask & ok
    if r_hist is None:
        al = runs_np(mask)
    else:                                                      # histerese: liga em tau, desliga abaixo de r*tau
        lo = (p >= r_hist * tau) & (ok if ok is not None else True)
        cs_hi = np.concatenate([[0], np.cumsum(mask)])
        out = []
        for a0, a1 in runs_np(lo):
            if cs_hi[a1 + 1] - cs_hi[a0] > 0:
                ini = a0 + int(np.argmax(mask[a0:a1 + 1]))
                out.append((ini, a1))
        al = np.array(out, int) if out else np.empty((0, 2), int)
    if M > 1 and al.size:
        al = al[(al[:, 1] - al[:, 0] + 1) >= M]
    if cap and al.size:
        al = np.column_stack([al[:, 0], np.minimum(al[:, 1], al[:, 0] + cap - 1)])
    return al


def metricas(s, al):
    det = detecta(al, s["ev"])
    fa_n, fa_b = falsos(al, s["cs"])
    horas = s["n_real"] * BLOCO_SEG / 3600
    bl = int((al[:, 1] - al[:, 0] + 1).sum()) if al.size else 0
    return dict(sens=det / max(s["ev_len"], 1), FA_min_h=fa_b * BLOCO_SEG / 60 / horas, duty=bl / s["n_real"])


def fronteira(linhas):
    d = pd.DataFrame(linhas)
    d = d[d.duty <= DUTY_MAX].sort_values("FA_min_h")
    fr, mx = [], -1
    for _, r in d.iterrows():
        if r["sens"] > mx:
            fr.append(r); mx = r["sens"]
    return pd.DataFrame(fr)


def sens_a_custo(fr):
    if len(fr) < 2:
        return np.full(len(ALVOS), np.nan)
    return np.interp(ALVOS, fr.FA_min_h.values, fr.sens.values, left=np.nan, right=fr.sens.values[-1])


def varre(s, taus, **kw):
    return sens_a_custo(fronteira([dict(tau=t, **metricas(s, alarmes(kw.get("p", s["p"]), t, **{k: v for k, v in kw.items() if k != "p"}))) for t in taus]))


def grade_taus(p, n=100):
    return np.unique(np.round(np.nanpercentile(p[np.isfinite(p)], np.linspace(30, 99.95, n)), 4))


def ganho(base, v):
    g = v - base
    return g, float(np.nanmean(g)), float(g[-1])


R = {}
LIN = []


def registar(via, d, param, g, media, a10, escolhido=False):
    LIN.append(dict(via=via, direcao=d, parametro=param, ganho_medio_4a10=media, ganho_10=a10, escolhido_no_dev=escolhido))


# linha de base e quantílica contra classificador
BASE = {}
for d in DIRS:
    taus = grade_taus(S[("dev", d)]["p"])
    BASE[d] = dict(taus=taus, dev=varre(S[("dev", d)], taus), teste=varre(S[("teste", d)], taus))
    sc = prepara(CAT_CLF, d)
    tc = grade_taus(sc["p"])
    cl = varre(sc, tc)
    g, m, a10 = ganho(cl, BASE[d]["teste"])
    registar("quantilica_contra_classificador", d, "", g, m, a10, True)
    print(d, "base teste", np.round(BASE[d]["teste"], 3), "classificador", np.round(cl, 3), flush=True)

# portão
GATE = {}
for d in DIRS:
    sd, st = S[("dev", d)], S[("teste", d)]
    melhor, mg = None, -9
    for tipo in ("std", "amp", "dif", "rel"):
        for n in (4, 8, 12, 20):
            est_d = estatistica(sd["p"], tipo, n)
            for pc in (50, 75, 90, 95, 99):
                lim = float(np.nanpercentile(est_d[np.isfinite(est_d)], pc))
                ok = ~np.isfinite(est_d) | (est_d <= lim)
                taus = grade_taus(np.where(ok, sd["p"], np.nan), 60)
                v = varre(sd, taus, ok=ok)
                g, m, a10 = ganho(BASE[d]["dev"], v)
                if np.isfinite(m) and m > mg:
                    mg, melhor = m, (tipo, n, pc, lim)
    tipo, n, pc, lim = melhor
    est_t = estatistica(st["p"], tipo, n)
    ok_t = ~np.isfinite(est_t) | (est_t <= lim)
    est_d = estatistica(sd["p"], tipo, n)
    ok_d = ~np.isfinite(est_d) | (est_d <= lim)
    taus = grade_taus(np.where(ok_d, sd["p"], np.nan))
    v = varre(st, taus, ok=ok_t)
    g, m, a10 = ganho(BASE[d]["teste"], v)
    GATE[d] = dict(tipo=tipo, n=n, percentil=pc, limite=lim, ok_d=ok_d, ok_t=ok_t, sens_teste=v)
    registar("portao", d, f"{tipo}, {n} blocos, limite {lim:.2f} mmHg (percentil {pc})", g, m, a10, True)
    R.setdefault("portao", {})[d] = dict(tipo=tipo, n=n, percentil=pc, limite=lim, ganho_dev=mg, ganho_teste_medio=m, ganho_teste_10=a10)
    print(d, "portão", melhor, "dev", round(mg, 4), "teste médio", round(m, 4), "a 10", round(a10, 4), flush=True)

# persistência, histerese, suavização, limite de duração
for d in DIRS:
    sd, st = S[("dev", d)], S[("teste", d)]
    taus = BASE[d]["taus"]
    for M in (3, 5, 7, 9, 11, 13, 15, 19, 23):
        g, m, a10 = ganho(BASE[d]["teste"], varre(st, taus, M=M))
        registar("persistencia", d, f"{M * BLOCO_SEG} s", g, m, a10)
    cand = {}
    for r in (0.5, 0.6, 0.7, 0.8, 0.9):
        cand[r] = ganho(BASE[d]["dev"], varre(sd, taus, r_hist=r))[1]
    r_best = max(cand, key=lambda k: cand[k] if np.isfinite(cand[k]) else -9)
    for r in cand:
        g, m, a10 = ganho(BASE[d]["teste"], varre(st, taus, r_hist=r))
        registar("histerese", d, f"limiar inferior {r:.1f} x limiar", g, m, a10, r == r_best)
    for k in (2, 4, 6, 8):
        ps_d = pd.Series(sd["p"]).rolling(k, min_periods=k).mean().values
        ps_t = pd.Series(st["p"]).rolling(k, min_periods=k).mean().values
        tk = grade_taus(ps_d)
        gd = ganho(BASE[d]["dev"], varre(sd, tk, p=ps_d))
        g, m, a10 = ganho(BASE[d]["teste"], varre(st, tk, p=ps_t))
        registar("suavizacao", d, f"{k} blocos", g, m, a10)
    cand = {}
    for N in (4, 8, 16, 32):
        cand[N] = ganho(BASE[d]["dev"], varre(sd, taus, cap=N))[1]
    n_best = max(cand, key=lambda k: cand[k] if np.isfinite(cand[k]) else -9)
    for N in cand:
        g, m, a10 = ganho(BASE[d]["teste"], varre(st, taus, cap=N))
        registar("limite_duracao", d, f"{N * BLOCO_SEG // 60} min" if N * BLOCO_SEG % 60 == 0 else f"{N * BLOCO_SEG} s", g, m, a10, N == n_best)
    print(d, "persistência, histerese, suavização e duração feitas", flush=True)

pd.DataFrame(LIN).to_csv(f"{AQUI}/vias_tempo.csv", index=False)

# figura do portão: sensibilidade a custo igualado, sem e com portão, no teste
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs(f"{AQUI}/figuras", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
for a_, d, cor, nome in ((ax[0], "descida", "#c0392b", "Downward"), (ax[1], "subida", "#1f5fa8", "Upward")):
    a_.plot(ALVOS, BASE[d]["teste"], "o-", color="#666666", lw=1.6, label="no gate")
    a_.plot(ALVOS, GATE[d]["sens_teste"], "s-", color=cor, lw=1.8, label="with gate")
    a_.set_title(nome, fontsize=10); a_.set_xlabel("False-alarm minutes per hour"); a_.grid(ls=":", alpha=0.6); a_.legend(loc="lower right", fontsize=8)
ax[0].set_ylabel("Sensitivity per episode")
fig.tight_layout(); fig.savefig(f"{AQUI}/figuras/portao_tempo.png", dpi=200, bbox_inches="tight")


# ── ponto de operação conjunto com e sem portão ───────────────────────────────────────────────────────
def uniao(cat, taus, oks, n_real):
    n = len(cat["p_descida"])
    ev = {d: runs_np(cat[f"y_{d}"].astype(bool)) for d in DIRS}
    cs_any = mascara_eventos(runs_np(cat["y_descida"].astype(bool) | cat["y_subida"].astype(bool)), n)
    M = {d: {} for d in DIRS}
    for d in DIRS:
        for t in taus[d]:
            al = alarmes(cat[f"p_{d}"], t, ok=oks[d])
            M[d][float(t)] = (al, preencher(np.zeros(n, bool), al), detecta(al, ev[d]))
    horas = n_real * BLOCO_SEG / 3600
    linhas = []
    for td in taus["descida"]:
        for ts in taus["subida"]:
            m = M["descida"][float(td)][1] | M["subida"][float(ts)][1]
            al_u = runs_np(m)
            fa_n, fa_b = falsos(al_u, cs_any)
            bl = int((al_u[:, 1] - al_u[:, 0] + 1).sum()) if al_u.size else 0
            linhas.append(dict(tau_d=float(td), tau_s=float(ts), sens_d=M["descida"][float(td)][2] / max(len(ev["descida"]), 1),
                               sens_s=M["subida"][float(ts)][2] / max(len(ev["subida"]), 1), fa_min_h=fa_b * BLOCO_SEG / 60 / horas, duty=m.sum() / n_real,
                               alarmes_h=len(al_u) / horas, falsos_h=fa_n / horas, verd_h=(len(al_u) - fa_n) / horas, verd_min_h=(bl - fa_b) * BLOCO_SEG / 60 / horas,
                               vpp=1 - fa_n / max(len(al_u), 1)))
    return pd.DataFrame(linhas)


n_real = {c: int(np.isfinite(CAT[c]["p_descida"]).sum()) for c in CAT}
CP = {}
for nome, usar in (("sem_portao", False), ("com_portao", True)):
    oks = {c: {d: (GATE[d][f"ok_{'d' if c == 'dev' else 't'}"] if usar else None) for d in DIRS} for c in CAT}
    taus = {d: np.round(np.linspace(*np.nanpercentile(np.where(oks["dev"][d], CAT["dev"][f"p_{d}"], np.nan) if usar else CAT["dev"][f"p_{d}"], [2, 98]), 18), 2) for d in DIRS}
    gd = uniao(CAT["dev"], taus, oks["dev"], n_real["dev"])
    gt = uniao(CAT["teste"], taus, oks["teste"], n_real["teste"])
    c = gd[(gd.fa_min_h <= 10) & (gd.duty <= DUTY_MAX)]
    e = c.loc[np.minimum(c.sens_d, c.sens_s).idxmax()]
    h = gt[(gt.tau_d == e.tau_d) & (gt.tau_s == e.tau_s)].iloc[0]
    CP[nome] = {k: float(h[k]) for k in ("tau_d", "tau_s", "sens_d", "sens_s", "fa_min_h", "duty", "alarmes_h", "falsos_h", "verd_h", "verd_min_h", "vpp")}
    print(nome, {k: round(v, 3) for k, v in CP[nome].items()}, flush=True)
R["ponto_conjunto"] = CP
R["vias"] = LIN
for k in list(R["portao"]):
    pass
json.dump(R, open(f"{AQUI}/vias_tempo.json", "w"), indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("FIM")
