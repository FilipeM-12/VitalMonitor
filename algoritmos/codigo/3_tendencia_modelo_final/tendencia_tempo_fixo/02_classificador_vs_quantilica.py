"""02_classificador_vs_quantilica.py — classificador contra regressão quantílica (XGBoost), com a janela de 5 min ANCORADA NO TEMPO.

Repete a comparação da secção «classificação ou regressão quantílica» na definição final: as mesmas 43 características, a mesma janela por
tempo (tempo_fixo.py), os mesmos folds e o mesmo orçamento (20 tentativas de Optuna, objetivo = AUC fora da amostra, 3 dobras por sujeito).
  · o classificador (XGBClassifier, scale_pos_weight, alvo binário) é afinado aqui (estudos em optuna/optuna_clf_<direção>_tudo.db);
  · o regressor quantílico usa os hiperparâmetros de 01_optuna_tempo.py (best_params_quant_<direção>_tudo.json).
Para cada modelo: AUC (fora da amostra no desenvolvimento e no teste); e a sensibilidade de cada direção a vários níveis de carga da UNIÃO dos dois
alarmes (4, 6, 8, 10 min/h de falso alarme), com o limiar escolhido no desenvolvimento (critério maximin, ciclo de serviço <= 30 %, sem portão
e sem limite de duração, como o ponto de operação adotado) e uma só leitura no teste. Os limiares percorrem os mesmos 30 valores, entre os
percentis 90 e 99,8 do score no desenvolvimento, para os dois modelos (os scores têm escalas diferentes). A 10 min/h dá-se o IC 95 % da diferença
(quantílica - classificador) por bootstrap de doentes (1000 reamostragens, iguais para os dois modelos).
CONJUNTO=base repete tudo com as 19 características (estudos e ficheiros com sufixo _base).
Uso: [CONJUNTO=base] python3 02_classificador_vs_quantilica.py [pasta_dados] [pasta_com_04] [n_threads] [n_trials_em_paralelo]
"""
import importlib.util
import json
import os
import sys

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier, XGBRegressor

AQUI = os.path.dirname(os.path.abspath(__file__))
DADOS = sys.argv[1] if len(sys.argv) > 1 else f"{AQUI}/dados"
BASE = sys.argv[2] if len(sys.argv) > 2 else AQUI
NJOBS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
NPAR = int(sys.argv[4]) if len(sys.argv) > 4 else 3
N_TRIALS, N_BOOT, ALVOS = 20, 1000, [4, 6, 8, 10]
CONJUNTO = os.environ.get("CONJUNTO", "tudo")      # base = as 19 de antes; tudo = as 43
SUF = "" if CONJUNTO == "tudo" else f"_{CONJUNTO}"   # sufixo dos ficheiros de saída (o conjunto 'tudo' mantém os nomes já usados)
sys.path.insert(0, AQUI)
import tempo_fixo as tf  # noqa: E402
optuna.logging.set_verbosity(optuna.logging.WARNING)
spec = importlib.util.spec_from_file_location("c", f"{BASE}/04_otimizacao_conjunta_causal.py")
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
c.CAPS_MIN = [None]

dev_g, V = tf.preparar_grade(DADOS, 'dev')
hold_g, _ = tf.preparar_grade(DADOS, 'holdout')
feats = V[CONJUNTO]
dev_g['subject_id'] = dev_g.subject_id.astype(str); hold_g['subject_id'] = hold_g.subject_id.astype(str)
okd, okh = dev_g.delta.notna().values, hold_g.delta.notna().values
dev, hold = dev_g[okd].reset_index(drop=True), hold_g[okh].reset_index(drop=True)
SIGNO = {'descida': -1.0, 'subida': 1.0}
QUANTIL = {'descida': 0.10, 'subida': 0.90}


def folds_de(direcao):
    X, yb, suj = dev[feats].values, dev[f'y_{direcao}'].values, dev['subject_id'].values
    return X, yb, list(GroupKFold(n_splits=3).split(X, yb, suj))


def afinar_classificador(direcao):
    X, yb, folds = folds_de(direcao)
    pw = (len(yb) - yb.sum()) / max(yb.sum(), 1)
    FOLDS = [(X[tr], yb[tr], X[va], va) for tr, va in folds]

    def objective(trial):
        p = dict(n_estimators=trial.suggest_int('n_estimators', 200, 800, step=100), learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                 max_depth=trial.suggest_int('max_depth', 3, 9), min_child_weight=trial.suggest_int('min_child_weight', 1, 12),
                 subsample=trial.suggest_float('subsample', 0.6, 1.0), colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                 gamma=trial.suggest_float('gamma', 1e-3, 5.0, log=True), reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                 reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True))
        oof = np.zeros(len(yb))
        for Xtr, ytr, Xva, va in FOLDS:
            m = XGBClassifier(**p, scale_pos_weight=pw, eval_metric='auc', n_jobs=NJOBS, tree_method='hist')
            oof[va] = m.fit(Xtr, ytr).predict_proba(Xva)[:, 1]
        return roc_auc_score(yb, oof)
    os.makedirs(f"{AQUI}/optuna", exist_ok=True)
    s = optuna.create_study(direction="maximize", storage=f"sqlite:///{AQUI}/optuna/optuna_clf_{direcao}_{CONJUNTO}.db",
                            study_name=f"clf_{direcao}_{CONJUNTO}_tempo", load_if_exists=True)
    feitos = sum(t.state == optuna.trial.TrialState.COMPLETE for t in s.trials)
    if N_TRIALS - feitos > 0:
        print(f"classificador/{direcao}: a correr mais {N_TRIALS - feitos} tentativas", flush=True)
        s.optimize(objective, n_trials=N_TRIALS - feitos, n_jobs=NPAR)
    json.dump(dict(modo='classificador', direcao=direcao, best_cv=s.best_value, best_params=s.best_params, feats=feats),
              open(f"{AQUI}/best_params_clf_{direcao}_{CONJUNTO}.json", "w"), indent=2)
    return s.best_params


def scores(modelo, direcao, bp):
    """(oof_dev, teste) do modelo; score maior = mais risco."""
    X, yb, folds = folds_de(direcao)
    yr = dev['delta'].values
    pw = (len(yb) - yb.sum()) / max(yb.sum(), 1)

    def ajustar(Xtr, idx):
        if modelo == 'clf':
            return XGBClassifier(**bp, scale_pos_weight=pw, eval_metric='auc', n_jobs=NJOBS * NPAR, tree_method='hist').fit(Xtr, yb[idx])
        return XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=QUANTIL[direcao], n_jobs=NJOBS * NPAR, tree_method='hist').fit(Xtr, yr[idx])
    pred = (lambda m, A: m.predict_proba(A)[:, 1]) if modelo == 'clf' else (lambda m, A: SIGNO[direcao] * m.predict(A))
    oof = np.zeros(len(yb))
    for tr, va in folds:
        oof[va] = pred(ajustar(X[tr], tr), X[va])
    return oof, pred(ajustar(X, np.arange(len(X))), hold[feats].values)


def pontuacao_grade(oof_d, p_d, oof_s, p_s):
    d = dev_g[['session_id', 'bloco_id']].copy(); h = hold_g[['session_id', 'bloco_id']].copy()
    for nome, a, b in (('descida', oof_d, p_d), ('subida', oof_s, p_s)):
        x = np.full(len(dev_g), np.nan); x[okd] = a; d[f'p_{nome}'] = x
        y = np.full(len(hold_g), np.nan); y[okh] = b; h[f'p_{nome}'] = y
        d[f'y_{nome}'] = dev_g[f'y_{nome}'].fillna(0).values; h[f'y_{nome}'] = hold_g[f'y_{nome}'].fillna(0).values
    return d, h


def varrer_modelo(d, h):
    cfg = json.load(open(f"{BASE}/ponto_operacao_quantilica.json"))
    for k in cfg:
        cfg[k]["portao_limite"] = 1e9
    cols = ["p_descida", "p_subida", "y_descida", "y_subida"]
    cat = {"dev": c.concatenar(d, cols), "hold": c.concatenar(h, cols)}
    TAUS = {k: np.unique(np.round(np.nanpercentile(d[f"p_{k}"], np.linspace(90, 99.8, 30)), 4)) for k in ("descida", "subida")}
    t_dev, _, _ = c.varrer(cat["dev"], cfg, TAUS)
    t_hold, nd, ns = c.varrer(cat["hold"], cfg, TAUS)
    escolhas = {}
    for alvo in ALVOS:
        sel = t_dev[(t_dev.u_FA_min_h <= alvo) & (t_dev.u_duty <= 0.30)]
        if not len(sel):
            continue
        e = sel.loc[np.minimum(sel.sens_d, sel.sens_s).idxmax()]
        r = t_hold[(t_hold.tau_d == e.tau_d) & (t_hold.tau_s == e.tau_s)].iloc[0]
        escolhas[alvo] = dict(tau_d=float(e.tau_d), tau_s=float(e.tau_s), sens_d=float(r.sens_d), sens_s=float(r.sens_s),
                              min_falso_h=float(r.u_FA_min_h), ciclo=float(r.u_duty), alarmes_h=np.nan,
                              dev_sens_d=float(e.sens_d), dev_sens_s=float(e.sens_s), dev_min_falso_h=float(e.u_FA_min_h), dev_ciclo=float(e.u_duty))
    return escolhas


def por_doente(h, tau_d, tau_s, grade_ref=None):
    linhas = {}
    for suj, idx in (hold_g if grade_ref is None else grade_ref).groupby('subject_id', sort=False).indices.items():
        g = h.iloc[idx]
        cat = c.concatenar(g, ["p_descida", "p_subida", "y_descida", "y_subida"])
        n = len(cat['p_descida'])
        md, ms = cat['p_descida'] >= tau_d, cat['p_subida'] >= tau_s
        al_d, al_s = c.runs_np(md), c.runs_np(ms)
        ev_d, ev_s = c.runs_np(cat['y_descida'].astype(bool)), c.runs_np(cat['y_subida'].astype(bool))
        masc = lambda al: np.isin(np.arange(n), np.concatenate([np.arange(a0, a1 + 1) for a0, a1 in al]) if len(al) else [])
        m_u = masc(al_d) | masc(al_s)
        al_u = c.runs_np(m_u)
        cs_any = c.mascara_eventos(c.runs_np(cat['y_descida'].astype(bool) | cat['y_subida'].astype(bool)), n)
        fa_n, fa_b = c.falsos(al_u, cs_any)
        linhas[suj] = dict(ev_d=len(ev_d), det_d=c.detecta(al_d, ev_d), ev_s=len(ev_s), det_s=c.detecta(al_s, ev_s), al_u=len(al_u), fa_u=fa_n,
                           fa_min=fa_b * c.BLOCO_SEG / 60.0, horas=int(np.isfinite(cat['p_descida']).sum()) * c.BLOCO_SEG / 3600,
                           ciclo=int(m_u.sum()), blocos=int(np.isfinite(cat['p_descida']).sum()))
    return pd.DataFrame(linhas).T.astype(float)


def medidas(t):
    s = t.sum()
    return dict(sens_descida=s.det_d / s.ev_d, sens_subida=s.det_s / s.ev_s, min_falso_h=s.fa_min / s.horas, ciclo_servico=s.ciclo / s.blocos,
                alarmes_h=s.al_u / s.horas, vpp_uniao=1 - s.fa_u / s.al_u)


S, tabela, auc = {}, [], []
for direcao in ('descida', 'subida'):
    bp_clf = afinar_classificador(direcao)
    bp_q = json.load(open(f"{AQUI}/best_params_quant_{direcao}_{CONJUNTO}.json"))['best_params']
    for modelo, bp in (('clf', bp_clf), ('quant', bp_q)):
        o, t = scores(modelo, direcao, bp)
        S[(modelo, direcao)] = (o, t)
        a = dict(modelo=modelo, direcao=direcao, auc_dev=roc_auc_score(dev[f'y_{direcao}'], o), auc_teste=roc_auc_score(hold[f'y_{direcao}'], t))
        auc.append(a); print(a, flush=True)
pd.DataFrame(auc).to_csv(f"{AQUI}/comparacao_auc{SUF}.csv", index=False)

E, T, TD = {}, {}, {}
for modelo in ('clf', 'quant'):
    d, h = pontuacao_grade(*S[(modelo, 'descida')], *S[(modelo, 'subida')])
    E[modelo] = varrer_modelo(d, h)
    for alvo, r in E[modelo].items():
        tabela.append(dict(modelo=modelo, alvo_min_h=alvo, **r)); print(modelo, alvo, {k: round(v, 3) for k, v in r.items()}, flush=True)
    r10 = E[modelo][10]
    T[modelo] = por_doente(h, r10['tau_d'], r10['tau_s'])
    TD[modelo] = por_doente(d, r10['tau_d'], r10['tau_s'], dev_g)       # os mesmos limiares, medidos no desenvolvimento (previsões fora da amostra)
    print(modelo, '10 min/h por doente:', {k: round(v, 3) for k, v in medidas(T[modelo]).items()}, flush=True)
pd.DataFrame(tabela).to_csv(f"{AQUI}/comparacao_niveis_carga{SUF}.csv", index=False)

rng = np.random.default_rng(0)
n = len(T['clf']); amostras = [rng.choice(n, n) for _ in range(N_BOOT)]
B = {m: pd.DataFrame([medidas(T[m].iloc[i]) for i in amostras]) for m in T}
out = []
for k in B['clf'].columns:
    p = {m: medidas(T[m])[k] for m in T}; dif = B['quant'][k] - B['clf'][k]
    out.append(dict(medida=k, clf=p['clf'], clf_ic_baixo=B['clf'][k].quantile(.025), clf_ic_alto=B['clf'][k].quantile(.975), quant=p['quant'],
                    quant_ic_baixo=B['quant'][k].quantile(.025), quant_ic_alto=B['quant'][k].quantile(.975), dif=p['quant'] - p['clf'],
                    dif_ic_baixo=dif.quantile(.025), dif_ic_alto=dif.quantile(.975)))
r = pd.DataFrame(out); r.to_csv(f"{AQUI}/comparacao_bootstrap_10{SUF}.csv", index=False)
print(r.round(3).to_string(index=False))

# ── o mesmo no DESENVOLVIMENTO (previsões fora da amostra; limiares escolhidos aqui, por isso ambos os modelos são igualmente otimistas) ──
n = len(TD['clf']); amostras = [rng.choice(n, n) for _ in range(N_BOOT)]
B = {m: pd.DataFrame([medidas(TD[m].iloc[i]) for i in amostras]) for m in TD}
out = []
for k in B['clf'].columns:
    p = {m: medidas(TD[m])[k] for m in TD}; dif = B['quant'][k] - B['clf'][k]
    out.append(dict(medida=k, clf=p['clf'], clf_ic_baixo=B['clf'][k].quantile(.025), clf_ic_alto=B['clf'][k].quantile(.975), quant=p['quant'],
                    quant_ic_baixo=B['quant'][k].quantile(.025), quant_ic_alto=B['quant'][k].quantile(.975), dif=p['quant'] - p['clf'],
                    dif_ic_baixo=dif.quantile(.025), dif_ic_alto=dif.quantile(.975)))
rd = pd.DataFrame(out); rd.to_csv(f"{AQUI}/comparacao_bootstrap_10_dev{SUF}.csv", index=False)
print("DESENVOLVIMENTO"); print(rd.round(3).to_string(index=False))
pd.DataFrame([dict(modelo=m, alvo_min_h=a, **{k: v for k, v in r_.items() if k.startswith('dev_')}) for m in E for a, r_ in E[m].items()]).to_csv(f"{AQUI}/comparacao_niveis_carga_dev{SUF}.csv", index=False)
print("FIM")
