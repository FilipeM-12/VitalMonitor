"""janela_dev.py — escolha da janela temporal W (2, 3, 5 e 10 min) SÓ COM O DESENVOLVIMENTO (a janela não pode ser
escolhida com os dados de teste).

Para cada W e direção: alvo Δ_W = PA(t) - PA(t-W) por TEMPO exato (grade de 15 s), evento Δ_W <= -15 (descida) ou >= +15 (subida). Modelo: XGBoost quantílico
das 43 características (as mesmas para todas as janelas), com uma afinação própria (20 tentativas de Optuna, objetivo = AUC fora da amostra, 3 dobras por doente,
como em 01_optuna_tempo.py). Medidas, SÓ no desenvolvimento (previsões fora da amostra, 3 dobras por doente):
  auc_dev           AUC por bloco
  sens_Xminh/vpp_Xminh   sensibilidade por episódio e VPP a 4, 6, 8 e 10 min/h de falso alarme (fronteira por direção, como em sensibilidade_tempo.py)
  prevalencia       fração de blocos em evento
O teste só entra como informação, em colunas à parte (teste_*), de um modelo treinado em todo o desenvolvimento; NÃO serve para escolher W.
Retoma sozinho (estudos Optuna em optuna/, resultados por (W, direção) em cache/).   Uso: python3 janela_dev.py [pasta_dados] [n_threads] [trials_em_paralelo]
"""
import json
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

AQUI = os.path.dirname(os.path.abspath(__file__))
TF = os.path.normpath(os.path.join(AQUI, "..", "tendencia_tempo_fixo"))        # tempo_fixo.py e os best_params_*.json só existem em tendencia_tempo_fixo/
sys.path.insert(0, TF)
import tempo_fixo as tf  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
DADOS = sys.argv[1] if len(sys.argv) > 1 else f"{AQUI}/dados"
N_THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
N_PAR = int(sys.argv[3]) if len(sys.argv) > 3 else 3
D, N_TRIALS, BLOCO_SEG, GAP_MERGE, PRE_CREDITO, DUTY_MAX = 15, 20, 15, 4, 8, 0.30
JANELAS = (2, 3, 5, 10)
ALVOS = np.array([4, 6, 8, 10])
TAU = {'descida': 0.10, 'subida': 0.90}
SINAL = {'descida': -1.0, 'subida': 1.0}
os.makedirs(f"{AQUI}/optuna", exist_ok=True)
os.makedirs(f"{AQUI}/cache", exist_ok=True)

dev, V = tf.preparar_grade(DADOS, 'dev')
hold, _ = tf.preparar_grade(DADOS, 'holdout')
FEATS = V['tudo']
for d_ in (dev, hold):
    d_['subject_id'] = d_.subject_id.astype(str)


def runs(mask, gap=GAP_MERGE):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    cortes = np.where(np.diff(idx) > gap + 1)[0]
    return [(g[0], g[-1]) for g in np.split(idx, cortes + 1)]


def contagens(y, alarme, eventos):
    al = runs(alarme)
    toca = lambda a0, a1: any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (e0, e1) in eventos)
    det = sum(any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (a0, a1) in al) for (e0, e1) in eventos)
    falsos = [(a0, a1) for (a0, a1) in al if not toca(a0, a1)]
    fa = sum((a1 - a0 + 1) * BLOCO_SEG / 60.0 for (a0, a1) in falsos)
    return det, len(eventos), fa, sum(a1 - a0 + 1 for (a0, a1) in al), len(y), len(al), len(falsos)


def fronteira(soma):
    vazio = np.full((len(ALVOS), 2), np.nan)
    horas = soma[:, 4] * BLOCO_SEG / 3600.0
    sens, fa = soma[:, 0] / np.maximum(soma[:, 1], 1), soma[:, 2] / horas
    vpp = 1 - soma[:, 6] / np.maximum(soma[:, 5], 1)
    ok = (soma[:, 3] / soma[:, 4]) <= DUTY_MAX
    if ok.sum() < 2:
        return vazio
    s, f, v = sens[ok], fa[ok], vpp[ok]
    o = np.argsort(f, kind='stable'); s, f, v = s[o], f[o], v[o]
    m = np.concatenate([[True], s[1:] > np.maximum.accumulate(s)[:-1]]); s, f, v = s[m], f[m], v[m]
    if len(s) < 2:
        return vazio
    return np.column_stack([np.interp(ALVOS, f, x, left=np.nan, right=x[-1]) for x in (s, v)])


def por_doente(frame, score, y_col, lim):
    """(n_doentes, T, 7): soma das contagens das sessões de cada doente, a cada limiar."""
    docs = sorted(frame.subject_id.unique()); pos = {s: i for i, s in enumerate(docs)}
    grupos = {}
    for sid, g in frame.groupby('session_id', sort=False):
        grupos.setdefault(pos[g.subject_id.iloc[0]], []).append((g[y_col].fillna(0).values.astype(bool), np.nan_to_num(score[g.index.values], nan=-np.inf)))

    def um(i):
        acc = np.zeros((len(lim), 7))
        for y, sc in grupos[i]:
            ev = runs(y)
            for t, tau in enumerate(lim):
                acc[t] += contagens(y, sc >= tau, ev)
        return i, acc
    out = np.zeros((len(docs), len(lim), 7))
    for i, acc in Parallel(n_jobs=14)(delayed(um)(i) for i in sorted(grupos)):
        out[i] = acc
    return out


def alvo_janela(frame, W, direcao):
    n = int(W * 60 / BLOCO_SEG)
    delta = frame['sbp_target'] - frame.groupby('session_id')['sbp_target'].shift(n)
    y = np.where(delta.isna(), np.nan, ((delta <= -D) if direcao == 'descida' else (delta >= D)).astype(float))
    return delta, y


linhas = []
for W in JANELAS:
    for direcao in ('descida', 'subida'):
        cp = f"{AQUI}/cache/W{W}_{direcao}.json"
        if os.path.exists(cp):
            linhas.append(json.load(open(cp))); print(f"W={W} {direcao}: em cache", flush=True); continue
        t0 = time.time()
        for frame in (dev, hold):
            frame['delta_W'], frame['y_W'] = alvo_janela(frame, W, direcao)
        td, th = dev.delta_W.notna().values, hold.delta_W.notna().values
        dv, hv = dev[td], hold[th]
        X, yb, yr, suj = dv[FEATS].values, dv['y_W'].values, dv['delta_W'].values, dv['subject_id'].values
        folds = list(GroupKFold(n_splits=3).split(X, yb, suj))
        FOLDS = [(X[tr], yr[tr], X[va], va) for tr, va in folds]

        def objective(trial):
            p = dict(n_estimators=trial.suggest_int('n_estimators', 200, 800, step=100), learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                     max_depth=trial.suggest_int('max_depth', 3, 9), min_child_weight=trial.suggest_int('min_child_weight', 1, 12),
                     subsample=trial.suggest_float('subsample', 0.6, 1.0), colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                     gamma=trial.suggest_float('gamma', 1e-3, 5.0, log=True), reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                     reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True))
            oof = np.zeros(len(yb))
            for Xtr, ytr, Xva, va in FOLDS:
                oof[va] = SINAL[direcao] * XGBRegressor(**p, objective='reg:quantileerror', quantile_alpha=TAU[direcao], n_jobs=N_THREADS,
                                                        tree_method='hist').fit(Xtr, ytr).predict(Xva)
            return roc_auc_score(yb, oof)
        study = optuna.create_study(direction='maximize', storage=f"sqlite:///{AQUI}/optuna/optuna_W{W}_{direcao}.db", study_name=f"W{W}_{direcao}", load_if_exists=True)
        feitos = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        if N_TRIALS - feitos > 0:
            study.optimize(objective, n_trials=N_TRIALS - feitos, n_jobs=N_PAR)
        bp = study.best_params
        # previsões fora da amostra no desenvolvimento com os melhores hiperparâmetros (as dobras são as mesmas)
        oof = np.zeros(len(yb))
        for Xtr, ytr, Xva, va in FOLDS:
            oof[va] = SINAL[direcao] * XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=TAU[direcao], n_jobs=N_THREADS * N_PAR,
                                                    tree_method='hist').fit(Xtr, ytr).predict(Xva)
        sc_d = np.full(len(dev), np.nan); sc_d[np.where(td)[0]] = oof
        lim = np.percentile(oof, np.linspace(55, 99.95, 80))
        fr = fronteira(por_doente(dev, sc_d, 'y_W', lim).sum(0))
        # teste: SÓ informativo (modelo treinado em todo o desenvolvimento)
        m = XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=TAU[direcao], n_jobs=N_THREADS * N_PAR, tree_method='hist').fit(X, yr)
        sc_h = np.full(len(hold), np.nan); sc_h[np.where(th)[0]] = SINAL[direcao] * m.predict(hv[FEATS].values)
        frt = fronteira(por_doente(hold, sc_h, 'y_W', lim := np.percentile(sc_h[th], np.linspace(55, 99.95, 80))).sum(0))
        row = dict(W=W, direcao=direcao, prevalencia_dev=float(yb.mean()), blocos_dev=int(td.sum()), eventos_dev=int(yb.sum()), auc_dev=float(study.best_value),
                   **{f'sens_{a}minh': float(v) for a, v in zip(ALVOS, fr[:, 0])}, **{f'vpp_{a}minh': float(v) for a, v in zip(ALVOS, fr[:, 1])},
                   teste_auc=float(roc_auc_score(hv['y_W'].values, sc_h[th])), **{f'teste_sens_{a}minh': float(v) for a, v in zip(ALVOS, frt[:, 0])},
                   best_params=bp, minutos=round((time.time() - t0) / 60, 1))
        json.dump(row, open(cp, 'w'), indent=1, default=float)
        linhas.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != 'best_params'}, flush=True)
        pd.DataFrame(linhas).to_csv(f"{AQUI}/resultados_janela_dev.csv", index=False)
print("FIM")
