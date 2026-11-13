"""ablacao_ecg_todas.py — o ECG (com as suas variações) acrescenta algo ao PPG na tendência? Só XGBoost, só desenvolvimento para decidir.

Diferenças face a ablacao_ecg_tempo (que não se altera): lá o ECG vinha da tabela ancorada no PAT (só 18 % dos blocos, porque o PAT só é plausível, entre 0,15 e 0,60 s, em ~18 % das
janelas do MIMIC, por desvios de relógio do PLETH). Aqui o ECG vem das janelas de /media/filipe/Dados/Features/features_*_baseline.csv (a mesma extração, com as mesmas chaves
subject_id/segment_name/window_idx das janelas do PPG), SEM exigir PAT: o QRS, o R, o QT, o TQ, o QTc e o SDI existem em ~95 % dos blocos. Só o PAT fica NaN onde é implausível.

Características do ECG (como as do PPG: níveis, variações a 2/5/10 min por tempo exato na grade de 15 s, volatilidade a 5 min):
  níveis (z-score causal pelos 10 primeiros blocos da sessão, como as do PPG): qrs_duration, r_amplitude, qrs_amplitude, qrs_area, qt_interval, tq_interval, qtc_interval,
  sdi_interaction e pat_mean (PAT só onde plausível)  → 9
  variações a 2, 5 e 10 min de cada uma  → 27;   volatilidade a 5 min de cada uma  → 9;   total 45.
Braço A «só PPG» = as 43 do modelo adotado (hiperparâmetros de best_params_quant_<direção>_tudo.json, sem voltar a afinar; o AUC fora da amostra tem de dar 0,8128/0,8329).
Braço B «PPG + ECG» = as 43 + as 45, com afinação própria (20 tentativas, objetivo = AUC fora da amostra, 3 dobras por doente, as mesmas dobras do braço A), nas MESMAS linhas.
Ganho emparelhado com IC 95 % por bootstrap de doentes (1000). Cortes informativos, sobre os mesmos resultados: só os blocos com PAT válido, e só os blocos sem PAT válido.
O teste (modelos treinados em todo o desenvolvimento) aparece só nas colunas teste_*, nunca para decidir.
Uso: python3 ablacao_ecg_todas.py [n_threads] [trials_em_paralelo]      (SMOKE=1 usa 8 % dos doentes e 1 tentativa)
"""
import json
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

AQUI = os.path.dirname(os.path.abspath(__file__))
A = "/home/filipe/Desktop/Tese/algoritmo-ml"
TF = f"{A}/Tentativa_22_10/tendencia_tempo_fixo"
sys.path.insert(0, TF)
sys.path.insert(0, f"{A}/03_ABSOLUTO_codigo_e_resultados/CODIGO")
import tempo_fixo as tf  # noqa: E402
from suavizacao_e_normalizacao_causal import normalizar_causal_calibracao  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
N_THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N_PAR = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SMOKE = os.environ.get("SMOKE") == "1"
N_TRIALS = 1 if SMOKE else int(os.environ.get("N_TRIALS", "20"))
B = 50 if SMOKE else 1000
D, BLOCO_SEG = 15, 15
TAU = {'descida': 0.10, 'subida': 0.90}
SINAL = {'descida': -1.0, 'subida': 1.0}
ECG_BRUTO = ['qrs_duration', 'r_amplitude', 'qrs_amplitude', 'qrs_area', 'qt_interval', 'tq_interval', 'qtc_interval', 'sdi_interaction', 'pat_mean']
PAT_MIN, PAT_MAX = 0.15, 0.60
os.makedirs(f"{AQUI}/optuna", exist_ok=True)

# ── ECG por bloco de 15 s (média das janelas de 5 s do bloco, como o PPG), sem exigir PAT ────────────────────────────────────────────────────────────
partes = []
for g in ("homens", "mulheres"):
    d = pd.read_csv(f"/media/filipe/Dados/Features/features_{g}_baseline.csv",
                    usecols=['subject_id', 'segment_name', 'segment_order', 'window_idx'] + ECG_BRUTO)
    d.loc[~d.pat_mean.between(PAT_MIN, PAT_MAX), 'pat_mean'] = np.nan          # só o PAT sai, a janela fica
    d['record_id'] = d.segment_name.astype(str).apply(lambda x: x.rsplit('_', 1)[0])
    d['bloco_id'] = d.window_idx // 3
    d['subject_id'] = d.subject_id.astype(str)
    partes.append(d.groupby(['subject_id', 'record_id', 'segment_order', 'bloco_id'])[ECG_BRUTO].mean().reset_index())
ECG = pd.concat(partes, ignore_index=True)
print(f"ECG: {len(ECG)} blocos em {ECG.subject_id.nunique()} sujeitos", flush=True)


def carregar(nome):
    g, V = tf.preparar_grade(TF + "/dados", nome)                                # grade de 15 s, com delta e as 43 características
    tab = pd.read_csv(f"{TF}/dados/{nome}_ppg.csv", usecols=['subject_id', 'session_id', 'bloco_id', 'record_id', 'segment_order'])
    tab['subject_id'] = tab.subject_id.astype(str); tab['record_id'] = tab.record_id.astype(str)
    tab = tab.sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
    tab = tab.merge(ECG, on=['subject_id', 'record_id', 'segment_order', 'bloco_id'], how='left')
    tab = normalizar_causal_calibracao(tab, ECG_BRUTO, n_calib=10, group_col='session_id', sufixo='_z')
    zc = [c + '_z' for c in ECG_BRUTO]
    g = g.merge(tab[['session_id', 'bloco_id'] + zc], on=['session_id', 'bloco_id'], how='left')
    g = g.sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
    gb = g.groupby('session_id')
    novas = list(zc)
    for c in zc:
        for m in (2, 5, 10):
            g[f'd_{c}_{m}m'] = g[c] - gb[c].shift(int(m * 60 / BLOCO_SEG)); novas.append(f'd_{c}_{m}m')
    W5 = int(5 * 60 / BLOCO_SEG)
    for c in zc:
        g[f'vol_{c}_5m'] = gb[c].transform(lambda s: s.rolling(W5, min_periods=W5).std()); novas.append(f'vol_{c}_5m')
    g['pat_valido'] = g['pat_mean_z'].notna()
    g['tem_ecg'] = g[[c for c in zc if c != 'pat_mean_z']].notna().any(axis=1)
    g = g.replace([np.inf, -np.inf], np.nan)
    return g, V['tudo'], novas


dev, FEATS, FEATS_ECG = carregar('dev')
hold, _, _ = carregar('holdout')
if SMOKE:
    suj = dev.subject_id.drop_duplicates().sample(frac=0.08, random_state=0)
    dev = dev[dev.subject_id.isin(suj)].reset_index(drop=True)
    hold = hold[hold.subject_id.isin(hold.subject_id.drop_duplicates().sample(frac=0.2, random_state=0))].reset_index(drop=True)
td, th = dev.delta.notna(), hold.delta.notna()
print(f"dev {int(td.sum())} blocos com alvo, com ECG {int((td & dev.tem_ecg).sum())} ({(td & dev.tem_ecg).sum() / td.sum():.1%}), com PAT válido {int((td & dev.pat_valido).sum())} ({(td & dev.pat_valido).sum() / td.sum():.1%}); "
      f"teste {int(th.sum())}, com ECG {int((th & hold.tem_ecg).sum())}; características do ECG: {len(FEATS_ECG)}", flush=True)


def xgb(bp, direcao, nj):
    return XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=TAU[direcao], n_jobs=nj, tree_method='hist')


def afinar(X, yb, yr, suj, direcao):
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
            oof[va] = SINAL[direcao] * xgb(p, direcao, N_THREADS).fit(Xtr, ytr).predict(Xva)
        return roc_auc_score(yb, oof)
    nome = "smoke_" if SMOKE else ""
    s = optuna.create_study(direction='maximize', storage=f"sqlite:///{AQUI}/optuna/{nome}ecg_todas_{direcao}.db", study_name=f"ecg_todas_{direcao}", load_if_exists=True)
    feitos = sum(t.state == optuna.trial.TrialState.COMPLETE for t in s.trials)
    if N_TRIALS - feitos > 0:
        s.optimize(objective, n_trials=N_TRIALS - feitos, n_jobs=N_PAR)
    return s.best_params, s.best_value, folds


def oof_com(bp, direcao, X, yr, folds):
    o = np.zeros(len(yr))
    for tr, va in folds:
        o[va] = SINAL[direcao] * xgb(bp, direcao, N_THREADS * N_PAR).fit(X[tr], yr[tr]).predict(X[va])
    return o


def boot_dif(y, a, b, suj, rng):
    us = np.unique(suj); idx = {u: np.where(suj == u)[0] for u in us}; d = []
    for _ in range(B):
        ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        if y[ii].min() != y[ii].max():
            d.append(roc_auc_score(y[ii], a[ii]) - roc_auc_score(y[ii], b[ii]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def corte(nome, mask, y, a, b, suj, rng):
    if mask.sum() < 1000 or y[mask].min() == y[mask].max():
        return {}
    ic = boot_dif(y[mask], b[mask], a[mask], suj[mask], rng)
    return {f'{nome}_n': int(mask.sum()), f'{nome}_auc_so_ppg': roc_auc_score(y[mask], a[mask]), f'{nome}_auc_ppg_ecg': roc_auc_score(y[mask], b[mask]),
            f'{nome}_ganho': roc_auc_score(y[mask], b[mask]) - roc_auc_score(y[mask], a[mask]), f'{nome}_ic_baixo': ic[0], f'{nome}_ic_alto': ic[1]}


linhas = []
sfx = "_smoke" if SMOKE else ""
for direcao in ('descida', 'subida'):
    t0 = time.time()
    for fr in (dev, hold):
        delta = fr['delta']
        fr['y'] = np.where(delta.isna(), np.nan, ((delta <= -D) if direcao == 'descida' else (delta >= D)).astype(float))
    di, hi = dev[td.values], hold[th.values]
    yb, yr, suj = di['y'].values, di['delta'].values, di['subject_id'].values
    XA = di[FEATS].values.astype(np.float32); XB = di[FEATS + FEATS_ECG].values.astype(np.float32)
    bpA = json.load(open(f"{TF}/best_params_quant_{direcao}_tudo.json"))['best_params']
    bpB, cvB, folds = afinar(XB, yb, yr, suj, direcao)
    oA, oB = oof_com(bpA, direcao, XA, yr, folds), oof_com(bpB, direcao, XB, yr, folds)
    print(f"{direcao}: AUC dev braço A {roc_auc_score(yb, oA):.4f} (adotado: {'0.8128' if direcao == 'descida' else '0.8329'} em todas as linhas) | braço B {roc_auc_score(yb, oB):.4f} | afinação B {cvB:.4f}", flush=True)
    rng = np.random.default_rng(0)
    lin = dict(direcao=direcao, n_dev=len(yb), sujeitos_dev=int(pd.Series(suj).nunique()), eventos_dev=int(yb.sum()), n_caracteristicas_ecg=len(FEATS_ECG), afinacao_B_cv=cvB)
    lin.update(corte('todas', np.ones(len(yb), bool), yb, oA, oB, suj, rng))
    lin.update(corte('com_ecg', di['tem_ecg'].values, yb, oA, oB, suj, rng))
    lin.update(corte('pat_valido', di['pat_valido'].values, yb, oA, oB, suj, rng))
    lin.update(corte('sem_pat_valido', ~di['pat_valido'].values, yb, oA, oB, suj, rng))
    mA, mB = xgb(bpA, direcao, N_THREADS * N_PAR).fit(XA, yr), xgb(bpB, direcao, N_THREADS * N_PAR).fit(XB, yr)
    sA = SINAL[direcao] * mA.predict(hi[FEATS].values.astype(np.float32)); sB = SINAL[direcao] * mB.predict(hi[FEATS + FEATS_ECG].values.astype(np.float32))
    yh = hi['y'].values; ich = boot_dif(yh, sB, sA, hi['subject_id'].values, rng)
    lin.update(dict(n_teste=len(yh), teste_auc_so_ppg=roc_auc_score(yh, sA), teste_auc_ppg_ecg=roc_auc_score(yh, sB), teste_ganho=roc_auc_score(yh, sB) - roc_auc_score(yh, sA),
                    teste_ic_baixo=ich[0], teste_ic_alto=ich[1], minutos=round((time.time() - t0) / 60, 1)))
    # importância das características do ECG no braço B (ganho médio), para o texto
    imp = pd.Series(mB.feature_importances_, index=FEATS + FEATS_ECG).sort_values(ascending=False)
    imp.to_csv(f"{AQUI}/importancia_B_{direcao}{sfx}.csv", header=['importancia'])
    json.dump({k: float(v) if hasattr(v, 'item') or isinstance(v, float) else v for k, v in bpB.items()}, open(f"{AQUI}/best_params_B_{direcao}{sfx}.json", "w"), indent=1)
    linhas.append(lin); print(lin, flush=True)
    pd.DataFrame(linhas).to_csv(f"{AQUI}/resultados_ablacao_ecg_todas{sfx}.csv", index=False)
print("FIM")
