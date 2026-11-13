"""ablacao_ecg_tempo.py — o ECG acrescenta algo ao PPG? Calculado só com o desenvolvimento e com a janela de 5 min por tempo exato.

Lado PPG: a tabela da tendência (tendencia_tempo_fixo/dados, 43 características, grade de 15 s). Lado ECG: a tabela resultados_janela_XGB_causal.csv (14_construir_ecg_causal.py: PAT, QTc, QRS, SDI e derivadas por bloco, que se mantêm como estão). O que muda é só o que dependia da posição:
as variações a 2, 5 e 10 min das 3 colunas de declive do ECG (pat_mean_z, qtc_interval_global_z, pat_mean_ewma_longo) e a volatilidade a 5 min de pat_mean_z calculam-se na
grade, por tempo exato (valor em falta se o bloco a t-W não existe), como as do PPG. A etiqueta é a da janela por tempo (PA(t) - PA(t-5 min), evento a 15 mmHg).

Dois cortes, ambos com o modelo adotado (XGBoost quantílico, afinação própria de 20 tentativas por braço e direção, objetivo = AUC fora da amostra, 3 dobras por doente):
 A) INTERSEÇÃO (blocos com as duas cadeias): braço «só PPG» (as 43) contra «PPG + ECG» (as 43 + 14 de ECG + 10 variações/volatilidade de ECG), nas MESMAS linhas.
 B) SISTEMA REALISTA (todas as linhas): o modelo PPG de sempre contra a cascata «ECG onde existe, PPG nos restantes» (as pontuações de modelos diferentes entram na mesma curva
    convertidas em percentil da respetiva distribuição fora da amostra).
Mede-se o AUC fora da amostra no DESENVOLVIMENTO e o ganho emparelhado (IC 95 % por bootstrap de doentes, 1000). O teste (modelos treinados em todo o desenvolvimento) aparece só
em colunas informativas (teste_*), nunca para decidir.     Uso: python3 ablacao_ecg_tempo.py [pasta_dados] [csv_ecg] [n_threads] [trials_em_paralelo]
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
TF = os.path.normpath(os.path.join(AQUI, "..", "tendencia_tempo_fixo"))        # tempo_fixo.py e os best_params_*.json só existem em tendencia_tempo_fixo/
sys.path.insert(0, TF)
import tempo_fixo as tf  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
DADOS = sys.argv[1] if len(sys.argv) > 1 else f"{AQUI}/dados"
CSV_ECG = sys.argv[2] if len(sys.argv) > 2 else f"{AQUI}/resultados_janela_XGB_causal.csv"
N_THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
N_PAR = int(sys.argv[4]) if len(sys.argv) > 4 else 3
D, B, BLOCO_SEG = 15, 1000, 15
N_TRIALS = int(os.environ.get('N_TRIALS', '20'))     # 20 na corrida real (1 só para o teste de fumo)
B = int(os.environ.get('N_BOOT', B))
TAU = {'descida': 0.10, 'subida': 0.90}
SINAL = {'descida': -1.0, 'subida': 1.0}
CHAVE = ['subject_id', 'record_id', 'segment_order', 'bloco_id']
FEAT_ECG = ['pat_velocity', 'pat_trend', 'qtc_velocity', 'pat_mean_diff_adaptativo', 'pat_mean_ewma_longo', 'pat_variance_ratio', 'pat_mean_z', 'pat_lag1',
            'qtc_interval_global_z', 'sdi_interaction_global_z', 'qrs_duration_global_z', 'r_amplitude_global_z', 'qrs_amplitude_global_z', 'qrs_area_global_z']
SLOPE_ECG = ['pat_mean_z', 'qtc_interval_global_z', 'pat_mean_ewma_longo']
os.makedirs(f"{AQUI}/optuna", exist_ok=True)

ecg = pd.read_csv(CSV_ECG, usecols=lambda c: c in set(CHAVE + FEAT_ECG))
ecg['subject_id'] = ecg.subject_id.astype(str)
ecg = ecg.drop_duplicates(CHAVE)


def carregar(nome):
    g, V = tf.preparar_grade(DADOS, nome)
    g['subject_id'] = g.subject_id.astype(str)
    mapa = pd.read_csv(f"{DADOS}/{nome}_ppg.csv", usecols=['session_id', 'bloco_id', 'record_id', 'segment_order'])
    g = g.merge(mapa, on=['session_id', 'bloco_id'], how='left').merge(ecg, on=CHAVE, how='left')
    g = g.sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
    gb = g.groupby('session_id')
    novas = []
    for c in SLOPE_ECG:
        for m in (2, 5, 10):
            g[f'd_{c}_{m}m'] = g[c] - gb[c].shift(int(m * 60 / BLOCO_SEG)); novas.append(f'd_{c}_{m}m')
    g['vol_pat_mean_z_5m'] = gb['pat_mean_z'].transform(lambda s: s.rolling(20, min_periods=20).std()); novas.append('vol_pat_mean_z_5m')
    feats_ecg = FEAT_ECG + novas
    g['tem_ecg'] = g[FEAT_ECG].notna().all(axis=1)
    return g, V['tudo'], feats_ecg


dev, FEATS, FEATS_ECG = carregar('dev')
hold, _, _ = carregar('holdout')
print(f"dev {len(dev)} linhas na grade, com alvo {int(dev.delta.notna().sum())}, com ECG {int((dev.delta.notna() & dev.tem_ecg).sum())}; "
      f"teste com alvo {int(hold.delta.notna().sum())}, com ECG {int((hold.delta.notna() & hold.tem_ecg).sum())}", flush=True)


def xgb(bp, direcao, nj):
    return XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=TAU[direcao], n_jobs=nj, tree_method='hist')


def afinar(nome, X, yb, yr, suj, direcao):
    FOLDS = [(X[tr], yr[tr], X[va], va) for tr, va in GroupKFold(n_splits=3).split(X, yb, suj)]

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
    s = optuna.create_study(direction='maximize', storage=f"sqlite:///{AQUI}/optuna/optuna_ecg_{nome}_{direcao}.db", study_name=f"ecg_{nome}_{direcao}", load_if_exists=True)
    feitos = sum(t.state == optuna.trial.TrialState.COMPLETE for t in s.trials)
    if N_TRIALS - feitos > 0:
        s.optimize(objective, n_trials=N_TRIALS - feitos, n_jobs=N_PAR)
    return s.best_params, s.best_value, [(tr, va) for tr, va in GroupKFold(n_splits=3).split(X, yb, suj)]


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


def percentil(ref, x):
    return np.searchsorted(np.sort(ref), x, side='right') / len(ref)


linhas = []
for direcao in ('descida', 'subida'):
    t0 = time.time()
    for fr in (dev, hold):
        delta = fr['delta']
        fr['y'] = np.where(delta.isna(), np.nan, ((delta <= -D) if direcao == 'descida' else (delta >= D)).astype(float))
    ti, th = (dev.delta.notna() & dev.tem_ecg).values, (hold.delta.notna() & hold.tem_ecg).values
    di, hi = dev[ti], hold[th]
    yb, yr, suj = di['y'].values, di['delta'].values, di['subject_id'].values
    XA, XB = di[FEATS].values, di[FEATS + FEATS_ECG].values
    bpA, cvA, folds = afinar('so_PPG', XA, yb, yr, suj, direcao)
    bpB, cvB, _ = afinar('PPG_ECG', XB, yb, yr, suj, direcao)
    oA, oB = oof_com(bpA, direcao, XA, yr, folds), oof_com(bpB, direcao, XB, yr, folds)
    rng = np.random.default_rng(0)
    ic = boot_dif(yb, oB, oA, suj, rng)
    mA = xgb(bpA, direcao, N_THREADS * N_PAR).fit(XA, yr); mB = xgb(bpB, direcao, N_THREADS * N_PAR).fit(XB, yr)
    sA, sB = SINAL[direcao] * mA.predict(hi[FEATS].values), SINAL[direcao] * mB.predict(hi[FEATS + FEATS_ECG].values)
    yh = hi['y'].values; ich = boot_dif(yh, sB, sA, hi['subject_id'].values, rng)
    linhas.append(dict(corte='A interseção (blocos com ECG)', direcao=direcao, n_dev=int(ti.sum()), sujeitos_dev=int(di.subject_id.nunique()), eventos_dev=int(yb.sum()),
                       auc_dev_so_ppg=roc_auc_score(yb, oA), auc_dev_ppg_ecg=roc_auc_score(yb, oB), ganho_dev=roc_auc_score(yb, oB) - roc_auc_score(yb, oA),
                       ic_baixo=ic[0], ic_alto=ic[1], n_teste=int(th.sum()), teste_auc_so_ppg=roc_auc_score(yh, sA), teste_auc_ppg_ecg=roc_auc_score(yh, sB),
                       teste_ganho=roc_auc_score(yh, sB) - roc_auc_score(yh, sA), teste_ic_baixo=ich[0], teste_ic_alto=ich[1], minutos=round((time.time() - t0) / 60, 1)))
    print(linhas[-1], flush=True)

    # B) sistema realista: modelo PPG em todas as linhas contra a cascata «ECG onde existe»
    tf_, th_ = dev.delta.notna().values, hold.delta.notna().values
    df_, hf_ = dev[tf_], hold[th_]
    Xf, yrf, ybf, sf = df_[FEATS].values, df_['delta'].values, df_['y'].values, df_['subject_id'].values
    bpF = json.load(open(f"{TF}/best_params_quant_{direcao}_tudo.json"))['best_params']
    foldsF = list(GroupKFold(n_splits=3).split(Xf, ybf, sf))
    oF = oof_com(bpF, direcao, Xf, yrf, foldsF)
    com_ecg = df_['tem_ecg'].values
    pos_int = np.where(ti[tf_])[0]                                    # linhas da interseção dentro das linhas com alvo
    casc_pct = percentil(oF, oF)                                      # percentil de cada linha no modelo PPG completo
    casc_pct[pos_int] = percentil(oB, oB)                             # nas da interseção: percentil no modelo PPG+ECG (fora da amostra)
    rng = np.random.default_rng(1)
    ic2 = boot_dif(ybf, casc_pct, percentil(oF, oF), sf, rng)
    mF = xgb(bpF, direcao, N_THREADS * N_PAR).fit(Xf, yrf)
    shF = SINAL[direcao] * mF.predict(hf_[FEATS].values)
    shB = np.full(len(hf_), np.nan); posh = np.where(th[th_])[0]
    shB[posh] = SINAL[direcao] * mB.predict(hf_.iloc[posh][FEATS + FEATS_ECG].values)
    ch = percentil(oF, shF); ch[posh] = percentil(oB, shB[posh])
    yhf = hf_['y'].values; ic3 = boot_dif(yhf, ch, percentil(oF, shF), hf_['subject_id'].values, rng)
    linhas.append(dict(corte='B sistema realista (todas as linhas)', direcao=direcao, n_dev=int(tf_.sum()), sujeitos_dev=int(df_.subject_id.nunique()), eventos_dev=int(ybf.sum()),
                       frac_com_ecg=float(com_ecg.mean()), auc_dev_so_ppg=roc_auc_score(ybf, oF), auc_dev_ppg_ecg=roc_auc_score(ybf, casc_pct),
                       ganho_dev=roc_auc_score(ybf, casc_pct) - roc_auc_score(ybf, oF), ic_baixo=ic2[0], ic_alto=ic2[1], n_teste=int(th_.sum()),
                       teste_auc_so_ppg=roc_auc_score(yhf, shF), teste_auc_ppg_ecg=roc_auc_score(yhf, ch), teste_ganho=roc_auc_score(yhf, ch) - roc_auc_score(yhf, shF),
                       teste_ic_baixo=ic3[0], teste_ic_alto=ic3[1], minutos=round((time.time() - t0) / 60, 1)))
    print(linhas[-1], flush=True)
    pd.DataFrame(linhas).to_csv(f"{AQUI}/resultados_ablacao_ecg_tempo.csv", index=False)
print("FIM")
