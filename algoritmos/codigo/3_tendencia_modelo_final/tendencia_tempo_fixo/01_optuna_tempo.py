"""01_optuna_tempo.py — afinação do XGBoost quantílico da tendência com a janela de 5 min ANCORADA NO TEMPO.

Mesmo
espaço de pesquisa, mesmo objetivo (AUC de deteção sobre previsões fora da amostra, GroupKFold de 3 por sujeito, folds fixos), 20 tentativas
por direção, regressão quantílica (q=0,10 na descida e 0,90 na subida). Muda só os dados: tempo_fixo.preparar_grade (alvo e variações por
tempo exato; características em falta ficam NaN) e o conjunto de características, escolhido com CONJUNTO (base, apg, tudo).

Saídas (pasta de trabalho): optuna/optuna_quant_<direção>_<conjunto>.db (estudo retomável), best_params_quant_<direção>_<conjunto>.json.
Uso:  CONJUNTO=tudo python3 01_optuna_tempo.py [pasta_dados] [n_threads_por_fit] [n_trials_em_paralelo]
"""
import json
import os
import sys

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import tempo_fixo as tf  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
DADOS = sys.argv[1] if len(sys.argv) > 1 else f"{AQUI}/dados"
N_THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
N_PARALELO = int(sys.argv[3]) if len(sys.argv) > 3 else 3
CONJUNTO = os.environ.get("CONJUNTO", "tudo")
N_TRIALS = 20

dev_g, V = tf.preparar_grade(DADOS, 'dev')
hold_g, _ = tf.preparar_grade(DADOS, 'holdout')
feats = V[CONJUNTO]
dev, hold = dev_g[dev_g.delta.notna()].reset_index(drop=True), hold_g[hold_g.delta.notna()].reset_index(drop=True)
del dev_g, hold_g

for direcao in ('descida', 'subida'):
    q, sinal = (0.10, -1.0) if direcao == 'descida' else (0.90, 1.0)
    X, yb, yr, suj = dev[feats].values, dev[f'y_{direcao}'].values, dev['delta'].values, dev['subject_id'].values
    FOLDS = [(X[tr], yr[tr], X[va], va) for tr, va in GroupKFold(n_splits=3).split(X, yb, suj)]
    print(f"\n{'=' * 70}\n{CONJUNTO} / {direcao}: {len(feats)} características, dev {len(dev)} blocos, {int(yb.sum())} eventos ({100 * yb.mean():.2f} %)", flush=True)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int('n_estimators', 200, 800, step=100),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            max_depth=trial.suggest_int('max_depth', 3, 9),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 12),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
            gamma=trial.suggest_float('gamma', 1e-3, 5.0, log=True),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True))
        oof = np.zeros(len(yb))
        for Xtr, ytr, Xva, va in FOLDS:
            m = XGBRegressor(**params, objective='reg:quantileerror', quantile_alpha=q, n_jobs=N_THREADS, tree_method='hist')
            oof[va] = sinal * m.fit(Xtr, ytr).predict(Xva)
        return roc_auc_score(yb, oof)

    os.makedirs(f"{AQUI}/optuna", exist_ok=True)
    study = optuna.create_study(direction="maximize", storage=f"sqlite:///{AQUI}/optuna/optuna_quant_{direcao}_{CONJUNTO}.db",
                                study_name=f"quant_{direcao}_{CONJUNTO}_tempo", load_if_exists=True)
    feitos = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if N_TRIALS - feitos > 0:
        print(f"tentativas completas: {feitos} | a correr mais {N_TRIALS - feitos}", flush=True)
        study.optimize(objective, n_trials=N_TRIALS - feitos, n_jobs=N_PARALELO)
    bp = study.best_params
    final = XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=q, n_jobs=N_THREADS * N_PARALELO, tree_method='hist').fit(X, yr)
    auc_h = roc_auc_score(hold[f'y_{direcao}'].values, sinal * final.predict(hold[feats].values))
    print(f"{CONJUNTO}/{direcao}: AUC dev (OOF, melhor tentativa) {study.best_value:.4f} | teste {auc_h:.4f}", flush=True)
    json.dump(dict(modo='quant', direcao=direcao, conjunto=CONJUNTO, janela='tempo', best_cv=study.best_value, holdout=float(auc_h),
                   n_features=len(feats), feats=feats, best_params=bp),
              open(f"{AQUI}/best_params_quant_{direcao}_{CONJUNTO}.json", "w"), indent=2)
print("FIM")
