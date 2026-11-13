"""Janela de 5 min ancorada no tempo e as características novas. Dados e sequências de hibrido_quantilico_tempo.preparar (CONJUNTO=tudo: 43 tabulares e
11 canais por instante). Acrescenta o XGBoost classificador tabular como referência emparelhada (mesmas linhas) e guarda as previsões do teste.

Hibridos rede temporal + XGBoost na formulacao de TENDENCIA (causal, so-PPG),
com busca CONJUNTA de hiperparametros (rede e XGBoost do topo afinados juntos).

Em comparacao_arquiteturas_classificador.py a rede era afinada como
classificador puro e o XGBoost do topo tinha hiperparametros fixos
(XGB_HIBRIDO_PARAMS). Aqui, em cada trial e em cada dobra do GroupKFold(3) por
sujeito: treina-se a rede na dobra de treino (early stopping na de validacao,
como em treinar_seq_trial), extrai-se a camada 'emb', ajusta-se o XGBoost sobre
[tabulares + embedding] e mede-se o AUC da dobra de validacao. O objetivo e a
media das 3 dobras.

Espacos de busca: os da rede pura e do XGBoost tabular de comparacao_arquiteturas_classificador.py,
juntos (prefixos rede_/xgb_). 20 trials, o orcamento da formulacao de tendencia.

Avaliacao final: igual a avaliar_final_seq, com o XGBoost do topo afinado.
Correr via correr_hibrido_resiliente.sh (reinicio automatico por memoria).

Saida: resumo_hibrido_conjunto.csv, cache_hibrido_conjunto/*.json,
       optuna_arquiteturas/optuna_HIB_{rede}_{direcao}_conjunta.db
"""
import gc
import json
import os
import time

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from xgboost import XGBClassifier

from base_hib import (
    N_FOLDS, N_TRIALS, OUT, PASTA_OPTUNA, T, _sair_se_memoria_alta,
    construir_bilstm, construir_bitcn,
)
from hibrido_quantilico_tempo import preparar

CACHE_DIR = f"{OUT}/cache_hibrido_classificador"
RESUMO = f"{OUT}/resumo_hibrido_classificador.csv"
PREDS = f"{OUT}/holdout_preds_hibrido_classificador_{{direcao}}.csv.gz"
THREADS_XGB = [-1]      # 3 tentativas em paralelo com 5 threads na fase só-XGBoost; com as redes uma de cada vez e todos os núcleos


def params_de_trial(trial, nome):
    xgb = dict(
        n_estimators=trial.suggest_int('xgb_n_estimators', 200, 800, step=100),
        learning_rate=trial.suggest_float('xgb_learning_rate', 0.01, 0.2, log=True),
        max_depth=trial.suggest_int('xgb_max_depth', 3, 9),
        min_child_weight=trial.suggest_int('xgb_min_child_weight', 1, 12),
        subsample=trial.suggest_float('xgb_subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
        reg_alpha=trial.suggest_float('xgb_reg_alpha', 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float('xgb_reg_lambda', 1e-3, 10.0, log=True),
    )
    if nome == 'BiLSTM':
        net = dict(
            learning_rate=trial.suggest_float('rede_learning_rate', 1e-4, 5e-3, log=True),
            lstm_units=trial.suggest_int('rede_lstm_units', 16, 128, step=16),
            dropout_rate=trial.suggest_float('rede_dropout_rate', 0.0, 0.4),
        )
    else:
        net = dict(
            learning_rate=trial.suggest_float('rede_learning_rate', 1e-4, 3e-3, log=True),
            nb_filters=trial.suggest_int('rede_nb_filters', 16, 64, step=16),
            kernel_size=trial.suggest_int('rede_kernel_size', 3, 7),
            dropout_rate=trial.suggest_float('rede_dropout_rate', 0.1, 0.4),
        )
    return net, xgb


def separar_params(best):
    return ({k[5:]: v for k, v in best.items() if k.startswith('rede_')},
            {k[4:]: v for k, v in best.items() if k.startswith('xgb_')})


def treinar_rede(nome, net, Xs_tr, y_tr, Xs_va, y_va, F):
    tf.keras.backend.clear_session()
    pw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    modelo = construir_bilstm(net, T, F) if nome == 'BiLSTM' else construir_bitcn(net, T, F)
    modelo.compile(optimizer=tf.keras.optimizers.Adam(net['learning_rate']),
                   loss='binary_crossentropy', metrics=['AUC'])
    modelo.fit(Xs_tr, y_tr, validation_data=(Xs_va, y_va), epochs=30, batch_size=512,
               class_weight={0: 1.0, 1: float(pw)}, verbose=0,
               callbacks=[EarlyStopping(monitor='val_AUC', mode='max', patience=5,
                                         restore_best_weights=True)])
    return modelo


def xgb_topo(xgb, y):
    pw = (len(y) - y.sum()) / max(y.sum(), 1)
    return XGBClassifier(**xgb, scale_pos_weight=pw, eval_metric='auc', n_jobs=THREADS_XGB[0], tree_method='hist')


def objetivo_cv(nome, net, xgb, dados):
    Xs, Xt, y, subj = dados['Xs_d_n'], dados['Xt_d'], dados['y_d'], dados['subj_d']
    aucs = []
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(Xs, y, subj):
        modelo = treinar_rede(nome, net, Xs[tr], y[tr], Xs[va], y[va], dados['F'])
        extr = Model(modelo.input, modelo.get_layer('emb').output)
        e_tr = extr.predict(Xs[tr], batch_size=1024, verbose=0)
        e_va = extr.predict(Xs[va], batch_size=1024, verbose=0)
        m = xgb_topo(xgb, y[tr])
        m.fit(np.hstack([Xt[tr], e_tr]), y[tr])
        aucs.append(roc_auc_score(y[va], m.predict_proba(np.hstack([Xt[va], e_va]))[:, 1]))
        del modelo, extr, m, e_tr, e_va
        gc.collect()
    return float(np.mean(aucs))


def objetivo_cv_xgb(xgb, dados):
    Xt, y, subj = dados['Xt_d'], dados['y_d'], dados['subj_d']
    aucs = []
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(Xt, y, subj):
        m = xgb_topo(xgb, y[tr]).fit(Xt[tr], y[tr])
        aucs.append(roc_auc_score(y[va], m.predict_proba(Xt[va])[:, 1]))
    return float(np.mean(aucs))


def otimizar(nome, direcao, dados):
    nf = f"HIBC_{nome}_{direcao}"
    study = optuna.create_study(direction="maximize",
                                storage=f"sqlite:///{PASTA_OPTUNA}/optuna_{nf}.db",
                                study_name=nf, load_if_exists=True)

    def objective(trial):
        net, xgb = params_de_trial(trial, 'BiLSTM' if nome == 'XGB' else nome)
        return objetivo_cv_xgb(xgb, dados) if nome == 'XGB' else objetivo_cv(nome, net, xgb, dados)

    feitos = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if N_TRIALS - feitos > 0:
        print(f"[HIBC {nome}/{direcao}] trials completos: {feitos} | a correr mais {N_TRIALS - feitos}", flush=True)
        THREADS_XGB[0] = 5 if nome == 'XGB' else -1
        study.optimize(objective, n_trials=N_TRIALS - feitos, n_jobs=3 if nome == 'XGB' else 1, show_progress_bar=False,
                       callbacks=[_sair_se_memoria_alta])
    print(f"[HIBC {nome}/{direcao}] melhor AUC CV = {study.best_value:.4f} | params: {study.best_params}", flush=True)
    return study.best_params, study.best_value


def avaliar_final(nome, best, dados):
    """(AUC no teste, scores do teste). XGB = só tabular; nas redes, XGBoost sobre [tabular + embedding]."""
    THREADS_XGB[0] = -1
    net, xgb = separar_params(best)
    y_d = dados['y_d']
    if nome == 'XGB':
        m = xgb_topo(xgb, y_d).fit(dados['Xt_d'], y_d)
        s = m.predict_proba(dados['Xt_h'])[:, 1]
    else:
        modelo = treinar_rede(nome, net, dados['Xs_d_n'][dados['tr']], y_d[dados['tr']],
                              dados['Xs_d_n'][dados['va']], y_d[dados['va']], dados['F'])
        extr = Model(modelo.input, modelo.get_layer('emb').output)
        e_d = extr.predict(dados['Xs_d_n'], batch_size=1024, verbose=0)
        e_h = extr.predict(dados['Xs_h_n'], batch_size=1024, verbose=0)
        m = xgb_topo(xgb, y_d).fit(np.hstack([dados['Xt_d'], e_d]), y_d)
        s = m.predict_proba(np.hstack([dados['Xt_h'], e_h]))[:, 1]
    return float(roc_auc_score(dados['y_h'], s)), s


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    linhas = []
    for direcao in ('descida', 'subida'):
        dados = None
        p_preds = PREDS.format(direcao=direcao)
        preds = pd.read_csv(p_preds) if os.path.exists(p_preds) else None
        auc_xgb = None
        for nome in ('XGB', 'BiLSTM', 'BiTCN'):
            cp = f"{CACHE_DIR}/{nome}_{direcao}.json"
            if os.path.exists(cp):
                c = json.load(open(cp))
                print(f"[HIBC {nome}/{direcao}] em cache: HOLDOUT AUC = {c['auc_holdout']:.4f} (a saltar)", flush=True)
            else:
                if dados is None:
                    dados = preparar(direcao)
                    print(f"[{direcao}] dev {dados['Xs_d_n'].shape} | holdout {dados['Xs_h_n'].shape} | eventos dev {dados['y_d'].mean()*100:.2f}% "
                          f"| holdout {dados['y_h'].mean()*100:.2f}%", flush=True)
                if preds is None:
                    preds = pd.DataFrame(dict(subject_id=dados['subj_h'], session_id=dados['sess_h'], bloco_id=dados['blo_h'], y=dados['y_h'], delta=dados['dl_h']))
                t0 = time.time()
                best, best_cv = otimizar(nome, direcao, dados)
                auc, sc = avaliar_final(nome, best, dados)
                preds[f'score_{nome}'] = sc
                preds.to_csv(p_preds, index=False, compression='gzip')
                c = dict(auc_holdout=auc, auc_cv=best_cv, best_params=best)
                json.dump(c, open(cp, "w"), indent=2)
                print(f"[HIBC {nome}/{direcao}] HOLDOUT AUC = {auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if nome == 'XGB':
                auc_xgb = c['auc_holdout']
            linhas.append(dict(direcao=direcao, modelo='XGB_classificador' if nome == 'XGB' else f"HIBRIDO_{nome}_XGB_classificador",
                               auc_cv=round(c['auc_cv'], 4), auc_holdout=round(c['auc_holdout'], 4),
                               ganho_vs_xgb=None if nome == 'XGB' else round(c['auc_holdout'] - auc_xgb, 4), best_params=json.dumps(c['best_params'])))
            pd.DataFrame(linhas).to_csv(RESUMO, index=False)
        del dados
        gc.collect()
    print(f"\n{'='*74}\nRESULTADO FINAL (classificador, busca conjunta, {N_TRIALS} trials x {N_FOLDS}-fold)\n{'='*74}")
    print(pd.DataFrame(linhas)[['direcao', 'modelo', 'auc_cv', 'auc_holdout', 'ganho_vs_xgb']].to_string(index=False))


if __name__ == "__main__":
    main()
