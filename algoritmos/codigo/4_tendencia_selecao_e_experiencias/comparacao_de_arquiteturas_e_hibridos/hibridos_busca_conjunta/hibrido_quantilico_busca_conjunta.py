"""
Hibridos rede temporal + XGBoost na formulacao de TENDENCIA com REGRESSAO
QUANTILICA sobre o Delta (causal, so-PPG), busca CONJUNTA de hiperparametros,
e o XGBoost quantilico tabular como referencia, tudo sobre as MESMAS linhas.

PORQUE EXISTE
  O regressor quantilico adotado so foi ajustado com XGBoost, e a comparacao
  de arquiteturas (comparacao_arquiteturas_classificador.py e
  hibrido_classificador_busca_conjunta.py) usou o classificador binario. Este script
  responde se as redes acrescentam alguma coisa tambem na formulacao
  quantilica.

PORQUE REAJUSTA TAMBEM O XGBOOST QUANTILICO
  O XGBoost quantilico adotado foi ajustado sobre dev_ppg.csv e com outro
  filtro de linhas. As redes precisam de
  sequencias de T=20 blocos, que so existem a partir do 20.o bloco de cada
  sessao, por isso a comparacao emparelhada exige reajustar o XGBoost nas
  mesmas linhas. O numero tabular deste script e a referencia emparelhada,
  nao substitui o da tese.

PROTOCOLO (o de hibrido_classificador_busca_conjunta.py, trocando a tarefa)
  - dados, sequencias (T=20) e as 23 caracteristicas tabulares de
    comparacao_arquiteturas_classificador.py, alvo Delta = PA(t) - PA(t-5 min),
    sem as linhas em que o Delta nao esta definido
  - rede com a mesma arquitetura e a mesma camada 'emb', saida linear e perda
    pinball no quantil TAU (os do regressor adotado). O alvo e dividido pelo
    desvio padrao so para treinar a rede (a perda pinball e equivariante a
    escala, o otimo nao muda)
  - XGBoost (topo do hibrido e referencia tabular) com
    objective='reg:quantileerror' e o espaco de busca do regressor adotado
  - score = -previsao na descida, +previsao na subida
  - 20 trials, GroupKFold(3) por sujeito, objetivo = media do AUC das 3 dobras,
    early stopping da rede pela perda pinball na dobra de validacao (como o
    hibrido do classificador faz com o AUC)
  - uma semente (SEMENTE), fixada no TF/numpy, no XGBoost e no amostrador
  - avaliacao final: rede treinada em dev['tr'] com early stopping em
    dev['va'] (GroupShuffleSplit 85/15), XGBoost em todo o dev, uma medicao
    no holdout

Saida: resumo_hibrido_quantilico.csv, cache_hibrido_quantilico/*.json,
       holdout_preds_hibrido_quantilico_{direcao}.csv.gz (para bootstrap
       emparelhado por sujeito), optuna_arquiteturas/optuna_HIBQ_*.db
Correr via correr_hibrido_quantilico.sh (reinicio automatico por memoria).
"""
import gc
import json
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # comparacao_arquiteturas_classificador.py está na pasta acima
from comparacao_arquiteturas_classificador import (
    BLOCO_SEG, D, FEAT_SEQ, FEAT_TAB, N_FOLDS, N_TRIALS, OUT, PASTA_OPTUNA, RAIZ_DADOS,
    T, W_MIN, _sair_se_memoria_alta, add_multi, construir_bilstm, construir_bitcn,
)

SEMENTE = 0
TAU = {'descida': 0.10, 'subida': 0.90}
EPOCAS = 30
MODELOS = ('XGB', 'BiLSTM', 'BiTCN')
CACHE_DIR = f"{OUT}/cache_hibrido_quantilico"
RESUMO = f"{OUT}/resumo_hibrido_quantilico.csv"
PREDS = f"{OUT}/holdout_preds_hibrido_quantilico_{{direcao}}.csv.gz"


def carregar(path, direcao):
    uc = list(dict.fromkeys(['subject_id', 'session_id', 'bloco_id', 'sbp_target',
                             'age', 'gender_bin'] + FEAT_SEQ))
    df = pd.read_csv(path, usecols=uc).sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
    df, novos = add_multi(df)
    W = int(W_MIN * 60 / BLOCO_SEG)
    df['delta'] = df['sbp_target'] - df.groupby('session_id')['sbp_target'].shift(W)
    df['y'] = ((df['delta'] <= -D) if direcao == 'descida' else (df['delta'] >= +D)).astype('float')
    # sem Delta nao ha alvo de regressao (o classificador punha estas linhas a 0)
    df.loc[df['delta'].isna(), 'y'] = np.nan
    return df, novos


def construir_sequencias(df, feat_tab):
    Xs, Xt, ys, ds, subj, sess, blo = [], [], [], [], [], [], []
    for sid, g in df.groupby('session_id'):
        g = g.sort_values('bloco_id')
        n = len(g)
        if n < T:
            continue
        sw = np.transpose(sliding_window_view(g[FEAT_SEQ].values.astype('float32'), T, axis=0),
                          (0, 2, 1))
        idx = np.arange(T - 1, n)
        tab, yv, dv = g[feat_tab].values[idx], g['y'].values[idx], g['delta'].values[idx]
        ok = (~np.isnan(tab).any(axis=1) & ~np.isnan(yv)
              & ~np.isnan(sw.reshape(len(sw), -1)).any(axis=1))
        if not ok.any():
            continue
        Xs.append(sw[ok]); Xt.append(tab[ok]); ys.append(yv[ok]); ds.append(dv[ok])
        subj.append(np.repeat(g['subject_id'].iloc[0], ok.sum()))
        sess.append(np.repeat(str(sid), ok.sum())); blo.append(g['bloco_id'].values[idx][ok])
    cat = np.concatenate
    return cat(Xs), cat(Xt), cat(ys), cat(ds), cat(subj), cat(sess), cat(blo)


def preparar(direcao):
    dev, novos = carregar(f"{RAIZ_DADOS}/dev_ppg.csv", direcao)
    hold, _ = carregar(f"{RAIZ_DADOS}/holdout_ppg.csv", direcao)
    feat_tab = FEAT_TAB + novos
    assert len(feat_tab) == 19, len(feat_tab)
    Xs_d, Xt_d, y_d, dl_d, subj_d, _, _ = construir_sequencias(dev, feat_tab)
    Xs_h, Xt_h, y_h, dl_h, subj_h, sess_h, blo_h = construir_sequencias(hold, feat_tab)
    del dev, hold
    F = Xs_d.shape[2]
    sc = StandardScaler().fit(Xs_d.reshape(-1, F))

    def norm(X):
        return sc.transform(X.reshape(-1, F)).reshape(X.shape).astype('float32')

    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
                  .split(Xs_d, y_d, subj_d))
    return dict(Xs_d_n=norm(Xs_d), Xt_d=Xt_d, y_d=y_d, dl_d=dl_d, subj_d=subj_d,
                Xs_h_n=norm(Xs_h), Xt_h=Xt_h, y_h=y_h, dl_h=dl_h, subj_h=subj_h,
                sess_h=sess_h, blo_h=blo_h, tr=tr, va=va, F=F)


def score_de(pred, direcao):
    return -pred if direcao == 'descida' else pred


def pinball(tau):
    def perda(y, yhat):
        e = y - yhat
        return tf.reduce_mean(tf.maximum(tau * e, (tau - 1.0) * e))
    return perda


def params_xgb(trial, prefixo=''):
    return dict(
        n_estimators=trial.suggest_int(f'{prefixo}n_estimators', 200, 800, step=100),
        learning_rate=trial.suggest_float(f'{prefixo}learning_rate', 0.01, 0.2, log=True),
        max_depth=trial.suggest_int(f'{prefixo}max_depth', 3, 9),
        min_child_weight=trial.suggest_int(f'{prefixo}min_child_weight', 1, 12),
        subsample=trial.suggest_float(f'{prefixo}subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float(f'{prefixo}colsample_bytree', 0.6, 1.0),
        gamma=trial.suggest_float(f'{prefixo}gamma', 1e-3, 5.0, log=True),
        reg_alpha=trial.suggest_float(f'{prefixo}reg_alpha', 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float(f'{prefixo}reg_lambda', 1e-3, 10.0, log=True),
    )


def params_rede(trial, nome):
    if nome == 'BiLSTM':
        return dict(
            learning_rate=trial.suggest_float('rede_learning_rate', 1e-4, 5e-3, log=True),
            lstm_units=trial.suggest_int('rede_lstm_units', 16, 128, step=16),
            dropout_rate=trial.suggest_float('rede_dropout_rate', 0.0, 0.4),
        )
    return dict(
        learning_rate=trial.suggest_float('rede_learning_rate', 1e-4, 3e-3, log=True),
        nb_filters=trial.suggest_int('rede_nb_filters', 16, 64, step=16),
        kernel_size=trial.suggest_int('rede_kernel_size', 3, 7),
        dropout_rate=trial.suggest_float('rede_dropout_rate', 0.1, 0.4),
    )


def separar_params(nome, best):
    if nome == 'XGB':
        return None, best
    return ({k[5:]: v for k, v in best.items() if k.startswith('rede_')},
            {k[4:]: v for k, v in best.items() if k.startswith('xgb_')})


def xgb_quantilico(xgb, tau):
    return XGBRegressor(**xgb, objective='reg:quantileerror', quantile_alpha=tau,
                        n_jobs=-1, tree_method='hist', random_state=SEMENTE)


def treinar_rede(nome, net, Xs_tr, d_tr, Xs_va, d_va, F, tau):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEMENTE)
    base = construir_bilstm(net, T, F) if nome == 'BiLSTM' else construir_bitcn(net, T, F)
    saida = Dense(1, activation='linear', name='quantil')(base.get_layer('emb').output)
    modelo = Model(base.input, saida)
    escala = float(np.std(d_tr)) or 1.0
    modelo.compile(optimizer=tf.keras.optimizers.Adam(net['learning_rate']), loss=pinball(tau))
    modelo.fit(Xs_tr, d_tr / escala, validation_data=(Xs_va, d_va / escala),
               epochs=EPOCAS, batch_size=512, verbose=0,
               callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=5,
                                        restore_best_weights=True)])
    return modelo


def com_embedding(nome, net, dados, tr, va, tau, conjuntos):
    """Treina a rede UMA vez (early stopping em va) e devolve, para cada par
    (Xs, Xt) de `conjuntos`, as tabulares com o embedding ao lado."""
    Xs, dl = dados['Xs_d_n'], dados['dl_d']
    modelo = treinar_rede(nome, net, Xs[tr], dl[tr], Xs[va], dl[va], dados['F'], tau)
    extr = Model(modelo.input, modelo.get_layer('emb').output)
    saida = [np.hstack([Xt_c, extr.predict(Xs_c, batch_size=1024, verbose=0)]) for Xs_c, Xt_c in conjuntos]
    del modelo, extr
    return saida


def auc_cv(nome, net, xgb, dados, direcao):
    tau, Xs, Xt, y, dl = TAU[direcao], dados['Xs_d_n'], dados['Xt_d'], dados['y_d'], dados['dl_d']
    aucs = []
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(Xt, y, dados['subj_d']):
        if nome == 'XGB':
            a_tr, a_va = Xt[tr], Xt[va]
        else:
            a_tr, a_va = com_embedding(nome, net, dados, tr, va, tau,
                                       [(Xs[tr], Xt[tr]), (Xs[va], Xt[va])])
        m = xgb_quantilico(xgb, tau).fit(a_tr, dl[tr])
        aucs.append(roc_auc_score(y[va], score_de(m.predict(a_va), direcao)))
        del m, a_tr, a_va
        gc.collect()
    return float(np.mean(aucs))


def otimizar(nome, direcao, dados):
    nf = f"HIBQ_{nome}_{direcao}"
    os.makedirs(PASTA_OPTUNA, exist_ok=True)
    study = optuna.create_study(direction='maximize',
                                storage=f"sqlite:///{PASTA_OPTUNA}/optuna_{nf}.db",
                                study_name=nf, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEMENTE))

    def objective(trial):
        if nome == 'XGB':
            return auc_cv('XGB', None, params_xgb(trial), dados, direcao)
        return auc_cv(nome, params_rede(trial, nome), params_xgb(trial, 'xgb_'), dados, direcao)

    feitos = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if N_TRIALS - feitos > 0:
        print(f"[HIBQ {nome}/{direcao}] trials completos: {feitos} | a correr mais {N_TRIALS - feitos}",
              flush=True)
        study.optimize(objective, n_trials=N_TRIALS - feitos, n_jobs=1, show_progress_bar=False,
                       callbacks=[_sair_se_memoria_alta])
    print(f"[HIBQ {nome}/{direcao}] melhor AUC CV = {study.best_value:.4f} | params: {study.best_params}",
          flush=True)
    return study.best_params, study.best_value


def avaliar_final(nome, best, dados, direcao):
    tau = TAU[direcao]
    net, xgb = separar_params(nome, best)
    if nome == 'XGB':
        a_d, a_h = dados['Xt_d'], dados['Xt_h']
    else:
        # a rede usa tr/va so para o early stopping; o XGBoost treina em todo o dev
        a_d, a_h = com_embedding(nome, net, dados, dados['tr'], dados['va'], tau,
                                 [(dados['Xs_d_n'], dados['Xt_d']), (dados['Xs_h_n'], dados['Xt_h'])])
    m = xgb_quantilico(xgb, tau).fit(a_d, dados['dl_d'])
    s = score_de(m.predict(a_h), direcao)
    return float(roc_auc_score(dados['y_h'], s)), s


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    linhas = []
    for direcao in ('descida', 'subida'):
        dados = None
        p_preds = PREDS.format(direcao=direcao)
        preds = pd.read_csv(p_preds) if os.path.exists(p_preds) else None
        auc_xgb = None
        for nome in MODELOS:
            cp = f"{CACHE_DIR}/{nome}_{direcao}.json"
            if os.path.exists(cp):
                c = json.load(open(cp))
                print(f"[HIBQ {nome}/{direcao}] em cache: HOLDOUT AUC = {c['auc_holdout']:.4f} (a saltar)",
                      flush=True)
            else:
                if dados is None:
                    t0 = time.time()
                    dados = preparar(direcao)
                    print(f"[{direcao}] dados prontos em {time.time()-t0:.0f}s | dev {dados['Xs_d_n'].shape} "
                          f"| holdout {dados['Xs_h_n'].shape} | eventos dev {dados['y_d'].mean()*100:.2f}% "
                          f"| holdout {dados['y_h'].mean()*100:.2f}% | tau {TAU[direcao]}", flush=True)
                if preds is None:
                    preds = pd.DataFrame(dict(subject_id=dados['subj_h'], session_id=dados['sess_h'],
                                              bloco_id=dados['blo_h'], y=dados['y_h'], delta=dados['dl_h']))
                t0 = time.time()
                best, cv = otimizar(nome, direcao, dados)
                auc, s = avaliar_final(nome, best, dados, direcao)
                preds[f'score_{nome}'] = s
                preds.to_csv(p_preds, index=False, compression='gzip')
                c = dict(auc_holdout=auc, auc_cv=cv, best_params=best, tau=TAU[direcao],
                         semente=SEMENTE, n_trials=N_TRIALS)
                json.dump(c, open(cp, "w"), indent=2)
                print(f"[HIBQ {nome}/{direcao}] HOLDOUT AUC = {auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if nome == 'XGB':
                auc_xgb = c['auc_holdout']
            linhas.append(dict(
                direcao=direcao,
                modelo='XGB_quantilico' if nome == 'XGB' else f'HIBRIDO_{nome}_XGB_quantilico',
                auc_cv=round(c['auc_cv'], 4), auc_holdout=round(c['auc_holdout'], 4),
                ganho_vs_xgb=None if nome == 'XGB' else round(c['auc_holdout'] - auc_xgb, 4),
                best_params=json.dumps(c['best_params'])))
            pd.DataFrame(linhas).to_csv(RESUMO, index=False)
        del dados
        gc.collect()
    print(f"\n{'='*74}\nRESULTADO FINAL (quantilica, busca conjunta, {N_TRIALS} trials x {N_FOLDS}-fold, "
          f"semente {SEMENTE})\n{'='*74}")
    print(pd.DataFrame(linhas)[['direcao', 'modelo', 'auc_cv', 'auc_holdout', 'ganho_vs_xgb']]
          .to_string(index=False))


if __name__ == "__main__":
    main()
