"""rede_pura_quantilica_tempo.py — as redes SOZINHAS (BiLSTM e BiTCN) como regressores quantílicos, com a janela de 5 min por tempo.

Completa a comparação de arquiteturas na formulação adotada (regressão quantílica sobre o Δ): XGBoost tabular, rede pura e híbrido (rede + XGBoost,
hibrido_quantilico_tempo.py). A rede é a mesma dos híbridos (mesmas camadas e perda pinball no quantil da direção), mas aqui a PREVISÃO DA PRÓPRIA REDE é o score
(sem XGBoost por cima): score = -previsão na descida e +previsão na subida. Mesmos dados e sequências de hibrido_quantilico_tempo.preparar (43 características
tabulares e 11 canais por instante; só entram as linhas com a sequência completa), 20 tentativas de Optuna por rede e direção, objetivo = AUC médio de 3 dobras
por doente, uma semente. Só a rede é afinada (learning rate, unidades/filtros, dropout), como na rede pura do classificador.

Saída (resultados_<conjunto>/): optuna/optuna_REDEQ_<rede>_<direção>.db, cache_rede_quantilica/*.json, holdout_preds_rede_quantilica_<direção>.csv.gz com as
colunas score_BiLSTM, score_BiTCN e score_XGB (este vem das previsões do XGBoost quantílico tabular de hibrido_quantilico_tempo.py, nas MESMAS linhas).
Uso: python3 rede_pura_quantilica_tempo.py        (reinício automático: correr_rede_pura_quantilica_tempo.sh)
"""
import gc
import json
import os
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import hibrido_quantilico_tempo as hq
from base_hib import N_FOLDS, N_TRIALS, OUT, PASTA_OPTUNA, _sair_se_memoria_alta

optuna.logging.set_verbosity(optuna.logging.WARNING)
CACHE = f"{OUT}/cache_rede_quantilica"
PREDS = f"{OUT}/holdout_preds_rede_quantilica_{{direcao}}.csv.gz"
RESUMO = f"{OUT}/resumo_rede_quantilica.csv"


def prever(modelo, Xs, direcao):
    return hq.score_de(modelo.predict(Xs, batch_size=1024, verbose=0).ravel(), direcao)


def auc_cv_rede(nome, net, dados, direcao):
    tau, Xs, y, dl = hq.TAU[direcao], dados['Xs_d_n'], dados['y_d'], dados['dl_d']
    aucs = []
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(Xs, y, dados['subj_d']):
        m = hq.treinar_rede(nome, net, Xs[tr], dl[tr], Xs[va], dl[va], dados['F'], tau)
        aucs.append(roc_auc_score(y[va], prever(m, Xs[va], direcao)))
        del m
        gc.collect()
    return float(np.mean(aucs))


def otimizar(nome, direcao, dados):
    nf = f"REDEQ_{nome}_{direcao}"
    os.makedirs(PASTA_OPTUNA, exist_ok=True)
    study = optuna.create_study(direction='maximize', storage=f"sqlite:///{PASTA_OPTUNA}/optuna_{nf}.db", study_name=nf, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=hq.SEMENTE))
    feitos = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if N_TRIALS - feitos > 0:
        print(f"[REDEQ {nome}/{direcao}] trials completos: {feitos} | a correr mais {N_TRIALS - feitos}", flush=True)
        study.optimize(lambda trial: auc_cv_rede(nome, hq.params_rede(trial, nome), dados, direcao), n_trials=N_TRIALS - feitos, n_jobs=1,
                       show_progress_bar=False, callbacks=[_sair_se_memoria_alta])
    print(f"[REDEQ {nome}/{direcao}] melhor AUC CV = {study.best_value:.4f} | params: {study.best_params}", flush=True)
    return {k[5:]: v for k, v in study.best_params.items() if k.startswith('rede_')}, study.best_value


def main():
    os.makedirs(CACHE, exist_ok=True)
    linhas = []
    for direcao in ('descida', 'subida'):
        dados = None
        p_preds = PREDS.format(direcao=direcao)
        preds = pd.read_csv(p_preds) if os.path.exists(p_preds) else None
        for nome in ('BiLSTM', 'BiTCN'):
            cp = f"{CACHE}/{nome}_{direcao}.json"
            if os.path.exists(cp):
                c = json.load(open(cp))
                print(f"[REDEQ {nome}/{direcao}] em cache: HOLDOUT AUC = {c['auc_holdout']:.4f} (a saltar)", flush=True)
            else:
                if dados is None:
                    dados = hq.preparar(direcao)
                if preds is None:
                    preds = pd.DataFrame(dict(subject_id=dados['subj_h'], session_id=dados['sess_h'], bloco_id=dados['blo_h'], y=dados['y_h'], delta=dados['dl_h']))
                    ref = hq.PREDS.format(direcao=direcao)             # o XGBoost tabular quantílico, nas mesmas linhas
                    if os.path.exists(ref):
                        r = pd.read_csv(ref)
                        assert len(r) == len(preds) and (r['y'].values == preds['y'].values).all()
                        preds['score_XGB'] = r['score_XGB'].values
                t0 = time.time()
                net, cv = otimizar(nome, direcao, dados)
                m = hq.treinar_rede(nome, net, dados['Xs_d_n'][dados['tr']], dados['dl_d'][dados['tr']], dados['Xs_d_n'][dados['va']], dados['dl_d'][dados['va']],
                                    dados['F'], hq.TAU[direcao])
                s = prever(m, dados['Xs_h_n'], direcao)
                auc = float(roc_auc_score(dados['y_h'], s))
                preds[f'score_{nome}'] = s
                preds.to_csv(p_preds, index=False, compression='gzip')
                c = dict(auc_holdout=auc, auc_cv=cv, best_params=net, tau=hq.TAU[direcao], semente=hq.SEMENTE, n_trials=N_TRIALS)
                json.dump(c, open(cp, 'w'), indent=2)
                print(f"[REDEQ {nome}/{direcao}] HOLDOUT AUC = {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
                del m
                gc.collect()
            linhas.append(dict(direcao=direcao, modelo=f'{nome}_pura_quantilica', auc_cv=round(c['auc_cv'], 4), auc_holdout=round(c['auc_holdout'], 4),
                               best_params=json.dumps(c['best_params'])))
            pd.DataFrame(linhas).to_csv(RESUMO, index=False)
        del dados
        gc.collect()
    print(f"\n{'=' * 74}\nRESULTADO FINAL (redes puras quantílicas, {N_TRIALS} trials x {N_FOLDS}-fold, semente {hq.SEMENTE})\n{'=' * 74}")
    print(pd.DataFrame(linhas)[['direcao', 'modelo', 'auc_cv', 'auc_holdout']].to_string(index=False))


if __name__ == "__main__":
    main()
