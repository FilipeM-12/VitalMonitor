"""
Comparacao de arquiteturas para deteccao de tendencia (classificador binario): XGBoost tabular, BiLSTM e BiTCN,
e os hibridos rede + XGBoost, com 20 trials de Optuna por arquitetura (GroupKFold(3) por sujeito).

Usa os dados causais ({dev,holdout}_ppg.csv) e reaproveita:
  - add_multi / carregar / construir_sequencias (carregamento e features)
  - a definicao das 3 arquiteturas (XGB tabular, BiLSTM, BiTCN) e o padrao
    "hibrido": treinar a rede como classificador puro, extrair o embedding
    da penultima camada, e treinar um XGB por cima de [tabular + embedding]

Cada arquitetura tem os seus hiperparametros escolhidos por Optuna (20 trials), com o AUC do modelo "puro" (a
arquitetura em si, sem o XGB por cima) como objetivo. O "hibrido" calcula-se uma unica vez no final, com a melhor
configuracao de cada arquitetura (o XGB do hibrido usa sempre os MESMOS hiperparametros fixos, porque o que se
compara aqui e a capacidade da rede temporal, nao o XGB). O XGB tabular tambem usa Optuna, por coerencia.

Selecao: o objetivo de cada trial e o AUC medio de um GroupKFold(3) por sujeito, e nao o de um unico split de
validacao, que sobreajusta ao criterio de selecao (o "vencedor" tende a ter sorte nesse split). 20 trials x 3 dobras
custam ~o mesmo que 50 trials x 1 split, e o planalto demonstrado sugere que 20 candidatos bem avaliados chegam tao
longe como 50 mal avaliados. O split GroupShuffleSplit 85/15 (`dados['tr']`/`dados['va']`) serve so para o early
stopping do retreino final (avaliar_final_seq), que treina em todo o desenvolvimento e mede uma unica vez no holdout.

Correr para as duas direcoes (descida, subida).
"""
import gc
import json
import os
import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import optuna
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tcn import TCN

optuna.logging.set_verbosity(optuna.logging.WARNING)

for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

RAIZ_DADOS = "/home/filipe/Desktop/Tese/algoritmo-ml/2026-09-09_normalizacao_causal_tendencia/dados"
OUT = os.path.dirname(os.path.abspath(__file__))
PASTA_OPTUNA = f"{OUT}/optuna_arquiteturas"
BLOCO_SEG = 15
D, W_MIN = 15, 5
ESCALAS_MIN = [2, 5, 10]
T = 20
N_TRIALS = 20
N_FOLDS = 3
LIMITE_RSS_GB = 15


def _sair_se_memoria_alta(study, trial):
    """Fuga de memoria do TF/Keras: reinicia o processo entre trials (o trial ja
    esta gravado no .db) antes de o kernel matar por OOM a meio de um trial."""
    with open("/proc/self/status") as f:
        rss_gb = next(int(l.split()[1]) for l in f if l.startswith("VmRSS")) / 1024 / 1024
    if rss_gb > LIMITE_RSS_GB:
        print(f"RSS={rss_gb:.1f}GB > {LIMITE_RSS_GB}GB apos trial {trial.number}: a reiniciar o processo", flush=True)
        os._exit(75)

FEAT_SEQ = ['hr_mean_z', 'hr_lag1', 'alpha_womersley_global_z', 'area_mean_global_z',
            'w50_global_z', 'ppg_crest_time_global_z', 'ppg_decay_time_global_z']
SLOPE_COLS = ['hr_mean_z', 'area_mean_global_z', 'alpha_womersley_global_z']
VOL_COL = 'hr_mean_z'
FEAT_TAB = FEAT_SEQ + ['age', 'gender_bin']

# hiperparametros fixos do XGB usado SO no topo do hibrido (nao afinados
# aqui -- o que se afina em cada trial e a rede temporal por baixo)
XGB_HIBRIDO_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4,
                           min_child_weight=6, subsample=0.66, colsample_bytree=0.99,
                           reg_alpha=7.6, reg_lambda=8.3, eval_metric='auc',
                           n_jobs=-1, tree_method='hist')


def add_multi(df):
    gb = df.groupby('session_id')
    novos = []
    for c in SLOPE_COLS:
        for m in ESCALAS_MIN:
            Wm = int(m * 60 / BLOCO_SEG)
            df[f'd_{c}_{m}m'] = df[c] - gb[c].shift(Wm)
            novos.append(f'd_{c}_{m}m')
    W5 = int(5 * 60 / BLOCO_SEG)
    df[f'vol_{VOL_COL}_5m'] = gb[VOL_COL].transform(lambda s: s.rolling(W5, min_periods=W5).std())
    novos.append(f'vol_{VOL_COL}_5m')
    return df, novos


def carregar(path, direcao):
    uc = list(dict.fromkeys(['subject_id', 'session_id', 'bloco_id', 'sbp_target',
                              'age', 'gender_bin'] + FEAT_SEQ))
    df = pd.read_csv(path, usecols=uc).sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
    df, novos = add_multi(df)
    W = int(W_MIN * 60 / BLOCO_SEG)
    gb = df.groupby('session_id')
    delta = df['sbp_target'] - gb['sbp_target'].shift(W)
    df['y'] = ((delta <= -D) if direcao == 'descida' else (delta >= +D)).astype('float')
    return df, novos


def construir_sequencias(df, feat_tab):
    Xs, Xt, ys, subj, sess, blo = [], [], [], [], [], []
    for sid, g in df.groupby('session_id'):
        g = g.sort_values('bloco_id')
        Fseq = g[FEAT_SEQ].values.astype('float32')
        n = len(g)
        if n < T:
            continue
        sw = sliding_window_view(Fseq, T, axis=0)
        sw = np.transpose(sw, (0, 2, 1))
        idx_last = np.arange(T - 1, n)
        tab = g[feat_tab].values[idx_last]
        yv = g['y'].values[idx_last]
        ok = ~np.isnan(tab).any(axis=1) & ~np.isnan(yv) & ~np.isnan(sw.reshape(len(sw), -1)).any(axis=1)
        if ok.sum() == 0:
            continue
        Xs.append(sw[ok]); Xt.append(tab[ok]); ys.append(yv[ok])
        subj.append(np.repeat(g['subject_id'].iloc[0], ok.sum()))
        sess.append(np.repeat(str(sid), ok.sum())); blo.append(g['bloco_id'].values[idx_last][ok])
    return (np.concatenate(Xs), np.concatenate(Xt), np.concatenate(ys),
            np.concatenate(subj), np.concatenate(sess), np.concatenate(blo))


def preparar(direcao):
    dev, novos = carregar(f"{RAIZ_DADOS}/dev_ppg.csv", direcao)
    hold, _ = carregar(f"{RAIZ_DADOS}/holdout_ppg.csv", direcao)
    feat_tab = FEAT_TAB + novos
    assert len(feat_tab) == 19, len(feat_tab)
    Xs_d, Xt_d, y_d, subj_d, _, _ = construir_sequencias(dev, feat_tab)
    Xs_h, Xt_h, y_h, subj_h, sess_h, blo_h = construir_sequencias(hold, feat_tab)
    F = Xs_d.shape[2]
    sc = StandardScaler().fit(Xs_d.reshape(-1, F))
    Xs_d_n = sc.transform(Xs_d.reshape(-1, F)).reshape(Xs_d.shape).astype('float32')
    Xs_h_n = sc.transform(Xs_h.reshape(-1, F)).reshape(Xs_h.shape).astype('float32')
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr, va = next(gss.split(Xs_d_n, y_d, subj_d))
    return dict(feat_tab=feat_tab, Xs_d_n=Xs_d_n, Xt_d=Xt_d, y_d=y_d, subj_d=subj_d,
                Xs_h_n=Xs_h_n, Xt_h=Xt_h, y_h=y_h, sess_h=sess_h, blo_h=blo_h,
                tr=tr, va=va, F=F)


def treinar_xgb_tabular(params, dados):
    """GroupKFold(3) por sujeito -- a metrica de cada candidato e a media das
    3 dobras, nao o resultado de um unico split (ver a selecao no topo do ficheiro)."""
    y_d, Xt_d, subj_d = dados['y_d'], dados['Xt_d'], dados['subj_d']
    aucs = []
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(Xt_d, y_d, subj_d):
        pw = (len(y_d[tr]) - y_d[tr].sum()) / max(y_d[tr].sum(), 1)
        m = XGBClassifier(**params, scale_pos_weight=pw, eval_metric='auc', n_jobs=-1, tree_method='hist')
        m.fit(Xt_d[tr], y_d[tr])
        p_va = m.predict_proba(Xt_d[va])[:, 1]
        aucs.append(roc_auc_score(y_d[va], p_va))
    return float(np.mean(aucs))


def construir_bilstm(params, T, F):
    inp = Input(shape=(T, F))
    x = Bidirectional(LSTM(params['lstm_units'], return_sequences=False,
                            dropout=params['dropout_rate']))(inp)
    emb = Dense(8, activation='relu', name='emb')(x)
    out = Dense(1, activation='sigmoid')(emb)
    return Model(inp, out)


def construir_bitcn(params, T, F):
    inp2 = Input(shape=(T, F))
    fwd = TCN(nb_filters=params['nb_filters'], kernel_size=params['kernel_size'], nb_stacks=1,
              dilations=[1, 2, 4, 8], padding='causal', use_skip_connections=True,
              dropout_rate=params['dropout_rate'], return_sequences=False)(inp2)
    inp2_rev = tf.keras.layers.Lambda(lambda t: tf.reverse(t, axis=[1]))(inp2)
    bwd = TCN(nb_filters=params['nb_filters'], kernel_size=params['kernel_size'], nb_stacks=1,
              dilations=[1, 2, 4, 8], padding='causal', use_skip_connections=True,
              dropout_rate=params['dropout_rate'], return_sequences=False)(inp2_rev)
    concat = tf.keras.layers.Concatenate()([fwd, bwd])
    emb2 = Dense(8, activation='relu', name='emb')(concat)
    out2 = Dense(1, activation='sigmoid')(emb2)
    return Model(inp2, out2)


def treinar_seq_trial(nome, params, dados, epochs=30):
    """Treina a rede sequencial (BiLSTM/BiTCN) com GroupKFold(3) por sujeito
    e devolve o AUC medio das 3 dobras. Usado dentro do objective() do
    Optuna (ver a selecao no topo do ficheiro)."""
    y_d, subj_d = dados['y_d'], dados['subj_d']
    aucs = []
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(dados['Xs_d_n'], y_d, subj_d):
        tf.keras.backend.clear_session()
        pw = (len(y_d[tr]) - y_d[tr].sum()) / max(y_d[tr].sum(), 1)
        cw = {0: 1.0, 1: float(pw)}
        modelo = (construir_bilstm(params, T, dados['F']) if nome == 'BiLSTM'
                  else construir_bitcn(params, T, dados['F']))
        modelo.compile(optimizer=tf.keras.optimizers.Adam(params['learning_rate']),
                       loss='binary_crossentropy', metrics=['AUC'])
        modelo.fit(dados['Xs_d_n'][tr], y_d[tr], validation_data=(dados['Xs_d_n'][va], y_d[va]),
                   epochs=epochs, batch_size=512, class_weight=cw, verbose=0,
                   callbacks=[EarlyStopping(monitor='val_AUC', mode='max', patience=5,
                                             restore_best_weights=True)])
        p_va = modelo.predict(dados['Xs_d_n'][va], batch_size=1024, verbose=0).ravel()
        aucs.append(roc_auc_score(y_d[va], p_va))
        del modelo
        gc.collect()
    return float(np.mean(aucs))


def avaliar_final_seq(nome, params, dados):
    """Retreina com a melhor configuracao em TODO o dev (tr+va) e avalia uma
    unica vez no holdout -- puro e hibrido (XGB sobre [tabular+embedding])."""
    tf.keras.backend.clear_session()
    y_d = dados['y_d']
    pw = (len(y_d) - y_d.sum()) / max(y_d.sum(), 1)
    cw = {0: 1.0, 1: float(pw)}
    modelo = (construir_bilstm(params, T, dados['F']) if nome == 'BiLSTM'
              else construir_bitcn(params, T, dados['F']))
    modelo.compile(optimizer=tf.keras.optimizers.Adam(params['learning_rate']),
                   loss='binary_crossentropy', metrics=['AUC'])
    # early stopping com uma fatia do proprio dev (nunca o holdout)
    modelo.fit(dados['Xs_d_n'][dados['tr']], y_d[dados['tr']],
               validation_data=(dados['Xs_d_n'][dados['va']], y_d[dados['va']]),
               epochs=30, batch_size=512, class_weight=cw, verbose=0,
               callbacks=[EarlyStopping(monitor='val_AUC', mode='max', patience=5,
                                         restore_best_weights=True)])
    p_puro = modelo.predict(dados['Xs_h_n'], batch_size=1024, verbose=0).ravel()
    auc_puro = roc_auc_score(dados['y_h'], p_puro)

    extr = Model(modelo.input, modelo.get_layer('emb').output)
    emb_d = extr.predict(dados['Xs_d_n'], batch_size=1024, verbose=0)
    emb_h = extr.predict(dados['Xs_h_n'], batch_size=1024, verbose=0)
    Xh_d = np.hstack([dados['Xt_d'], emb_d])
    Xh_h = np.hstack([dados['Xt_h'], emb_h])
    xgb = XGBClassifier(**XGB_HIBRIDO_PARAMS, scale_pos_weight=pw)
    xgb.fit(Xh_d, y_d)
    p_hyb = xgb.predict_proba(Xh_h)[:, 1]
    auc_hyb = roc_auc_score(dados['y_h'], p_hyb)
    del modelo, extr, xgb, emb_d, emb_h, Xh_d, Xh_h
    gc.collect()
    return auc_puro, auc_hyb, p_puro, p_hyb


def otimizar(nome, direcao, dados):
    nome_ficheiro = f"{nome}_{direcao}_arquiteturas"
    os.makedirs(PASTA_OPTUNA, exist_ok=True)
    db = f"sqlite:///{PASTA_OPTUNA}/optuna_{nome_ficheiro}.db"
    study = optuna.create_study(direction="maximize", storage=db,
                                 study_name=nome_ficheiro, load_if_exists=True)

    if nome == 'XGB':
        def objective(trial):
            params = dict(
                n_estimators=trial.suggest_int('n_estimators', 200, 800, step=100),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                max_depth=trial.suggest_int('max_depth', 3, 9),
                min_child_weight=trial.suggest_int('min_child_weight', 1, 12),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            )
            return treinar_xgb_tabular(params, dados)
        n_jobs = 3
    elif nome == 'BiLSTM':
        def objective(trial):
            params = dict(
                learning_rate=trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
                lstm_units=trial.suggest_int('lstm_units', 16, 128, step=16),
                dropout_rate=trial.suggest_float('dropout_rate', 0.0, 0.4),
            )
            return treinar_seq_trial('BiLSTM', params, dados)
        n_jobs = 1
    else:  # BiTCN
        def objective(trial):
            params = dict(
                learning_rate=trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True),
                nb_filters=trial.suggest_int('nb_filters', 16, 64, step=16),
                kernel_size=trial.suggest_int('kernel_size', 3, 7),
                dropout_rate=trial.suggest_float('dropout_rate', 0.1, 0.4),
            )
            return treinar_seq_trial('BiTCN', params, dados)
        n_jobs = 1

    feitos = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if N_TRIALS - feitos > 0:
        print(f"[{nome}/{direcao}] trials completos: {feitos} | a correr mais {N_TRIALS - feitos}", flush=True)
        study.optimize(objective, n_trials=N_TRIALS - feitos, n_jobs=n_jobs, show_progress_bar=False,
                       callbacks=[_sair_se_memoria_alta])
    print(f"[{nome}/{direcao}] melhor valor (AUC va): {study.best_value:.4f} | params: {study.best_params}", flush=True)
    return study.best_params, study.best_value


CACHE_DIR = f"{OUT}/cache_resultados_finais"


def cache_path(direcao, nome):
    return f"{CACHE_DIR}/{nome}_{direcao}.json"


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    resultados_finais = []
    for direcao in ('descida', 'subida'):
        print(f"\n{'#'*74}\n# DIRECAO: {direcao.upper()}\n{'#'*74}", flush=True)
        t0 = time.time()
        dados = preparar(direcao)
        print(f"[{direcao}] dados prontos em {time.time()-t0:.0f}s | "
              f"dev seq {dados['Xs_d_n'].shape} | holdout seq {dados['Xs_h_n'].shape} | "
              f"eventos dev {dados['y_d'].mean()*100:.2f}% | eventos holdout {dados['y_h'].mean()*100:.2f}%",
              flush=True)

        # --- XGB tabular --------------------------------------------------
        cp_xgb = cache_path(direcao, 'XGB')
        if os.path.exists(cp_xgb):
            cached = json.load(open(cp_xgb))
            bp_xgb, auc_xgb = cached['best_params'], cached['auc_puro']
            print(f"[XGB/{direcao}] resultado final em cache: HOLDOUT AUC = {auc_xgb:.4f} "
                  f"(a saltar recalculo)", flush=True)
            p_xgb = None
        else:
            t0 = time.time()
            bp_xgb, best_va_xgb = otimizar('XGB', direcao, dados)
            pw = (len(dados['y_d']) - dados['y_d'].sum()) / max(dados['y_d'].sum(), 1)
            xgb_final = XGBClassifier(**bp_xgb, scale_pos_weight=pw, eval_metric='auc',
                                       n_jobs=-1, tree_method='hist')
            xgb_final.fit(dados['Xt_d'], dados['y_d'])
            p_xgb = xgb_final.predict_proba(dados['Xt_h'])[:, 1]
            auc_xgb = float(roc_auc_score(dados['y_h'], p_xgb))
            print(f"[XGB/{direcao}] HOLDOUT AUC = {auc_xgb:.4f} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(dict(auc_puro=auc_xgb, best_params=bp_xgb), open(cp_xgb, "w"), indent=2)
        resultados_finais.append(dict(direcao=direcao, modelo='XGB_tabular',
                                       auc_puro=round(float(auc_xgb), 4), auc_hibrido=None,
                                       best_params=json.dumps(bp_xgb)))

        # --- BiLSTM / BiTCN, puro + hibrido --------------------------------
        p_hold_extra = {'session_id': dados['sess_h'], 'bloco_id': dados['blo_h'], 'y': dados['y_h']}
        if p_xgb is not None:
            p_hold_extra['p_xgb_tabular'] = p_xgb
        for nome in ('BiLSTM', 'BiTCN'):
            cp = cache_path(direcao, nome)
            if os.path.exists(cp):
                cached = json.load(open(cp))
                bp, auc_puro, auc_hyb = cached['best_params'], cached['auc_puro'], cached['auc_hibrido']
                print(f"[{nome}/{direcao}] resultado final em cache: puro={auc_puro:.4f} "
                      f"hibrido={auc_hyb:.4f} (a saltar recalculo)", flush=True)
            else:
                t0 = time.time()
                bp, best_va = otimizar(nome, direcao, dados)
                auc_puro, auc_hyb, p_puro, p_hyb = avaliar_final_seq(nome, bp, dados)
                auc_puro, auc_hyb = float(auc_puro), float(auc_hyb)
                print(f"[{nome}/{direcao}] HOLDOUT puro={auc_puro:.4f} hibrido={auc_hyb:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                json.dump(dict(auc_puro=auc_puro, auc_hibrido=auc_hyb, best_params=bp),
                          open(cp, "w"), indent=2)
                p_hold_extra[f'p_{nome.lower()}_puro'] = p_puro
                p_hold_extra[f'p_{nome.lower()}_hibrido'] = p_hyb
            resultados_finais.append(dict(direcao=direcao, modelo=nome,
                                           auc_puro=round(float(auc_puro), 4),
                                           auc_hibrido=round(float(auc_hyb), 4),
                                           best_params=json.dumps(bp)))

        pd.DataFrame(resultados_finais).to_csv(f"{OUT}/resumo_arquiteturas.csv", index=False)
        if len(p_hold_extra) > 3:
            pd.DataFrame(p_hold_extra).to_csv(f"{OUT}/holdout_preds_arquiteturas_{direcao}.csv.gz",
                                                index=False, compression='gzip')
        print("Resultados parciais gravados em resumo_arquiteturas.csv", flush=True)

    print(f"\n{'='*74}\nRESULTADO FINAL (arquiteturas, causal, {N_TRIALS} trials x {N_FOLDS}-fold cada)\n{'='*74}")
    print(pd.DataFrame(resultados_finais).to_string(index=False))


if __name__ == "__main__":
    main()
