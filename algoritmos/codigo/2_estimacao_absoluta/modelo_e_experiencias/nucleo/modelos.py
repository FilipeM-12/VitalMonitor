"""Modelos do absoluto. Uma só função, `ajustar_e_prever`, treina e prevê qualquer família. É chamada igual pela
validação cruzada (busca de hiperparâmetros), pela avaliação no holdout e pelos testes de sementes.

Famílias (nome em maiúsculas):
  XGB                  AgeStratifiedXGBEnsemble: dois boosters, doentes < 60 anos e >= 60, com perda assimétrica
  BILSTM, BITCN        rede sobre uma janela de JANELA_SEQ blocos seguidos, prevê o desvio
  HIBRIDO_BILSTM_XGB   a rede é treinada e a sua penúltima camada (32 números) junta-se às características tabulares,
  HIBRIDO_BITCN_XGB    onde o XGBoost decide. Os hiperparâmetros da rede (rede_*) e do XGBoost (xgb_*) afinam-se juntos.
"""
import os
import random
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from . import config as C

FAMILIAS = ("XGB", "BILSTM", "BITCN", "HIBRIDO_BILSTM_XGB", "HIBRIDO_BITCN_XGB")
IDADE_CORTE = 60


def fixar_sementes(semente=C.SEMENTE):
    os.environ['PYTHONHASHSEED'] = str(semente)
    random.seed(semente)
    np.random.seed(semente)
    tf = sys.modules.get('tensorflow')          # só se já foi importado (o XGBoost não precisa do TensorFlow)
    if tf is not None:
        tf.random.set_seed(semente)


def configurar_tensorflow():
    import tensorflow as tf
    for g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


# ── XGBoost por faixa etária ──────────────────────────────────────────────────────────────────────────────────────
def _objetivo_assimetrico(fator):
    """Penaliza `fator` vezes mais subestimar uma subida > 12 mmHg ou sobrestimar uma descida < -12 mmHg."""
    def objetivo(preds, dtrain):
        y = dtrain.get_label()
        residuo = preds - y
        grad, hess = np.copy(residuo), np.ones_like(residuo)
        for mascara in ((y > 12) & (residuo < 0), (y < -12) & (residuo > 0)):
            grad[mascara] *= fator
            hess[mascara] *= fator
        return grad, hess
    return objetivo


class AgeStratifiedXGBEnsemble:
    def __init__(self, params):
        self.params = dict(params)
        self.modelo_jovem = self.modelo_senior = None
        self.fator_assimetria = self.params.pop('asymmetry_factor', 2.0)
        self.reg_alpha = self.params.pop('reg_alpha', 0.1)
        self.reg_lambda = self.params.pop('reg_lambda', 10.0)

    def fit(self, X, y, idades):
        p = dict(self.params)
        if 'learning_rate' in p:
            p['eta'] = p.pop('learning_rate')
        p.update(tree_method=p.get('tree_method', 'hist'), verbosity=0, alpha=self.reg_alpha, **{'lambda': self.reg_lambda})
        rondas = self.params.get('n_estimators', 100)
        obj = _objetivo_assimetrico(self.fator_assimetria)
        jovem, senior = idades < IDADE_CORTE, idades >= IDADE_CORTE
        y = np.asarray(y)
        if jovem.any():
            self.modelo_jovem = xgb.train(p, xgb.DMatrix(X[jovem], label=y[jovem]), num_boost_round=rondas, obj=obj)
        if senior.any():
            self.modelo_senior = xgb.train(p, xgb.DMatrix(X[senior], label=y[senior]), num_boost_round=rondas, obj=obj)

    def predict(self, X, idades):
        pred = np.zeros(len(X))
        jovem, senior = idades < IDADE_CORTE, idades >= IDADE_CORTE
        if jovem.any() and self.modelo_jovem is not None:
            pred[jovem] = self.modelo_jovem.predict(xgb.DMatrix(X[jovem]))
        if senior.any() and self.modelo_senior is not None:
            pred[senior] = self.modelo_senior.predict(xgb.DMatrix(X[senior]))
        if self.modelo_jovem is not None and senior.any() and self.modelo_senior is None:
            pred[senior] = self.modelo_jovem.predict(xgb.DMatrix(X[senior]))
        if self.modelo_senior is not None and jovem.any() and self.modelo_jovem is None:
            pred[jovem] = self.modelo_senior.predict(xgb.DMatrix(X[jovem]))
        return pred


# ── redes ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def construir_rede(rede, n_features, p):
    """BiTCN ou BiLSTM, terminadas em Dense(32) -> Dense(1). A penúltima camada é o embedding dos híbridos."""
    import tensorflow as tf
    from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Model
    entrada = Input(shape=(C.JANELA_SEQ, n_features))
    if rede == "BITCN":
        from tcn import TCN
        x = Bidirectional(TCN(nb_filters=p.get('nb_filters', 64), kernel_size=p.get('kernel_size', 4), nb_stacks=1,
                              dilations=[1, 2, 4, 8], padding='same', use_skip_connections=True,
                              dropout_rate=p.get('dropout_rate', 0.1), return_sequences=False))(entrada)
        x = Dense(64, activation='swish')(x)
        x = Dense(32, activation='swish')(x)
        lr = p.get('learning_rate', 0.0005)
    elif rede == "BILSTM":
        x = Bidirectional(LSTM(p.get('lstm_units_l1', 64), return_sequences=True))(entrada)
        x = Dropout(p.get('dropout_rate', 0.2))(x)
        x = Bidirectional(LSTM(p.get('lstm_units_l2', 32)))(x)
        x = Dense(32, activation='swish')(x)
        lr = p.get('learning_rate', 0.001)
    else:
        raise ValueError(rede)
    modelo = Model(inputs=entrada, outputs=Dense(1)(x))
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='huber')
    return modelo


def preparar_sequencias(df, features, janela=C.JANELA_SEQ):
    """Janelas deslizantes de `janela` blocos seguidos dentro de cada sessão. Devolve (X, y, índices da última linha)."""
    X, y, idx = [], [], []
    for _, g in df.groupby('session_id'):
        dados, alvo, ind = g[list(features)].values, g[C.ALVO].values, g.index.values
        for i in range(janela, len(dados) + 1):
            X.append(dados[i - janela:i])
            y.append(alvo[i - 1])
            idx.append(ind[i - 1])
    return np.array(X), np.array(y), np.array(idx)


def separar_params_hibrido(params):
    rede = {k[len('rede_'):]: v for k, v in params.items() if k.startswith('rede_')}
    arvore = {k[len('xgb_'):]: v for k, v in params.items() if k.startswith('xgb_')}
    return rede, arvore


# ── a função única ────────────────────────────────────────────────────────────────────────────────────────────────
def ajustar_e_prever(familia, params, dev, teste, features, semente=C.SEMENTE, parar_com_teste=False):
    """Treina em `dev` e prevê o desvio (sbp_delta) em `teste`. Devolve uma Series indexada pelas linhas de `teste`
    previstas (as redes precisam de JANELA_SEQ blocos, por isso não preveem os primeiros JANELA_SEQ-1 de cada sessão)."""
    familia = familia.upper()
    features = list(features)
    if familia != "XGB":
        import tensorflow  # noqa: F401  (antes das sementes, para que também fixem a do TensorFlow)
    fixar_sementes(semente)
    n_jobs = int(os.environ.get('HILOS_XGB', os.environ.get('NUCLEOS_MAX', os.cpu_count() or 1)))

    if familia == "XGB":
        sc = StandardScaler()
        p = dict(params)
        p.setdefault('n_jobs', n_jobs)
        modelo = AgeStratifiedXGBEnsemble(p)
        modelo.fit(sc.fit_transform(dev[features]), dev[C.ALVO].values, dev['age'].values)
        return pd.Series(modelo.predict(sc.transform(teste[features]), teste['age'].values), index=teste.index)

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.models import Model
    configurar_tensorflow()
    tf.keras.backend.clear_session()
    hibrido = familia.startswith("HIBRIDO_")
    rede = {"BILSTM": "BILSTM", "BITCN": "BITCN", "HIBRIDO_BILSTM_XGB": "BILSTM", "HIBRIDO_BITCN_XGB": "BITCN"}[familia]
    p_rede, p_arvore = separar_params_hibrido(params) if hibrido else (dict(params), None)

    X_d, y_d, i_d = preparar_sequencias(dev, features)
    X_t, _, i_t = preparar_sequencias(teste, features)
    if len(X_d) == 0 or len(X_t) == 0:
        return pd.Series(dtype=float)
    S, T, F = X_d.shape
    sc = StandardScaler()
    X_d = sc.fit_transform(X_d.reshape(-1, F)).reshape(S, T, F)
    X_t = sc.transform(X_t.reshape(-1, F)).reshape(-1, T, F)

    modelo = construir_rede(rede, F, p_rede)
    # Paragem antecipada, a mesma regra da tendência: na validação cruzada (parar_com_teste=True) com a dobra de validação;
    # no modelo final com 15 % dos SUJEITOS do desenvolvimento (GroupShuffleSplit). Nunca com o holdout.
    if parar_com_teste:
        X_t_val, y_t_val = X_t, teste.loc[i_t, C.ALVO].values
        tr = np.arange(len(X_d))
    else:
        from sklearn.model_selection import GroupShuffleSplit
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=C.SEMENTE)
                      .split(X_d, y_d, dev.loc[i_d, 'subject_id'].values))
        X_t_val, y_t_val = X_d[va], y_d[va]
    modelo.fit(X_d[tr], y_d[tr], epochs=15, batch_size=128, verbose=0, validation_data=(X_t_val, y_t_val),
               callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
    if not hibrido:
        return pd.Series(modelo.predict(X_t, verbose=0).flatten(), index=i_t)

    extrator = Model(modelo.input, modelo.layers[-2].output)
    sc_tab = StandardScaler()
    tab_d = sc_tab.fit_transform(dev.loc[i_d, features].values)
    tab_t = sc_tab.transform(teste.loc[i_t, features].values)
    p_arvore.setdefault('n_jobs', n_jobs)
    arvores = AgeStratifiedXGBEnsemble(p_arvore)
    arvores.fit(np.hstack([tab_d, extrator.predict(X_d, verbose=0)]), y_d, dev.loc[i_d, 'age'].values)
    pred = arvores.predict(np.hstack([tab_t, extrator.predict(X_t, verbose=0)]), teste.loc[i_t, 'age'].values)
    return pd.Series(pred, index=i_t)
