"""Base comum dos híbridos com a janela de 5 min ancorada no TEMPO: constantes, camadas e construtores iguais aos de
4_tendencia_selecao_e_experiencias/comparacao_de_arquiteturas_e_hibridos/comparacao_arquiteturas_classificador.py, com
RAIZ_DADOS (tabela da tendência) e OUT/PASTA_OPTUNA (esta pasta) próprios.
"""
import os

import optuna
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM
from tensorflow.keras.models import Model
from tcn import TCN

optuna.logging.set_verbosity(optuna.logging.WARNING)

for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

RAIZ_DADOS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dados"))   # tabela da tendência (construir_tabela_ppg_morf.py)
CONJUNTO = os.environ.get("CONJUNTO", "tudo")        # "base" = as 19 características de antes; "tudo" = as 43 (com as novas)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"resultados_{CONJUNTO}")
os.makedirs(OUT, exist_ok=True)
PASTA_OPTUNA = f"{OUT}/optuna"
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
