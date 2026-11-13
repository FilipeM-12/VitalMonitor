"""
Worker de tendencia instalado (servidor/workers): extracao so-PPG com as
caracteristicas morfologicas (extracao_ppg), derivacao causal por tempo exato
(derivacao_ppg, 43 caracteristicas, valores em falta passados como NaN) e o modelo
quantilico do servidor (joblib com 43 caracteristicas). Os hibridos usam a arquitetura afinada da
busca quantilica por tempo (cache_hibrido_quantilico), 11 caracteristicas na sequencia e as 43 na
parte tabular. 

Teste de carga do servico de tendencia: XGBoost (o que esta implantado) contra
os hibridos rede+XGBoost (BiLSTM+XGB e BiTCN+XGB, com a arquitetura afinada na
busca conjunta).

O QUE SE MEDE
  Uma unica instancia de worker (o processo do worker_tendencia_quantilica,
  com o mesmo ThreadPoolExecutor de 8 threads) a receber trabalho de N pulseiras
  simuladas. Cada pulseira gera um trabalho de janela de 5 s (stride) e, de 3 em
  3 janelas (bloco de 15 s), uma inferencia. Cada trabalho faz o mesmo que
  processar_lote: construir a lista de amostras do buffer (2500 amostras),
  extrair_features_janela_ppg, derivar_features_ppg (causal) e, quando o bloco
  fecha, a inferencia das duas direcoes (descida e subida).
  So a inferencia difere entre modelos:
    xgb    : 2 x Booster.predict sobre as 43 features (o modelo do joblib real)
    bilstm : 2 x [BiLSTM(T=20, F=11) -> camada 'emb' (8) -> Booster sobre 43+8]
    bitcn  : 2 x [BiTCN(T=20, F=11)  -> camada 'emb' (8) -> Booster sobre 43+8]

  Latencia = instante de fim do trabalho - instante em que deveria ter comecado.
  Um nivel de carga e "estavel" se o p95 da latencia < 5 s (o intervalo entre
  janelas de uma pulseira) e a fila esvazia no fim.

O QUE NAO E
  - Os pesos das redes sao aleatorios (as redes nao foram exportadas do treino):
    valido para tempo e memoria, sem qualquer valor para exatidao. O XGBoost do
    topo dos hibridos e ajustado sobre dados sinteticos com os hiperparametros
    afinados (mesmo n de arvores e profundidade maxima).
  - Nao ha broker MQTT nem base de dados no caminho: o custo de rede e de
    escrita e igual para os tres modelos e mede-se a parte que os distingue.
    Tambem fica de fora processar_via_rapida (FC/SpO2, comum aos tres).
  - Sinal: 6 aquisicoes reais da bancada (MAX30102) reproduzidas em ciclo pelo
    GeradorCSV do simulador (o gerador sintetico nao passa os portoes de
    qualidade da extracao so-PPG). Serve para o custo computacional; nao diz
    nada sobre a exatidao.
  - TensorFlow/XGBoost limitados a 1 thread de calculo cada, para representar
    um processo de worker num servidor onde correm varios em paralelo.

Uso: python3 teste_de_carga_tempo.py --modelo xgb|bilstm|bitcn [--duracao 45]
"""
import argparse
import copy
import json
import os
import resource
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # o teste e so de CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
T0_IMPORT = time.perf_counter()
import numpy as np  # noqa: E402

RAIZ = "/home/filipe/Desktop/Tese"
TRABALHADORES = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "servidor", "workers"))
sys.path.insert(0, TRABALHADORES)                # extracao_ppg, derivacao_ppg (as do worker instalado)
sys.path.insert(0, f"{RAIZ}/arduino/simulador")
CACHE = f"{RAIZ}/algoritmo-ml/Tentativa_22_10/tendencia_tempo_fixo/hibridos/resultados_tudo/cache_hibrido_quantilico"
JOBLIB = f"{RAIZ}/server/modelo_tendencia_ppg_quant.joblib"
SAIDA = os.path.dirname(os.path.abspath(__file__))

import joblib  # noqa: E402
import xgboost as xgb  # noqa: E402
import extracao_ppg as epp  # noqa: E402
import derivacao_ppg as dpp  # noqa: E402

FS, JANELA, MARGEM, STRIDE = 100, 1500, 500, 500
BUF = JANELA + 2 * MARGEM
T_SEQ = 20
FEAT_SEQ = ['hr_mean_z', 'hr_lag1', 'alpha_womersley_global_z', 'area_mean_global_z',
            'w50_global_z', 'ppg_crest_time_global_z', 'ppg_decay_time_global_z',
            'w25_global_z', 'w75_global_z', 'ppg_max_slope_norm_global_z', 'apg_b_a_global_z']
N_FONTES = 6
SEG_SINAL = 1650         # 900 s de aquecimento (historico de 10 min + blocos rejeitados) + folga para a carga
SEG_AQUEC = 1500
NIVEIS = [10, 25, 50, 100, 200, 400, 800, 1600, 3200]
AMOSTRAS_FEATS = []
PINFO = {"idade": 65, "genero": "M"}


def rss_mb():
    with open("/proc/self/status") as f:
        d = {l.split(":")[0]: int(l.split()[1]) for l in f if l.startswith(("VmRSS", "VmHWM"))}
    return d["VmRSS"] / 1024, d["VmHWM"] / 1024


# ───────────────────────── modelos ─────────────────────────
class ModeloXGB:
    def __init__(self):
        b = joblib.load(JOBLIB)
        self.boosters, self.feats, self.artefacto_bytes = {}, b["descida"]["feats"], 0
        for d in ("descida", "subida"):
            m = xgb.Booster()
            m.load_model(bytearray(b[d]["booster"]))
            m.set_param({"nthread": 1})
            self.boosters[d] = m
            self.artefacto_bytes += len(b[d]["booster"])
        self.estado_seq = None

    def novo_estado(self):
        return None

    def prever(self, feats, seq_estado):
        vec = np.array([[np.nan if feats.get(k) is None else float(feats[k]) for k in self.feats]])
        dm = xgb.DMatrix(vec)             # em falta = NaN, como no worker
        return {d: float(self.boosters[d].predict(dm)[0]) for d in self.boosters}


def construir_rede(nome, p):
    import tensorflow as tf
    from tensorflow.keras.layers import Bidirectional, Dense, Input, LSTM
    from tensorflow.keras.models import Model
    if nome == "bilstm":
        inp = Input(shape=(T_SEQ, len(FEAT_SEQ)))
        x = Bidirectional(LSTM(p["lstm_units"], return_sequences=False,
                               dropout=p["dropout_rate"]))(inp)
        emb = Dense(8, activation="relu", name="emb")(x)
        out = Dense(1, activation="sigmoid")(emb)
        return Model(inp, out)
    from tcn import TCN
    inp = Input(shape=(T_SEQ, len(FEAT_SEQ)))
    kw = dict(nb_filters=p["nb_filters"], kernel_size=p["kernel_size"], nb_stacks=1,
              dilations=[1, 2, 4, 8], padding="causal", use_skip_connections=True,
              dropout_rate=p["dropout_rate"], return_sequences=False)
    fwd = TCN(**kw)(inp)
    rev = tf.keras.layers.Lambda(lambda t: tf.reverse(t, axis=[1]))(inp)
    bwd = TCN(**kw)(rev)
    emb = Dense(8, activation="relu", name="emb")(tf.keras.layers.Concatenate()([fwd, bwd]))
    out = Dense(1, activation="sigmoid")(emb)
    return Model(inp, out)


class ModeloHibrido:
    def __init__(self, nome):
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        from tensorflow.keras.models import Model
        self.tf = tf
        b = joblib.load(JOBLIB)
        self.feats = b["descida"]["feats"]
        self.boosters, self.extratores, self.artefacto_bytes = {}, {}, 0
        self.info = {}
        rng = np.random.default_rng(0)
        for d in ("descida", "subida"):
            cfg = json.load(open(f"{CACHE}/{'BiLSTM' if nome == 'bilstm' else 'BiTCN'}_{d}.json"))
            bp = cfg["best_params"]
            rede_p = {k[5:]: v for k, v in bp.items() if k.startswith("rede_")}
            xgb_p = {k[4:]: v for k, v in bp.items() if k.startswith("xgb_")}
            tau = 0.10 if d == "descida" else 0.90
            tf.keras.utils.set_random_seed(1)
            modelo = construir_rede(nome, rede_p)
            extr = Model(modelo.input, modelo.get_layer("emb").output)
            fn = tf.function(lambda x, e=extr: e(x, training=False), reduce_retracing=True)
            fn(tf.zeros((1, T_SEQ, len(FEAT_SEQ)))).numpy()          # constroi o grafo
            self.extratores[d] = fn
            caminho = f"/tmp/_carga_{nome}_{d}.keras"
            modelo.save(caminho)
            self.artefacto_bytes += os.path.getsize(caminho)
            os.remove(caminho)
            # XGBoost do topo: mesmos hiperparametros, dados sinteticos com estrutura
            # suficiente para as arvores crescerem ate a profundidade maxima
            n, f = 20000, len(self.feats) + 8
            X = rng.normal(size=(n, f))
            z = X[:, :6] @ rng.normal(size=6) + np.sin(X[:, 6] * 2) + X[:, 7] * X[:, 8] \
                + rng.normal(scale=0.5, size=n)
            m = xgb.XGBRegressor(**xgb_p, objective="reg:quantileerror", quantile_alpha=tau,
                                 n_jobs=1, tree_method="hist").fit(X, z)
            bst = m.get_booster()
            bst.set_param({"nthread": 1})
            self.boosters[d] = bst
            self.artefacto_bytes += len(bst.save_raw("ubj"))
            self.info[d] = dict(rede=rede_p, n_arvores=xgb_p["n_estimators"],
                                prof_max=xgb_p["max_depth"],
                                n_parametros_rede=int(modelo.count_params()))
            del modelo

    def novo_estado(self):
        return deque(maxlen=T_SEQ)

    def prever(self, feats, seq):
        vec = np.array([[np.nan if feats.get(k) is None else float(feats[k]) for k in self.feats]])
        # no treino as sequencias com algum bloco em falta saem; aqui so interessa o custo, por isso o NaN passa a 0
        seq.append(np.nan_to_num(np.array([np.nan if feats.get(k) is None else feats[k] for k in FEAT_SEQ], dtype="float32")))
        if len(seq) < T_SEQ:
            return None
        x = np.stack(seq)[None]
        saida = {}
        for d in self.boosters:
            emb = self.extratores[d](self.tf.constant(x)).numpy()
            saida[d] = float(self.boosters[d].predict(xgb.DMatrix(np.hstack([vec, emb])))[0])
        return saida


# ───────────────────────── sinal e estados ─────────────────────────
def gerar_fontes():
    """Aquisicoes reais da bancada (MAX30102), reproduzidas em ciclo por reflexao."""
    from simulador_pulseira import GeradorCSV, CSV_PADRAO
    ficheiros = sorted(CSV_PADRAO.glob("*bruto*.csv"), key=lambda p: p.stat().st_size,
                       reverse=True)[:N_FONTES]
    fontes = []
    for f in ficheiros:
        g = GeradorCSV(f)
        blocos = [g.gerar(25) for _ in range(SEG_SINAL * FS // 25)]
        ir = np.concatenate([b[1] for b in blocos]); ecg = np.concatenate([b[0] for b in blocos])
        red = np.concatenate([b[2] for b in blocos])
        fontes.append([{"ecg": float(a), "ir": float(b), "red": float(c)}
                       for a, b, c in zip(ecg, ir, red)])
    return fontes


def fim_janela(k):
    return BUF + k * STRIDE          # indice (exclusivo) do fim da janela k


def processar(modelo, fonte, pos, estado_ppg, seq, tempos=None):
    t0 = time.perf_counter()
    snap = fonte[pos - BUF:pos]                       # equivale a list(buf)
    raw = epp.extrair_features_janela_ppg(snap, fs=FS, feature_window=JANELA,
                                          feature_margem=MARGEM)
    t1 = time.perf_counter()
    feats = dpp.derivar_features_ppg(raw, estado_ppg, PINFO)
    t2 = time.perf_counter()
    pred = None
    if feats is not None:
        pred = modelo.prever(feats, seq)
        if tempos is not None:
            AMOSTRAS_FEATS.append(dict(feats))
    t3 = time.perf_counter()
    if tempos is not None:
        tempos.append((t1 - t0, t2 - t1, t3 - t2, feats is not None, pred is not None))
    return t3


def aquecer(modelo, fontes):
    """Corre cada fonte ate ao fim do aquecimento e devolve os estados prontos."""
    estados, tempos = [], []
    n_janelas = (SEG_AQUEC * FS - BUF) // STRIDE
    for fonte in fontes:
        est, seq = dpp.EstadoSessaoPPG(), modelo.novo_estado()
        for k in range(n_janelas):
            processar(modelo, fonte, fim_janela(k), est, seq, tempos)
        estados.append((est, seq, n_janelas))
    return estados, tempos


def resumo(v):
    v = np.asarray(v, float)
    return {p: float(np.percentile(v, q)) for p, q in (("p50", 50), ("p95", 95), ("p99", 99))} | \
           {"max": float(v.max()), "media": float(v.mean())}


def nivel(modelo, fontes, estados, n, duracao):
    pulseiras = []
    for i in range(n):
        est, seq, k0 = estados[i % N_FONTES]
        pulseiras.append(dict(fonte=fontes[i % N_FONTES], est=copy.deepcopy(est),
                              seq=copy.deepcopy(seq), k0=k0, lock=threading.Lock(), fase=(i / n) * (STRIDE / FS)))
    n_por = int(duracao / (STRIDE / FS))
    eventos = sorted(((p["fase"] + k * STRIDE / FS, i, k)
                      for i, p in enumerate(pulseiras) for k in range(n_por)))
    resultados = []

    def trabalho(ag, i, k):
        p = pulseiras[i]
        t_ini = time.perf_counter()
        # cada pulseira e sequencial (mesmo comportamento do worker: os lotes de uma
        # pulseira nao se sobrepoem em estado); serializa por pulseira
        with p["lock"]:
            t = []
            fim = processar(modelo, p["fonte"], fim_janela(p["k0"] + k), p["est"], p["seq"], t)
        resultados.append((ag, t_ini, fim, t[0]))

    cpu0, w0 = time.process_time(), time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=8)
    base = time.perf_counter() + 0.5
    for ag, i, k in eventos:
        alvo = base + ag
        d = alvo - time.perf_counter()
        if d > 0.002:
            time.sleep(d)
        executor.submit(trabalho, alvo, i, k)
    executor.shutdown(wait=True)
    fim_tudo = time.perf_counter()
    cpu1 = time.process_time()
    lat = [fim - ag for ag, _, fim, _ in resultados]
    espera = [ini - ag for ag, ini, _, _ in resultados]
    bloco = [fim - ag for ag, _, fim, t in resultados if t[3]]
    cauda = fim_tudo - (base + eventos[-1][0])
    r = dict(n_pulseiras=n, trabalhos=len(resultados), trabalhos_por_s=len(resultados) / duracao,
             lat=resumo(lat), espera_fila=resumo(espera),
             lat_bloco=resumo(bloco) if bloco else None,
             cpu_nucleos=(cpu1 - cpu0) / (fim_tudo - w0), cauda_s=cauda,
             rss_mb=rss_mb()[0])
    r["estavel"] = bool(r["lat"]["p95"] < STRIDE / FS and cauda < STRIDE / FS)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True, choices=["xgb", "bilstm", "bitcn"])
    ap.add_argument("--duracao", type=float, default=45.0)
    ap.add_argument("--niveis", default=None)
    ap.add_argument("--sufixo", default="")
    a = ap.parse_args()
    niveis = [int(x) for x in a.niveis.split(",")] if a.niveis else NIVEIS

    rss_base = rss_mb()[0]
    t_imp = time.perf_counter() - T0_IMPORT
    t = time.perf_counter()
    modelo = ModeloXGB() if a.modelo == "xgb" else ModeloHibrido(a.modelo)
    t_carga = time.perf_counter() - t
    rss_modelo, _ = rss_mb()
    print(f"[{a.modelo}] imports {t_imp:.1f}s | carga+construcao {t_carga:.1f}s | "
          f"RSS {rss_base:.0f} -> {rss_modelo:.0f} MB", flush=True)

    t = time.perf_counter()
    fontes = gerar_fontes()
    print(f"[{a.modelo}] sinal sintetico gerado em {time.perf_counter()-t:.0f}s", flush=True)
    t = time.perf_counter()
    estados, tempos = aquecer(modelo, fontes)
    rss_pos_aquec = rss_mb()[0]
    print(f"[{a.modelo}] aquecimento em {time.perf_counter()-t:.0f}s", flush=True)
    tp = np.array([x[:3] for x in tempos])
    fez_feats = np.array([x[3] for x in tempos])
    fez_pred = np.array([x[4] for x in tempos])
    print(f"[{a.modelo}] janelas com features derivadas: {fez_feats.sum()}, com previsao: {fez_pred.sum()}", flush=True)
    assert fez_pred.sum() > 20, "aquecimento nao produziu previsoes: teste invalido"
    # custo de servico em thread unica, sem contencao (so nas janelas em que ha inferencia)
    micro = dict(extracao_ms=resumo(tp[:, 0] * 1e3), derivacao_ms=resumo(tp[:, 1] * 1e3),
                 inferencia_ms=resumo(tp[fez_pred, 2] * 1e3))
    print(f"[{a.modelo}] inferencia (ms, 1 thread): {micro['inferencia_ms']}", flush=True)

    # inferencia isolada: 300 chamadas sobre as features reais do aquecimento
    seq_iso = modelo.novo_estado()
    if seq_iso is not None:
        for f in AMOSTRAS_FEATS[:T_SEQ]:
            modelo.prever(f, seq_iso)
    ts = []
    for r in range(300):
        f = AMOSTRAS_FEATS[r % len(AMOSTRAS_FEATS)]
        t = time.perf_counter()
        modelo.prever(f, seq_iso)
        ts.append((time.perf_counter() - t) * 1e3)
    micro["inferencia_isolada_ms"] = resumo(ts)
    print(f"[{a.modelo}] inferencia isolada (ms): {micro['inferencia_isolada_ms']}", flush=True)

    linhas = []
    for n in niveis:
        r = nivel(modelo, fontes, estados, n, a.duracao)
        linhas.append(r)
        print(f"[{a.modelo}] N={n:5d} | {r['trabalhos_por_s']:7.1f} trab/s | lat p50 {r['lat']['p50']*1e3:8.1f} "
              f"p95 {r['lat']['p95']*1e3:9.1f} ms | CPU {r['cpu_nucleos']:.2f} nucleos | "
              f"RSS {r['rss_mb']:.0f} MB | estavel={r['estavel']}", flush=True)
        if not r["estavel"]:
            break
    rss_atual, rss_pico = rss_mb()
    out = dict(rss_pos_aquecimento_mb=rss_pos_aquec, modelo=a.modelo, duracao_nivel_s=a.duracao, imports_s=t_imp, carga_s=t_carga,
               rss_base_mb=rss_base, rss_modelo_mb=rss_modelo, rss_pico_mb=rss_pico,
               artefacto_bytes=modelo.artefacto_bytes,
               info=getattr(modelo, "info", None), micro=micro, niveis=linhas,
               maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    json.dump(out, open(f"{SAIDA}/carga_{a.modelo}{a.sufixo}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
