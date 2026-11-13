"""
worker_absoluto.py — Estimação CONTÍNUA da PA absoluta (ECG + PPG)
==================================================================
Variante que estima o valor absoluto da pressão, ancorada numa calibração
com manguito: sbp_pred = sbp_calib + modelo.predict(features_ml).

Dois níveis de processamento:
  Rápido  (1 s)  → BPM, SpO2, temp  → tabela medicao
  Lento   (5 s)  → features completas (raw + derivadas) → tabela medicao_absoluta

A configuração, o esquema da base de dados, a via rápida e a cadeia de
extracção/derivação vivem em comum.py, partilhados com as variantes de
tendência. Aqui fica só o que é próprio desta variante: o modelo de PA
absoluta, a inferência e o laço MQTT.

Modelo ML:
  Coloca modelo_sbp.joblib em modelos/. Se não existir, a previsão fica
  nula e o worker continua a gravar features.

Dependências: pip install paho-mqtt numpy scipy sqlalchemy psycopg2-binary joblib
"""

import json, os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import paho.mqtt.client as mqtt

from comum import (
    # configuração
    AUTH_TOKEN, MQTT_BROKER, MQTT_PORT, MQTT_USER, MQTT_PASS,
    NUM_WORKERS, WORKER_ID, THREAD_POOL_SIZE, log,
    # base de dados
    Session, MedicaoAbsoluta,
    get_sessao, nova_sessao, get_paciente_info, get_calibracao,
    # cadeia de sinal
    FS, FEATURE_WINDOW, FEATURE_STRIDE, FEATURE_MARGEM,
    FEATURES_ML_ORDER, SessionState,
    extrair_features_janela, derivar_features_ml,
    _get_state, _get_buf, _buf_lock, _feature_counters,
    # via rápida e repartição
    processar_via_rapida, is_mine,
)

# ─────────────────────────────────────────────
# MODELO ML (opcional)
# ─────────────────────────────────────────────
# Coloca modelo_sbp.joblib em modelos/.
# O modelo deve prever sbp_delta (Δ em relação à calibração), NÃO o SBP absoluto.
# sbp_final = sbp_calib + modelo_sbp.predict(features_ml)

# O caminho é resolvido a partir do PRÓPRIO ficheiro, não do cwd. Com um
# caminho relativo, o worker lançado pelo systemd (WorkingDirectory diferente)
# não encontrava o modelo e continuava a correr com a previsão desligada — só
# com um WARNING no log, que numa unidade de serviço passa despercebido. MODELO_
# SBP permite apontar para outro sítio (por exemplo um volume partilhado).
MODELO_PATH = os.environ.get(
    "MODELO_SBP", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelos",
                               "modelo_sbp.joblib"))

try:
    import joblib
    import xgboost as xgb
    _P = joblib.load(MODELO_PATH)
    _BOOST = {}
    for _k in ("booster_young", "booster_senior"):
        if _P.get(_k):
            _b = xgb.Booster()
            _b.load_model(bytearray(_P[_k]))
            _BOOST[_k] = _b
    MODELO = _P if _BOOST else None
    # o vector monta-se com .get(k, 0.0): uma feature que a derivação não produz
    # entraria como zero sem aviso. Melhor não prever.
    _em_falta = [k for k in _P["features"] if k not in FEATURES_ML_ORDER]
    if MODELO and _em_falta:
        raise ValueError(f"o modelo pede features que derivar_features_ml não produz: {_em_falta}")
    if MODELO:
        log.info("Modelo carregado: %s | %d features | dev %s",
                 _P["meta"]["variante"], len(_P["features"]), _P["meta"]["mae_dev_cv"])
except Exception as e:
    MODELO, _BOOST = None, {}
    log.warning("modelo não carregado de %s (%s) — previsão de TA DESACTIVADA, "
                "o worker continua a gravar features mas sbp_predicted fica nulo",
                MODELO_PATH, e)



def _kalman_passo(state: "SessionState", z: float, s: dict) -> float:
    """Um passo do filtro de nível local. Estado em state.kf_x / state.kf_P."""
    q, r_med, k_rob = s["q_over_r"] * s["r"], s["r"], s["robust_k"]
    if state.kf_x is None:
        state.kf_x, state.kf_P = float(z), 1e3
    state.kf_P += q
    S = state.kf_P + r_med
    inov = z - state.kf_x
    if k_rob is not None and S > 0:
        lim = k_rob * np.sqrt(S)
        inov = float(np.clip(inov, -lim, lim))
    K = state.kf_P / S
    state.kf_x += K * inov
    state.kf_P = (1.0 - K) * state.kf_P
    return state.kf_x


def prever_ta(features_ml: dict, cal: dict | None,
              state: "SessionState | None" = None) -> tuple[float | None, float | None]:
    """Cadeia de inferência completa, igual à do treino.

    O treinar_modelo_modular do notebook faz, por esta ordem:
      1. StandardScaler ajustado no dev
      2. previsão pelo estrato de idade (corte aos 60)
      3. subtrair a média das 3 primeiras previsões DA SESSÃO (âncora)
      4. filtro de Kalman causal por sessão (q_over_r=0.12, nível local, robust_k=4)
      5. sbp = sbp_calib + delta

    Só devolve valor depois da âncora estar formada (3 blocos).
    """
    if MODELO is None or cal is None or state is None:
        return None, None
    try:
        vec = np.array([[float(features_ml.get(k, 0.0)) for k in MODELO["features"]]])
        X = (vec - MODELO["scaler_mean"]) / MODELO["scaler_scale"]

        idade = features_ml.get("age", 0) or 0
        chave = "booster_senior" if idade >= MODELO["idade_corte"] else "booster_young"
        if chave not in _BOOST:                      # estrato sem modelo: usa o outro
            chave = next(iter(_BOOST))
        bruto = float(_BOOST[chave].predict(xgb.DMatrix(X))[0])

        # ── 3. âncora: média das N primeiras previsões da sessão ────────────
        n_anc = MODELO["ancora_offset_blocos"]
        s = MODELO["suavizacao"]
        if state.offset is None:
            state.preds_iniciais.append(bruto)
            if len(state.preds_iniciais) < n_anc:
                return None, None                    # âncora ainda a formar
            state.offset = float(np.mean(state.preds_iniciais))
            # ── 4a. reproduzir os passos do Kalman que ficaram para trás ────
            # No treino o filtro corre desde o bloco 0. Aqui o offset só é
            # conhecido ao N-ésimo bloco, por isso os N-1 primeiros passos têm de
            # ser reproduzidos agora — senão o estado do filtro nunca converge
            # para o mesmo.
            for b0 in state.preds_iniciais:
                _kalman_passo(state, b0 - state.offset, s)
            return round(cal["sbp"] + state.kf_x, 1), None

        # ── 4. Kalman causal de nível local, incremental ────────────────────
        _kalman_passo(state, bruto - state.offset, s)

        # ── 5. voltar a pressão absoluta ───────────────────────────────────
        return round(cal["sbp"] + state.kf_x, 1), None   # DBP não foi treinado
    except Exception as e:
        log.error("Erro na previsão ML: %s", e)
        return None, None


# ─────────────────────────────────────────────
# PROCESSAMENTO POR LOTE (chamado do thread pool)
# ─────────────────────────────────────────────

def processar_lote(pulseira_id: str, amostras: list) -> None:
    """
    Chamado a partir do ThreadPoolExecutor.
    numpy liberta o GIL → N pulseiras correm em paralelo real.
    """
    sessao = get_sessao(pulseira_id)
    if not sessao:
        return
    pac_id = sessao[0]
    nova_sessao(pulseira_id, sessao)   # outro doente ou nova atribuição: o comum apaga o estado

    buf = _get_buf(pulseira_id)

    with _buf_lock:
        for a in amostras:
            buf.append({
                "ecg": float(a.get("ecg", 0)),
                "ir":  float(a.get("ir",  0)),
                "red": float(a.get("red", 0)),
            })
        _feature_counters[pulseira_id] = _feature_counters.get(pulseira_id, 0) + len(amostras)
        snap = list(buf)

    # ── RÁPIDO: BPM + SpO2 (1×/s) ────────────────────────────────────────
    processar_via_rapida(pulseira_id, pac_id, amostras, snap)

    # ── LENTO: Features completas (1×/5s = a cada FEATURE_STRIDE novas amostras) ──
    fc = _feature_counters.get(pulseira_id, 0)
    if fc < FEATURE_STRIDE or len(snap) < FEATURE_WINDOW + 2 * FEATURE_MARGEM:
        return
    _feature_counters[pulseira_id] = max(0, fc - FEATURE_STRIDE)

    state = _get_state(pulseira_id)
    raw = extrair_features_janela(snap, fs=FS, estado=state)

    if get_sessao(pulseira_id) != sessao:
        return                      # a sessão mudou a meio do lote

    pinfo = get_paciente_info(pac_id)

    # derivar_features_ml TEM de ser chamada mesmo com raw=None. A chamada conta
    # para o alinhamento dos blocos (bloco_id = janela // 3), tal como no treino,
    # onde uma janela rejeitada continua a ocupar o seu window_idx. Saltá-la faz
    # os blocos agruparem as janelas erradas e desalinha o relógio da sessão.
    fml = derivar_features_ml(raw, state, pinfo)

    if fml is None:
        return                      # só se grava quando um bloco fecha

    cal = get_calibracao(pulseira_id)
    sbp_pred, dbp_pred = prever_ta(fml, cal, state) if fml else (None, None)

    s = Session()
    try:
        s.add(MedicaoAbsoluta(
            pulseira_id   = pulseira_id,
            paciente_id   = pac_id,
            features_ml   = json.dumps(fml) if fml else None,
            sbp_predicted = sbp_pred,
            dbp_predicted = dbp_pred,
        ))
        s.commit()
        if raw:
            log.debug("%s → PAT=%.3f HR=%.1f SBP_pred=%s",
                      pulseira_id, raw["pat_mean"], raw["hr_mean"], sbp_pred)
    except Exception as e:
        s.rollback(); log.error("DB Features %s: %s", pulseira_id, e)
    finally:
        Session.remove()


# ─────────────────────────────────────────────
# MQTT
# ─────────────────────────────────────────────

executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)




def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe("pulseiras/raw/#", qos=0)
        log.info("MQTT ligado")
    else:
        log.error("MQTT recusado rc=%d", rc)


def on_message(client, userdata, msg):
    try:
        pid = msg.topic.split("/")[2]
        if not is_mine(pid):
            return
        payload = json.loads(msg.payload.decode())
        if payload.get("token") != AUTH_TOKEN:
            return
        amostras = payload.get("amostras", [])
        if amostras:
            executor.submit(processar_lote, pid, amostras)
    except Exception as e:
        log.error("on_message: %s", e)


if __name__ == "__main__":
    log.info("Worker %d/%d | threads=%d | %s:%d",
             WORKER_ID, NUM_WORKERS, THREAD_POOL_SIZE, MQTT_BROKER, MQTT_PORT)

    client = mqtt.Client(client_id=f"worker-{WORKER_ID}", clean_session=True)
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.on_connect    = on_connect
    client.on_message    = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        log.info("A parar worker...")
    finally:
        executor.shutdown(wait=False)
        client.disconnect()
