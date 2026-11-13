"""
worker_tendencia_quantilica.py — deteção de TENDÊNCIA por REGRESSÃO
QUANTÍLICA sobre o Δ da pressão, usando SÓ o PPG.

O MODELO
  Treina sobre o Δ contínuo da pressão, com perda quantílica (percentil 10 na descida, 90 na subida), portanto aprende a cauda da
  distribuição, que é justamente onde os eventos vivem. Uma etiqueta binária deitaria fora a magnitude: -14 mmHg e +50 mmHg
  seriam ambos "não-evento".

  Resultados do modelo instalado: ver o campo `meta` do joblib (treinado a
  43 características, 1 102 650 blocos de desenvolvimento).

O SCORE
  É a QUEDA PREVISTA EM mmHg (percentil 10 do Δ; para a descida usa-se o simétrico, para que valores
  maiores continuem a significar "mais provável o evento"). Escala típica:
  ~2 a 20 mmHg.

  O limiar e o portão estão nesta escala e vêm no próprio joblib, escolhidos no dev. No modelo instalado:
      descida  limiar 11,34 mmHg
      subida   limiar 10,92 mmHg
      portão   janela de 4 blocos (1 min) com limite de amplitude de 1e9 mmHg,
               ou seja, praticamente ABERTO: não suprime nenhum alarme
      limite_min  None (sem limite mínimo de duração do alarme)

  Estes limiares vieram da otimização CONJUNTA das duas direções, não da
  otimização de cada uma isolada: os dois detetores correm em simultâneo e o
  doente sente a UNIÃO dos alarmes. No teste (campo `meta`): sensibilidade
  0,668 (descida) e 0,657 (subida); na união, 9,5 min/h de falso alarme, ciclo
  de serviço de 27,1 % e valor preditivo positivo de 25,3 %.

  MODELO CAUSAL. O treino normaliza as âncoras pela janela de calibração congelada,
  tal como a derivação em serviço, e o modelo absoluto também é causal.

  ATENÇÃO ao campo limite_min do joblib: é lido para CFG e registado, mas NÃO
  entra na decisão — o alarme está ligado enquanto score >= limiar e o portão
  estiver aberto, sem truncatura de duração. No modelo instalado vale None. Por
  isso o ponto de operação é escolhido só entre as combinações SEM limite de
  duração.

EXTRAÇÃO E DERIVAÇÃO
  A extração (extracao_ppg) e a derivação causal por tempo exato (derivacao_ppg): 43 características, âncoras causais por sessão e
  janela de 5 min por tempo exato. O modelo é um XGBRegressor.

  Este worker NUNCA lê o campo `ecg` do buffer: continua a prever com o AD8232 desligado.

Correr:  python3 worker_tendencia_quantilica.py
"""
import json
import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import joblib
import numpy as np
import paho.mqtt.client as mqtt
import xgboost as xgb
from sqlalchemy import (create_engine, Column, Integer, String, Float, Boolean,
                        DateTime, Text, Index, text)
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

# Versão com a janela por tempo exato e as 43 características.
import comum as bp                  # buffers, caches de BD, constantes de janela
import extracao_ppg as epp     # extração de fiduciais só com PPG, com as características novas
import derivacao_ppg as dpp    # derivação causal só com PPG, por tempo exato

VARIANTE = "regressao_delta"
NOME = f"tendencia-{VARIANTE}"

DATABASE_URL = os.environ.get("DATABASE_URL", bp.DATABASE_URL)
MQTT_BROKER = os.environ.get("MQTT_BROKER", bp.MQTT_BROKER)
MQTT_PORT = int(os.environ.get("MQTT_PORT", str(bp.MQTT_PORT)))
MQTT_USER = os.environ.get("MQTT_USER", bp.MQTT_USER)
MQTT_PASS = os.environ.get("MQTT_PASS", bp.MQTT_PASS)
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", bp.AUTH_TOKEN)
THREAD_POOL_SIZE = bp.THREAD_POOL_SIZE   # o mesmo em todos os workers, definido em comum.py

MODELO_PATH = os.environ.get(
    "MODELO_TENDENCIA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelos",
                 "modelo_tendencia_ppg_quant.joblib"))      # o mesmo ficheiro que o server.py lê para os limiares

# force=True: importar comum já chamou logging.basicConfig(); sem isto
# os workers a correr em simultâneo ficariam com logs indistinguíveis.
logging.basicConfig(level=logging.INFO,
                    format=f"[{NOME}] %(asctime)s %(levelname)s %(message)s", force=True)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# MODELO E CONFIGURAÇÃO DE OPERAÇÃO
# ─────────────────────────────────────────────
# Tudo vem do joblib — limiares, portão e limites de duração incluídos. Assim o
# worker não tem constantes mágicas que possam divergir do que foi avaliado.
_b = joblib.load(MODELO_PATH)
BOOSTERS, CFG = {}, {}
for _d in ("descida", "subida"):
    _m = xgb.Booster()
    _m.load_model(bytearray(_b[_d]["booster"]))
    BOOSTERS[_d] = _m
    CFG[_d] = {k: _b[_d][k] for k in
               ("feats", "sinal", "limiar", "limite_min", "portao_blocos", "portao_limite")}
log.info("Modelo %s carregado (%d features)", _b["meta"]["variante"], len(CFG["descida"]["feats"]))
for _d in ("descida", "subida"):
    log.info("  %s: limiar %.2f mmHg | portão %d blocos <= %.2f mmHg | limite %s",
             _d, CFG[_d]["limiar"], CFG[_d]["portao_blocos"], CFG[_d]["portao_limite"],
             CFG[_d]["limite_min"])

# ─────────────────────────────────────────────
# BASE DE DADOS
# ─────────────────────────────────────────────
engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=10, pool_pre_ping=True)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))


class MedicaoTendencia(Base):
    """Mesma tabela dos outros workers de tendência; distingue-se por `variante`.

    Nota: score_descida/score_subida guardam aqui a QUEDA/SUBIDA PREVISTA EM
    mmHg, não uma probabilidade. `variante` diz qual é a escala.
    """
    __tablename__ = "medicao_tendencia"
    __table_args__ = (Index("ix_mt_pulseira_ts", "pulseira_id", "timestamp"),
                      Index("ix_mt_variante", "variante"), {'extend_existing': True})
    id = Column(Integer, primary_key=True, autoincrement=True)
    pulseira_id = Column(String, nullable=False)
    paciente_id = Column(String, nullable=False)
    variante = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    score_descida = Column(Float)
    score_subida = Column(Float)
    estavel_descida = Column(Boolean)
    estavel_subida = Column(Boolean)
    features_ml = Column(Text)


Base.metadata.create_all(engine)
# A via rápida escreve na tabela `medicao`, que pertence ao esquema do
# worker_absoluto: garante-se aqui que existe, para esta variante poder correr
# sozinha sem depender de outro processo a ter criado o esquema primeiro.
bp.Base.metadata.create_all(engine)
with engine.begin() as _c:
    for _col, _tipo in (("estavel_descida", "boolean"),
                        ("estavel_subida", "boolean")):
        _c.execute(text(f"ALTER TABLE medicao_tendencia "
                        f"ADD COLUMN IF NOT EXISTS {_col} {_tipo}"))


# ─────────────────────────────────────────────
# ESTADO POR PULSEIRA
# ─────────────────────────────────────────────
class Estado:
    def __init__(self):
        self.ppg = dpp.EstadoSessaoPPG()
        # scores ANTERIORES por direção, para o portão de estabilidade; o bloco
        # atual só entra depois de ser avaliado, para o critério ser causal
        self.scores = {d: deque(maxlen=CFG[d]["portao_blocos"]) for d in ("descida", "subida")}


_estados: dict[str, Estado] = {}
_buffers: dict[str, deque] = {}
_contadores: dict[str, int] = {}


def _get_estado(pid):
    if pid not in _estados:
        _estados[pid] = Estado()
    return _estados[pid]


def _get_buf(pid):
    if pid not in _buffers:
        _buffers[pid] = deque(maxlen=bp.BUFFER_MAX)
    return _buffers[pid]


# ─────────────────────────────────────────────
# INFERÊNCIA
# ─────────────────────────────────────────────
def prever(feats: dict) -> dict:
    """Devolve a queda/subida prevista em mmHg para cada direção."""
    saida = {}
    for d in ("descida", "subida"):
        try:
            vec = np.array([[np.nan if feats.get(k) is None else float(feats[k])
                             for k in CFG[d]["feats"]]])
            # características em falta são NaN (o modelo foi treinado assim: a variação a t-W não existe se o bloco não existe)
            bruto = float(BOOSTERS[d].predict(xgb.DMatrix(vec))[0])
            saida[d] = round(CFG[d]["sinal"] * bruto, 3)
        except Exception as e:
            log.error("Erro na previsão %s: %s", d, e)
            saida[d] = None
    return saida


def portao(estado: Estado, direcao: str, score):
    """Amplitude do score no último minuto e se a janela é de confiança.

    O limite está em mmHg (a escala deste score) e é próprio de cada direção.
    """
    hist = estado.scores[direcao]
    if score is None:
        return None, None
    n = CFG[direcao]["portao_blocos"]
    if len(hist) < n:
        amp, estavel = None, True
    else:
        amp = float(max(hist) - min(hist))
        estavel = bool(amp <= CFG[direcao]["portao_limite"])
    hist.append(float(score))
    return amp, estavel


def processar_lote(pulseira_id: str, amostras: list) -> None:
    sessao = bp.get_sessao(pulseira_id)
    if not sessao:
        return
    pac_id = sessao[0]
    if bp.nova_sessao(pulseira_id, sessao):
        # outro doente ou nova atribuição: calibração, histórico e portão de novo
        _estados.pop(pulseira_id, None)
        _buffers.pop(pulseira_id, None)
        _contadores.pop(pulseira_id, None)

    buf = _get_buf(pulseira_id)
    for a in amostras:
        buf.append({"ecg": float(a.get("ecg", 0)), "ir": float(a.get("ir", 0)),
                    "red": float(a.get("red", 0))})
    _contadores[pulseira_id] = _contadores.get(pulseira_id, 0) + len(amostras)

    # ── RÁPIDO: FC/SpO2/temperatura (~1 Hz) para o painel ────────────────
    # Esta variante e a do worker_absoluto são implantações alternativas, por
    # isso a que estiver a correr tem de alimentar o painel sozinha.
    bp.processar_via_rapida(pulseira_id, pac_id, amostras, list(buf), sessao=Session)

    fc = _contadores.get(pulseira_id, 0)
    if fc < bp.FEATURE_STRIDE or len(buf) < bp.FEATURE_WINDOW + 2 * bp.FEATURE_MARGEM:
        return
    _contadores[pulseira_id] = max(0, fc - bp.FEATURE_STRIDE)

    est = _get_estado(pulseira_id)
    pinfo = bp.get_paciente_info(pac_id)

    # Cadeia inteiramente PPG: o campo "ecg" do buffer nunca é lido, por isso
    # esta variante continua a prever com o AD8232 desligado.
    raw = epp.extrair_features_janela_ppg(
        list(buf), fs=bp.FS, feature_window=bp.FEATURE_WINDOW,
        feature_margem=bp.FEATURE_MARGEM)
    feats = dpp.derivar_features_ppg(raw, est.ppg, pinfo)
    if feats is None:
        return

    p = prever(feats)
    if p["descida"] is None and p["subida"] is None:
        return

    _amp_d, est_d = portao(est, "descida", p["descida"])
    _amp_s, est_s = portao(est, "subida", p["subida"])

    s = Session()
    try:
        s.add(MedicaoTendencia(
            pulseira_id=pulseira_id, paciente_id=pac_id, variante=VARIANTE,
            score_descida=p["descida"], score_subida=p["subida"],
            estavel_descida=est_d, estavel_subida=est_s,
            features_ml=json.dumps({k: feats.get(k) for k in CFG["descida"]["feats"]}),
        ))
        s.commit()
        alarme = []
        for d, v, ok in (("D", p["descida"], est_d), ("S", p["subida"], est_s)):
            if v is not None and v >= CFG["descida" if d == "D" else "subida"]["limiar"] and ok:
                alarme.append(d)
        log.info("%s → queda prevista %.1f mmHg | subida prevista %.1f mmHg%s",
                 pulseira_id, p["descida"] or float('nan'), p["subida"] or float('nan'),
                 f"  ALARME[{'+'.join(alarme)}]" if alarme else "")
    except Exception as e:
        s.rollback()
        log.error("DB %s: %s", pulseira_id, e)
    finally:
        Session.remove()


# ─────────────────────────────────────────────
# MQTT
# ─────────────────────────────────────────────
executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe("pulseiras/raw/#", qos=0)
        log.info("MQTT ligado (%s)", NOME)
    else:
        log.error("MQTT recusado rc=%d", rc)


def on_message(client, userdata, msg):
    try:
        pid = msg.topic.split("/")[2]
        if not bp.is_mine(pid):
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
    log.info("Worker de tendência [%s] | %d/%d | modelo=%s | %s:%d",
             VARIANTE, bp.WORKER_ID, bp.NUM_WORKERS, MODELO_PATH, MQTT_BROKER, MQTT_PORT)
    client = mqtt.Client(client_id=f"worker-{NOME}-{bp.WORKER_ID}", clean_session=True)
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.on_connect = on_connect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        log.info("A parar %s...", NOME)
    finally:
        executor.shutdown(wait=False)
        client.disconnect()
