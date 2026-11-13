"""
server.py — Flask UI com configuração de alertas e histórico
"""

import sys
import os
import subprocess
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

LISBOA = ZoneInfo("Europe/Lisbon")
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import json
import joblib
import numpy as np
from scipy.signal import butter, filtfilt
import paho.mqtt.client as mqtt

app = Flask(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://hospital:hospital_pass@localhost:5432/hospital"
)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_size": 10,
    "max_overflow": 20,
    "pool_pre_ping": True,
}

db = SQLAlchemy()

# Só os limiares do detetor de tendência, para classificar o que o worker já
# escreveu na BD — o servidor não faz inferência, não precisa dos boosters.
# Lidos do MESMO joblib do worker_tendencia_quantilica.py, de propósito:
# assim nunca há dois sítios com o mesmo número que possam divergir do que
# foi avaliado. Falha alto e cedo se o ficheiro não existir, tal como o
# próprio worker.
MODELO_TENDENCIA_PATH = os.environ.get(
    "MODELO_TENDENCIA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelos",
                 "modelo_tendencia_ppg_quant.joblib"))
_tend = joblib.load(MODELO_TENDENCIA_PATH)
LIMIAR_TENDENCIA = {d: _tend[d]["limiar"] for d in ("descida", "subida")}

# ======================= PPG AO VIVO (para o painel) =======================
# O servidor não processa sinal (isso é dos workers), mas subscreve o mesmo
# tópico MQTT em bruto só para guardar os últimos segundos de ir/red por
# pulseira em memória — o suficiente para desenhar a onda ao vivo no painel
# quando um enfermeiro abre um doente. Nunca é gravado em disco nem na BD.
# Não se importa comum.py para isto — só pouparia 4 constantes.
MQTT_BROKER_PPG = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT_PPG   = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USER_PPG   = os.environ.get("MQTT_USER", "worker")
MQTT_PASS_PPG   = os.environ.get("MQTT_PASS", "worker_pass")
AUTH_TOKEN_PPG  = os.environ.get("AUTH_TOKEN", "segredo123")

# ~8 s a 100 Hz, a mesma janela usada no worker para BPM/SpO2.
PPG_BUFFER_MAX = 800
_ppg_buffers: dict[str, deque] = {}
_ppg_lock = threading.Lock()


def _ppg_on_message(client, userdata, msg):
    try:
        partes = msg.topic.split("/")
        if len(partes) < 3:
            return
        pid = partes[2]
        payload = json.loads(msg.payload.decode())
        if payload.get("token") != AUTH_TOKEN_PPG:
            return
        amostras = payload.get("amostras", [])
        if not amostras:
            return
        with _ppg_lock:
            buf = _ppg_buffers.get(pid)
            if buf is None:
                buf = _ppg_buffers[pid] = deque(maxlen=PPG_BUFFER_MAX)
            for a in amostras:
                buf.append((a["t"], a["ir"], a["red"]))
    except Exception:
        pass   # é só a onda ao vivo; uma mensagem perdida não é grave


def _iniciar_ppg_mqtt():
    cliente = mqtt.Client(client_id=f"painel-ppg-{os.getpid()}", clean_session=True)
    cliente.username_pw_set(MQTT_USER_PPG, MQTT_PASS_PPG)
    cliente.on_message = _ppg_on_message
    cliente.on_connect = lambda c, u, f, rc: c.subscribe("pulseiras/raw/#")
    cliente.connect(MQTT_BROKER_PPG, MQTT_PORT_PPG, keepalive=30)
    cliente.loop_start()


# Cada processo gunicorn corre isto uma vez e mantém o seu próprio buffer —
# tal como as caches de comum.py, é barato manter 4 cópias independentes e
# evita partilhar estado entre processos. Não arranca durante o --setup
# (criação das tabelas), que não precisa do broker.
if not (__name__ == "__main__" and "--setup" in sys.argv):
    _iniciar_ppg_mqtt()

# ======================= MODELOS =======================
class Medicao(db.Model):
    __tablename__ = "medicao"
    id          = db.Column(db.Integer, primary_key=True)
    pulseira_id = db.Column(db.String, nullable=False)
    paciente_id = db.Column(db.String, nullable=False)
    bpm         = db.Column(db.Integer)
    spo2        = db.Column(db.Integer)
    temp        = db.Column(db.Float)
    # Frequência respiratória derivada do PPG (ver frequencia_respiratoria.py).
    # É frequentemente nula: exige 60 s de histórico e é rejeitada quando as
    # duas derivações discordam. A interface tem de tolerar a ausência.
    resp        = db.Column(db.Float)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (
        db.Index("ix_medicao_paciente_ts", "paciente_id", "timestamp"),
        db.Index("ix_medicao_pulseira_ts", "pulseira_id", "timestamp"),
    )

class Paciente(db.Model):
    __tablename__ = "paciente"
    id           = db.Column(db.String, primary_key=True)
    nome         = db.Column(db.String)
    sexo         = db.Column(db.String(1))
    idade        = db.Column(db.Integer)
    foto         = db.Column(db.Text)
    data_registo = db.Column(db.DateTime, default=datetime.utcnow)

class Pulseira(db.Model):
    __tablename__ = "pulseira"
    id              = db.Column(db.String, primary_key=True)
    paciente_id     = db.Column(db.String, db.ForeignKey("paciente.id"))

class Atribuicao(db.Model):
    __tablename__ = "atribuicao"
    id          = db.Column(db.Integer, primary_key=True)
    pulseira_id = db.Column(db.String, nullable=False)
    paciente_id = db.Column(db.String, nullable=False)
    data_inicio = db.Column(db.DateTime, default=datetime.utcnow)
    data_fim    = db.Column(db.DateTime, nullable=True)

class MedicaoTendencia(db.Model):
    """Saídas do detetor de tendência, escritas por
    worker_tendencia_quantilica.py (variante 'regressao_delta').

    O esquema tem de coincidir com o desse worker. Antes esta
    tabela não tinha nenhum leitor no servidor — os workers escreviam e
    nada lia de volta. Esta classe é o primeiro consumo.

    score_descida/score_subida guardam a saída do modelo, que depende da
    variante: a queda/subida prevista em mmHg no regressor quantílico, ou uma
    probabilidade no classificador. estavel_descida/subida é o portão de
    estabilidade: quando falso, o próprio modelo não confia no score atual.
    """
    __tablename__ = "medicao_tendencia"
    id              = db.Column(db.Integer, primary_key=True)
    pulseira_id     = db.Column(db.String, nullable=False)
    paciente_id     = db.Column(db.String, nullable=False)
    variante        = db.Column(db.String, nullable=False)
    timestamp       = db.Column(db.DateTime, default=datetime.utcnow)
    score_descida    = db.Column(db.Float)
    score_subida     = db.Column(db.Float)
    estavel_descida = db.Column(db.Boolean)
    estavel_subida  = db.Column(db.Boolean)

class Calibracao(db.Model):
    """Referência de manguito para a estimativa de PA.

    O modelo prevê o DESVIO face a esta referência (sbp = sbp_calib + delta),
    por isso sem uma linha aqui o worker não consegue prever — o prever_ta
    devolve None e a previsão fica desligada em silêncio.

    O esquema tem de coincidir com o do worker_absoluto.py, que lê esta tabela.
    """
    __tablename__ = "calibracao"
    id          = db.Column(db.Integer, primary_key=True)
    pulseira_id = db.Column(db.String, nullable=False, index=True)
    paciente_id = db.Column(db.String, nullable=False)
    sbp         = db.Column(db.Integer, nullable=False)
    dbp         = db.Column(db.Integer, nullable=False)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)

class Configuracao(db.Model):
    __tablename__ = "configuracoes"
    id          = db.Column(db.Integer, primary_key=True)
    parametro   = db.Column(db.String, unique=True, nullable=False)
    valor       = db.Column(db.String, nullable=False)
    descricao   = db.Column(db.String)

    @staticmethod
    def get_limites():
        defaults = {
            "bpm_warning_min": "60", "bpm_warning_max": "100",
            "bpm_danger_min": "50", "bpm_danger_max": "120",
            "spo2_warning_min": "96", "spo2_danger_min": "92",
            "temp_warning_min": "35.5", "temp_warning_max": "37.5",
            "temp_danger_min": "35.0", "temp_danger_max": "38.5",
            # Limiares alinhados com a escala NEWS para a frequência
            # respiratória: 12-20 normal, 9-11 e 21-24 atenção, <=8 e >=25
            # crítico.
            "resp_warning_min": "12", "resp_warning_max": "20",
            "resp_danger_min": "9", "resp_danger_max": "24"
        }
        limites = {}
        for param, valor_default in defaults.items():
            cfg = Configuracao.query.filter_by(parametro=param).first()
            if cfg:
                limites[param] = cfg.valor
            else:
                limites[param] = valor_default
                db.session.add(Configuracao(parametro=param, valor=valor_default))
        db.session.commit()
        return {k: float(v) for k, v in limites.items()}

# ======================= UTILITÁRIOS =======================
def parse_datetime_flexivel(valor):
    """Aceita 'YYYY-MM-DDTHH:MM:SS' (usado pelo modo tempo real, com segundos)
    e também 'YYYY-MM-DDTHH:MM' (usado pelos inputs datetime-local do histórico,
    que não fornecem segundos)."""
    try:
        return datetime.strptime(valor, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return datetime.strptime(valor, "%Y-%m-%dT%H:%M")

# ======================= FUNÇÃO DE CLASSIFICAÇÃO COM CONFIG =======================
def classify_vitals(bpm, spo2, temp, resp=None):
    limites = Configuracao.get_limites()
    bpm_state = None
    if bpm is not None:
        if bpm < limites["bpm_danger_min"] or bpm > limites["bpm_danger_max"]:
            bpm_state = "danger"
        elif bpm < limites["bpm_warning_min"] or bpm > limites["bpm_warning_max"]:
            bpm_state = "warning"
        else:
            bpm_state = "normal"

    spo2_state = None
    if spo2 is not None:
        if spo2 < limites["spo2_danger_min"]:
            spo2_state = "danger"
        elif spo2 < limites["spo2_warning_min"]:
            spo2_state = "warning"
        else:
            spo2_state = "normal"

    temp_state = None
    if temp is not None:
        if temp < limites["temp_danger_min"] or temp > limites["temp_danger_max"]:
            temp_state = "danger"
        elif temp < limites["temp_warning_min"] or temp > limites["temp_warning_max"]:
            temp_state = "warning"
        else:
            temp_state = "normal"

    resp_state = None
    if resp is not None:
        if resp < limites["resp_danger_min"] or resp > limites["resp_danger_max"]:
            resp_state = "danger"
        elif resp < limites["resp_warning_min"] or resp > limites["resp_warning_max"]:
            resp_state = "warning"
        else:
            resp_state = "normal"

    return {"bpm_state": bpm_state, "spo2_state": spo2_state,
            "temp_state": temp_state, "resp_state": resp_state}

def label_state(state):
    return {
        "danger": "Crítico",
        "warning": "Atenção",
        "normal": "Normal",
        None: "Sem dados"
    }.get(state, "Sem dados")

def get_overall_state(states):
    # .get() porque há chamadores antigos que constroem o dict sem resp_state.
    todos = [states["bpm_state"], states["spo2_state"], states["temp_state"],
             states.get("resp_state")]
    if any(s == "danger" for s in todos):
        return "danger"
    if any(s == "warning" for s in todos):
        return "warning"
    return "normal"

def classify_tendencia_direcao(score, estavel, limiar):
    """Estado do detetor de tendência para uma direção.

    Deliberadamente BINÁRIO (alerta/normal): o limiar foi escolhido por uma
    otimização conjunta para uma decisão de duas vias, não para um sistema
    de zonas graduadas como os sinais vitais — não existe um "atenção"
    validado para o meio, e inventar uma fração do limiar sem o medir
    repetiria o erro que se corrigiu na ablação causal (números não
    verificados). "sem_confianca" e "sem_dados" não são níveis de
    gravidade: são estados de disponibilidade da própria informação.
    """
    if score is None:
        return "sem_dados"
    if estavel is False:
        return "sem_confianca"
    if score >= limiar:
        return "alerta"
    return "normal"

def get_overall_tendencia(tendencia):
    # pior primeiro: um alerta numa direção não se esconde atrás do estado
    # da outra. "sem_confianca" vem antes de "normal" pela mesma razão do
    # portão de estabilidade: o modelo não afirmar nada não é o mesmo que
    # afirmar que está tudo bem.
    ordem = {"alerta": 0, "sem_confianca": 1, "normal": 2, "sem_dados": 3}
    todos = [tendencia["descida"], tendencia["subida"]]
    return min(todos, key=lambda s: ordem.get(s, 4))

# ======================= ROTAS =======================
@app.route("/")
def index():
    return render_template("painel.html")

@app.route("/painel")
def painel():
    return render_template("painel.html")

# --- APIs para configuração ---
@app.route("/api/configuracoes", methods=["GET"])
def get_configuracoes():
    limites = Configuracao.get_limites()
    return jsonify(limites)

@app.route("/api/configuracoes", methods=["POST"])
def set_configuracoes():
    data = request.json
    for param, valor in data.items():
        cfg = Configuracao.query.filter_by(parametro=param).first()
        if cfg:
            cfg.valor = str(valor)
        else:
            cfg = Configuracao(parametro=param, valor=str(valor))
            db.session.add(cfg)
    db.session.commit()
    return jsonify({"status": "ok"})

# --- APIs para histórico (com agrupamento por visita) ---
@app.route("/api/medicoes_completas_paciente")
def medicoes_completas_paciente():
    paciente_id = request.args.get("id")
    if not paciente_id:
        return jsonify({"error": "id obrigatório"}), 400

    atribuicoes = Atribuicao.query.filter_by(paciente_id=paciente_id).order_by(Atribuicao.data_inicio.asc()).all()
    if not atribuicoes:
        return jsonify({"visitas": [], "medicoes": []})

    visitas = []
    todas_medicoes = []
    for a in atribuicoes:
        # Medições da pulseira entre data_inicio e data_fim (ou até agora)
        query = Medicao.query.filter(Medicao.pulseira_id == a.pulseira_id, Medicao.timestamp >= a.data_inicio)
        if a.data_fim:
            query = query.filter(Medicao.timestamp <= a.data_fim)
        medicoes = query.order_by(Medicao.timestamp.asc()).all()

        visitas.append({
            "pulseira_id": a.pulseira_id,
            "data_inicio": a.data_inicio.strftime("%d/%m/%Y %H:%M"),
            "data_fim": a.data_fim.strftime("%d/%m/%Y %H:%M") if a.data_fim else "Agora",
            "ativa": a.data_fim is None
        })
        for m in medicoes:
            todas_medicoes.append({
                "pulseira_id": m.pulseira_id,
                "bpm": m.bpm,
                "spo2": m.spo2,
                "temp": m.temp,
                "resp": m.resp,
                "timestamp": m.timestamp.strftime("%d/%m/%Y %H:%M:%S") if m.timestamp else ""
            })
    return jsonify({"visitas": visitas, "medicoes": todas_medicoes})

# --- Atribuição e desatribuição ---
@app.route("/atribuir", methods=["POST"])
def atribuir_post():
    try:
        pulseira_id = request.form.get("pulseira")
        paciente_id = request.form.get("paciente")
        nome = request.form.get("nome")
        sexo = request.form.get("sexo")
        idade_str = request.form.get("idade")
        pressao_sistolica_str = request.form.get("pressao_sistolica")
        pressao_diastolica_str = request.form.get("pressao_diastolica")
        foto = request.form.get("foto")

        if not pulseira_id or not paciente_id:
            return jsonify({"status": "error", "motivo": "ID da pulseira e paciente são obrigatórios"}), 400

        atribuicao_ativa = Atribuicao.query.filter_by(paciente_id=paciente_id, data_fim=None).first()
        if atribuicao_ativa:
            return jsonify({
                "status": "error",
                "motivo": "Paciente já tem pulseira atribuída",
                "pulseira_id": atribuicao_ativa.pulseira_id
            }), 400

        pulseira = Pulseira.query.get(pulseira_id)
        if pulseira and pulseira.paciente_id and pulseira.paciente_id != paciente_id:
            return jsonify({
                "status": "error",
                "motivo": "Pulseira já está atribuída a outro paciente",
                "paciente_id": pulseira.paciente_id
            }), 400

        paciente = db.session.get(Paciente, paciente_id)
        if not paciente:
            paciente = Paciente(id=paciente_id)

        if nome and nome.strip():
            paciente.nome = nome.strip()
        if sexo and sexo in ('M', 'F'):
            paciente.sexo = sexo
        if idade_str and idade_str.strip():
            try:
                paciente.idade = int(idade_str)
            except ValueError:
                pass
        # A pressão de referência é uma medição desta visita, não um dado do doente:
        # vai direta para a calibração (mais abaixo) e não fica guardada no paciente.
        pressao_sistolica = pressao_diastolica = None
        if pressao_sistolica_str and pressao_sistolica_str.strip():
            try:
                pressao_sistolica = int(pressao_sistolica_str)
            except ValueError:
                pass
        if pressao_diastolica_str and pressao_diastolica_str.strip():
            try:
                pressao_diastolica = int(pressao_diastolica_str)
            except ValueError:
                pass
        if foto:
            try:
                paciente.foto = foto
            except ValueError:
                pass

        # Garantir que existe a pulseira
        if not pulseira:
            pulseira = Pulseira(id=pulseira_id)
            db.session.add(pulseira)
        pulseira.paciente_id = paciente_id

        # Criar nova atribuição
        nova_atribuicao = Atribuicao(
            pulseira_id=pulseira_id,
            paciente_id=paciente_id,
            data_inicio=datetime.utcnow()
        )

        db.session.add(nova_atribuicao)
        db.session.add(paciente)
        db.session.add(pulseira)

        # Referência de manguito para a estimativa de PA. Tem de ser gravada aqui:
        # o worker lê a tabela `calibracao`. Sem esta linha a previsão fica
        # desligada sem dar erro.
        # Nota clínica: o valor deve ser medido no momento em que a pulseira
        # começa a adquirir. O modelo calibra as features nos primeiros ~2,5 min,
        # e o desvio é medido a partir da pressão desse mesmo período.
        if pressao_sistolica and pressao_diastolica:
            db.session.add(Calibracao(
                pulseira_id=pulseira_id,
                paciente_id=paciente_id,
                sbp=pressao_sistolica,
                dbp=pressao_diastolica,
            ))
        else:
            app.logger.warning(
                "Pulseira %s atribuída sem tensão de referência — "
                "a estimativa de PA fica desactivada para este doente", pulseira_id)

        db.session.commit()
        return jsonify({"status": "ok"})

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Erro na atribuição: {str(e)}")
        return jsonify({"status": "error", "motivo": str(e)}), 500

@app.route("/api/desatribuir", methods=["POST"])
def api_desatribuir():
    data = request.get_json()
    paciente_id = data.get("paciente_id")
    if not paciente_id:
        return jsonify({"status": "error", "motivo": "Falta paciente_id"}), 400

    atribuicao_ativa = Atribuicao.query.filter_by(paciente_id=paciente_id, data_fim=None).first()
    if not atribuicao_ativa:
        return jsonify({"status": "error", "motivo": "Nenhuma atribuição activa encontrada"}), 404

    atribuicao_ativa.data_fim = datetime.utcnow()

    pulseira = Pulseira.query.get(atribuicao_ativa.pulseira_id)
    if pulseira:
        pulseira.paciente_id = None

    db.session.commit()
    return jsonify({"status": "ok"})

@app.route("/ler_cartao")
def ler_cartao():
    try:
        output = subprocess.check_output(["./ler_cartao"]).decode().strip()
        return jsonify(json.loads(output))
    except:
        return jsonify({"paciente_id": "", "nome": "", "sexo": "", "idade": None})

def _filtrar_cardiaco(x: np.ndarray, fs: float) -> np.ndarray:
    """Isola a banda cardíaca (0,5-8 Hz), a mesma usada em comum.py para
    BPM/SpO2. Sem isto a deriva lenta da linha de base (RIIV, a mesma que
    alimenta a FR) domina visualmente o traçado e esconde a pulsação."""
    ny = 0.5 * fs
    b, a = butter(2, [0.5 / ny, min(8.0 / ny, 0.99)], btype="band")
    return filtfilt(b, a, x)


@app.route("/api/valores_atuais")
def api_valores_atuais():
    """Último valor de CADA sinal vital, cada um com o seu próprio instante —
    não a última linha da tabela `medicao`, que tipicamente tem 3 dos 4
    campos a NULL (cada um só é escrito quando fica disponível; a FR, por
    exemplo, só atualiza a cada ~90 s). Mostrar o instante de cada valor, em
    vez de escondê-lo depois de x minutos, deixa o enfermeiro julgar por si
    a validade de um valor mais antigo."""
    paciente_id = request.args.get("id")
    if not paciente_id:
        return jsonify({"error": "Falta id"}), 400
    a = Atribuicao.query.filter_by(paciente_id=paciente_id, data_fim=None).first()
    if not a:
        return jsonify({"error": "Sem atribuição ativa"}), 404

    def ultimo(campo):
        coluna = getattr(Medicao, campo)
        row = (Medicao.query
               .filter(Medicao.paciente_id == paciente_id,
                       Medicao.timestamp >= a.data_inicio,
                       coluna.isnot(None))
               .order_by(Medicao.timestamp.desc()).first())
        return (None, None) if not row else (getattr(row, campo), row.timestamp)

    def fmt(t):
        return t.strftime("%H:%M:%S") if t else None

    bpm,  t_bpm  = ultimo("bpm")
    spo2, t_spo2 = ultimo("spo2")
    temp, t_temp = ultimo("temp")
    resp, t_resp = ultimo("resp")
    states = classify_vitals(bpm, spo2, temp, resp)

    ultima_tend = MedicaoTendencia.query.filter(
        MedicaoTendencia.paciente_id == paciente_id,
        MedicaoTendencia.variante == "regressao_delta",
        MedicaoTendencia.timestamp >= a.data_inicio
    ).order_by(MedicaoTendencia.timestamp.desc()).first()
    if ultima_tend:
        tendencia = {
            "descida": classify_tendencia_direcao(
                ultima_tend.score_descida, ultima_tend.estavel_descida,
                LIMIAR_TENDENCIA["descida"]),
            "subida": classify_tendencia_direcao(
                ultima_tend.score_subida, ultima_tend.estavel_subida,
                LIMIAR_TENDENCIA["subida"]),
        }
        t_tend = ultima_tend.timestamp
    else:
        tendencia = {"descida": "sem_dados", "subida": "sem_dados"}
        t_tend = None

    return jsonify({
        "bpm":  {"valor": bpm,  "estado": states["bpm_state"],  "timestamp": fmt(t_bpm)},
        "spo2": {"valor": spo2, "estado": states["spo2_state"], "timestamp": fmt(t_spo2)},
        "temp": {"valor": temp, "estado": states["temp_state"], "timestamp": fmt(t_temp)},
        "resp": {"valor": resp, "estado": states["resp_state"], "timestamp": fmt(t_resp)},
        "tendencia": {**tendencia, "timestamp": fmt(t_tend)},
    })


@app.route("/api/ppg_atual")
def api_ppg_atual():
    """Últimos ~8 s de ir/red de uma pulseira, filtrados na banda cardíaca e
    invertidos, para a onda ao vivo do painel. Não lê a base de dados — vem só
    do buffer em memória alimentado por _ppg_on_message (ver PPG AO VIVO no
    topo do ficheiro).

    Inversão: o MAX30102 devolve intensidade de luz RECEBIDA — mais sangue
    absorve mais luz, por isso o pico sistólico chega como MÍNIMO. Sem
    inverter, a onda mostrada ficaria de pernas para o ar face ao que um
    traçado de PPG normalmente parece (subida rápida na sístole).
    """
    pulseira_id = request.args.get("pulseira_id")
    if not pulseira_id:
        return jsonify({"error": "Falta pulseira_id"}), 400
    with _ppg_lock:
        buf = list(_ppg_buffers.get(pulseira_id, ()))
    if len(buf) < 30:
        return jsonify({"t": [], "ir": [], "red": []})

    t   = np.array([b[0] for b in buf], dtype=float)
    ir  = np.array([b[1] for b in buf], dtype=float)
    red = np.array([b[2] for b in buf], dtype=float)
    dur = (t[-1] - t[0]) / 1000.0
    fs = (len(t) - 1) / dur if dur > 0 else 100.0

    ir_f  = -_filtrar_cardiaco(ir, fs)
    red_f = -_filtrar_cardiaco(red, fs)
    t_ref = t[-1]
    return jsonify({
        "t":   [round((tt - t_ref) / 1000.0, 2) for tt in t],
        "ir":  [round(float(v), 1) for v in ir_f],
        "red": [round(float(v), 1) for v in red_f],
    })

# --- APIs para gráficos e medições (respeitam a data da pulseira activa) ---
@app.route("/api/ultimas_medicoes")
def api_ultimas_medicoes():
    paciente_id = request.args.get("id")
    if not paciente_id:
        return jsonify({"error": "Falta id"}), 400

    # Obter atribuição activa do paciente
    atribuicao_ativa = Atribuicao.query.filter_by(paciente_id=paciente_id, data_fim=None).first()
    if not atribuicao_ativa:
        return jsonify({"labels": [], "bpm": [], "spo2": [], "temp": [], "resp": []})

    data_inicio = atribuicao_ativa.data_inicio
    intervalo = request.args.get("intervalo", "24h")
    agora = datetime.utcnow()

    query = Medicao.query.filter(Medicao.paciente_id == paciente_id, Medicao.timestamp >= data_inicio)

    if intervalo == "1h":
        limite = agora - timedelta(hours=1)
        if limite < data_inicio:
            limite = data_inicio
        query = query.filter(Medicao.timestamp >= limite)
    elif intervalo == "24h":
        limite = agora - timedelta(hours=24)
        if limite < data_inicio:
            limite = data_inicio
        query = query.filter(Medicao.timestamp >= limite)
    elif intervalo == "7d":
        limite = agora - timedelta(days=7)
        if limite < data_inicio:
            limite = data_inicio
        query = query.filter(Medicao.timestamp >= limite)
    elif intervalo == "30d":
        limite = agora - timedelta(days=30)
        if limite < data_inicio:
            limite = data_inicio
        query = query.filter(Medicao.timestamp >= limite)
    elif intervalo == "custom":
        inicio_str = request.args.get("inicio")
        fim_str = request.args.get("fim")
        if inicio_str and fim_str:
            inicio = parse_datetime_flexivel(inicio_str)
            fim = parse_datetime_flexivel(fim_str)
            if inicio < data_inicio:
                inicio = data_inicio
            query = query.filter(Medicao.timestamp >= inicio, Medicao.timestamp <= fim)
        else:
            return jsonify({"error": "Datas inválidas"}), 400
    else:
        limite = agora - timedelta(hours=24)
        if limite < data_inicio:
            limite = data_inicio
        query = query.filter(Medicao.timestamp >= limite)

    medicoes = query.order_by(Medicao.timestamp.asc()).all()
    return jsonify({
        "labels": [m.timestamp.strftime("%d/%m/%Y %H:%M:%S") for m in medicoes],
        "bpm": [m.bpm for m in medicoes],
        "spo2": [m.spo2 for m in medicoes],
        "temp": [m.temp for m in medicoes],
        "resp": [m.resp for m in medicoes]
    })

@app.route("/api/medicoes_latest")
def medicoes_latest():
    paciente_id = request.args.get("id")

    # Obter atribuição activa do paciente
    atribuicao_ativa = Atribuicao.query.filter_by(paciente_id=paciente_id, data_fim=None).first()
    if not atribuicao_ativa:
        return jsonify({"labels": [], "bpm": [], "spo2": [], "temp": [], "resp": []})
    data_inicio = atribuicao_ativa.data_inicio

    m = Medicao.query.filter(
        Medicao.paciente_id == paciente_id,
        Medicao.timestamp >= data_inicio
    ).order_by(Medicao.timestamp.desc()).first()

    if not m:
        return jsonify({"ok": False})
    states = classify_vitals(m.bpm, m.spo2, m.temp, m.resp)
    return jsonify({
        "ok": True,
        "bpm": m.bpm,
        "spo2": m.spo2,
        "temp": m.temp,
        "resp": m.resp,
        **states,
        "bpm_label": label_state(states["bpm_state"]),
        "spo2_label": label_state(states["spo2_state"]),
        "temp_label": label_state(states["temp_state"]),
        "resp_label": label_state(states["resp_state"]),
        "timestamp": m.timestamp.strftime("%d/%m/%Y %H:%M") if m.timestamp else None
    })

@app.route("/api/ultimas_medicoes_visita")
def api_ultimas_medicoes_visita():
    paciente_id = request.args.get("id")
    pulseira_id = request.args.get("pulseira_id")
    data_inicio_str = request.args.get("data_inicio")
    data_fim_str = request.args.get("data_fim")

    if not paciente_id or not pulseira_id or not data_inicio_str:
        return jsonify({"error": "Parâmetros insuficientes"}), 400

    # Converter datas do formato "dd/mm/aaaa HH:MM"
    data_inicio = datetime.strptime(data_inicio_str, "%d/%m/%Y %H:%M")
    data_fim = None
    if data_fim_str and data_fim_str not in ("Agora", "desconhecido"):
        data_fim = datetime.strptime(data_fim_str, "%d/%m/%Y %H:%M")

    query = Medicao.query.filter(
        Medicao.paciente_id == paciente_id,
        Medicao.pulseira_id == pulseira_id,
        Medicao.timestamp >= data_inicio
    )
    if data_fim:
        query = query.filter(Medicao.timestamp <= data_fim)

    medicoes = query.order_by(Medicao.timestamp.asc()).all()
    return jsonify({
        "labels": [m.timestamp.strftime("%d/%m/%Y %H:%M") for m in medicoes],
        "bpm": [m.bpm for m in medicoes],
        "spo2": [m.spo2 for m in medicoes],
        "temp": [m.temp for m in medicoes],
        "resp": [m.resp for m in medicoes]
    })

@app.route("/api/tendencia_historico")
def api_tendencia_historico():
    """Histórico do detetor de tendência para o gráfico de auditoria.

    Aceita os dois padrões de parâmetros já usados para os sinais vitais,
    consoante o modal foi aberto a partir de "Ver histórico" (id + intervalo,
    igual a /api/ultimas_medicoes) ou a partir de uma visita específica
    (id + pulseira_id + data_inicio + data_fim, igual a
    /api/ultimas_medicoes_visita). Devolve também os limiares atuais (fixos
    por direção, vindos do joblib) para o front-end desenhar as linhas de
    referência.
    """
    paciente_id = request.args.get("id")
    if not paciente_id:
        return jsonify({"error": "Falta id"}), 400

    pulseira_id = request.args.get("pulseira_id")
    data_inicio_str = request.args.get("data_inicio")
    data_fim_str = request.args.get("data_fim")

    if pulseira_id and data_inicio_str:
        data_inicio = datetime.strptime(data_inicio_str, "%d/%m/%Y %H:%M")
        data_fim = None
        if data_fim_str and data_fim_str not in ("Agora", "desconhecido"):
            data_fim = datetime.strptime(data_fim_str, "%d/%m/%Y %H:%M")
        query = MedicaoTendencia.query.filter(
            MedicaoTendencia.paciente_id == paciente_id,
            MedicaoTendencia.pulseira_id == pulseira_id,
            MedicaoTendencia.variante == "regressao_delta",
            MedicaoTendencia.timestamp >= data_inicio
        )
        if data_fim:
            query = query.filter(MedicaoTendencia.timestamp <= data_fim)
    else:
        atribuicao_ativa = Atribuicao.query.filter_by(paciente_id=paciente_id, data_fim=None).first()
        if not atribuicao_ativa:
            return jsonify({"labels": [], "score_descida": [], "score_subida": [],
                             "estavel_descida": [], "estavel_subida": [],
                             "limiar_descida": LIMIAR_TENDENCIA["descida"],
                             "limiar_subida": LIMIAR_TENDENCIA["subida"]})
        data_inicio = atribuicao_ativa.data_inicio
        intervalo = request.args.get("intervalo", "24h")
        agora = datetime.utcnow()
        limites_intervalo = {"1h": timedelta(hours=1), "24h": timedelta(hours=24),
                              "7d": timedelta(days=7), "30d": timedelta(days=30)}
        query = MedicaoTendencia.query.filter(
            MedicaoTendencia.paciente_id == paciente_id,
            MedicaoTendencia.pulseira_id == atribuicao_ativa.pulseira_id,
            MedicaoTendencia.variante == "regressao_delta",
            MedicaoTendencia.timestamp >= data_inicio
        )
        if intervalo == "custom":
            inicio_str, fim_str = request.args.get("inicio"), request.args.get("fim")
            if inicio_str and fim_str:
                inicio = parse_datetime_flexivel(inicio_str)
                fim = parse_datetime_flexivel(fim_str)
                if inicio < data_inicio:
                    inicio = data_inicio
                query = query.filter(MedicaoTendencia.timestamp >= inicio, MedicaoTendencia.timestamp <= fim)
            else:
                return jsonify({"error": "Datas inválidas"}), 400
        else:
            limite = agora - limites_intervalo.get(intervalo, timedelta(hours=24))
            if limite < data_inicio:
                limite = data_inicio
            query = query.filter(MedicaoTendencia.timestamp >= limite)

    medicoes = query.order_by(MedicaoTendencia.timestamp.asc()).all()
    return jsonify({
        "labels": [m.timestamp.strftime("%d/%m/%Y %H:%M") for m in medicoes],
        "score_descida": [m.score_descida for m in medicoes],
        "score_subida": [m.score_subida for m in medicoes],
        "estavel_descida": [m.estavel_descida for m in medicoes],
        "estavel_subida": [m.estavel_subida for m in medicoes],
        "limiar_descida": LIMIAR_TENDENCIA["descida"],
        "limiar_subida": LIMIAR_TENDENCIA["subida"],
    })

@app.route("/api/dashboard_resumo")
def api_dashboard_resumo():
    # Buscar todas as atribuições activas (sem data_fim). Ordenadas por
    # data_inicio: o sort por gravidade abaixo é estável, por isso esta
    # ordem de base decide o desempate dentro do mesmo nível de gravidade
    # (mais antigo primeiro). Sem isto, a ordem dentro do mesmo nível ficava
    # ao critério do Postgres, que não garante nada sem ORDER BY.
    atribuicoes_ativas = Atribuicao.query.filter_by(data_fim=None).order_by(Atribuicao.data_inicio).all()
    pacientes_info = []
    for a in atribuicoes_ativas:
        paciente_id = a.paciente_id
        if not paciente_id:
            continue
        # Última medição apenas após o início desta atribuição
        ultima = Medicao.query.filter(
            Medicao.paciente_id == paciente_id,
            Medicao.timestamp >= a.data_inicio
        ).order_by(Medicao.timestamp.desc()).first()
        if ultima:
            states = classify_vitals(ultima.bpm, ultima.spo2, ultima.temp, ultima.resp)
            overall = get_overall_state(states)
        else:
            states = {"bpm_state": None, "spo2_state": None, "temp_state": None, "resp_state": None}
            overall = "normal"
        # Última previsão do detetor de tendência, mesmo âmbito temporal da
        # via rápida (só desde o início desta atribuição). Informativo por
        # agora: não entra em "estado" nem na ordenação da lista.
        ultima_tend = MedicaoTendencia.query.filter(
            MedicaoTendencia.paciente_id == paciente_id,
            MedicaoTendencia.variante == "regressao_delta",
            MedicaoTendencia.timestamp >= a.data_inicio
        ).order_by(MedicaoTendencia.timestamp.desc()).first()
        if ultima_tend:
            tendencia = {
                "descida": classify_tendencia_direcao(
                    ultima_tend.score_descida, ultima_tend.estavel_descida,
                    LIMIAR_TENDENCIA["descida"]),
                "subida": classify_tendencia_direcao(
                    ultima_tend.score_subida, ultima_tend.estavel_subida,
                    LIMIAR_TENDENCIA["subida"]),
            }
        else:
            tendencia = {"descida": "sem_dados", "subida": "sem_dados"}
        tendencia_geral = get_overall_tendencia(tendencia)
        if a.data_inicio:
            delta = datetime.utcnow() - a.data_inicio
            minutos = int(delta.total_seconds() / 60)
            horas = minutos // 60
            mins_rest = minutos % 60
            tempo_espera = f"{horas}h {mins_rest}m" if horas > 0 else f"{minutos}m"
        else:
            tempo_espera = "N/D"
        pacientes_info.append({
            "id": paciente_id,
            "tempo_espera": tempo_espera,
            "ultima_medicao": {
                "bpm": ultima.bpm if ultima else None,
                "spo2": ultima.spo2 if ultima else None,
                "temp": ultima.temp if ultima else None,
                "resp": ultima.resp if ultima else None,
                "timestamp": ultima.timestamp.strftime("%d/%m/%Y %H:%M") if ultima and ultima.timestamp else None
            },
            "estado": overall,
            "estado_bpm": states["bpm_state"],
            "estado_spo2": states["spo2_state"],
            "estado_temp": states["temp_state"],
            "estado_resp": states["resp_state"],
            "tendencia": tendencia_geral,
            "tendencia_descida": tendencia["descida"],
            "tendencia_subida": tendencia["subida"]
        })
    ordem_estado = {"danger": 0, "warning": 1, "normal": 2}
    pacientes_info.sort(key=lambda x: ordem_estado.get(x["estado"], 3))
    total = len(pacientes_info)
    alertas_criticos = sum(1 for p in pacientes_info if p["estado"] == "danger")
    observacoes = sum(1 for p in pacientes_info if p["estado"] == "warning")

    # Tempo de espera médio de hoje: só sessões já FECHADAS (data_fim
    # preenchida), ou seja, doentes que já foram atendidos e a quem a
    # pulseira foi retirada. Não usa os doentes ainda em espera (esses
    # entram em "pacientes_info" acima, com o tempo a subir em tempo real,
    # o que enviesaria a média para baixo enquanto a espera deles não
    # termina).
    #
    # "Hoje" é o dia civil em Lisboa, não em UTC: à meia-noite seria o
    # mesmo, mas às 23h de verão (UTC+1) o dia em Lisboa já mudou e em UTC
    # ainda não, o que empurrava sessões de "hoje" para "ontem" nessa
    # janela de uma hora. Em vez de converter cada linha (o que a base de
    # dados guarda sempre em UTC ingénuo, coerente com datetime.utcnow()
    # usado em todo o resto do modelo), calcula-se aqui a meia-noite de
    # hoje e de amanhã em Lisboa e converte-se essas duas fronteiras para
    # UTC; o zoneinfo escolhe sozinho o deslocamento certo (+00:00 ou
    # +01:00) consoante a hora do ano, por isso não há nada a corrigir
    # manualmente na mudança de hora.
    agora_lisboa = datetime.now(LISBOA)
    inicio_hoje_lisboa = agora_lisboa.replace(hour=0, minute=0, second=0, microsecond=0)
    fim_hoje_lisboa = inicio_hoje_lisboa + timedelta(days=1)
    inicio_hoje_utc = inicio_hoje_lisboa.astimezone(timezone.utc).replace(tzinfo=None)
    fim_hoje_utc = fim_hoje_lisboa.astimezone(timezone.utc).replace(tzinfo=None)
    atribuicoes_hoje = Atribuicao.query.filter(
        Atribuicao.data_fim.isnot(None),
        Atribuicao.data_fim >= inicio_hoje_utc,
        Atribuicao.data_fim < fim_hoje_utc
    ).all()
    duracoes_min = [
        (a.data_fim - a.data_inicio).total_seconds() / 60
        for a in atribuicoes_hoje if a.data_inicio
    ]
    if duracoes_min:
        media_min = int(sum(duracoes_min) / len(duracoes_min))
        horas_m, mins_m = divmod(media_min, 60)
        tempo_espera_medio_hoje = f"{horas_m}h {mins_m}m" if horas_m > 0 else f"{media_min}m"
    else:
        tempo_espera_medio_hoje = "N/D"

    return jsonify({
        "total_pacientes_ativos": total,
        "observacoes_pendentes": observacoes,
        "alertas_criticos": alertas_criticos,
        "tempo_espera_medio_hoje": tempo_espera_medio_hoje,
        "sessoes_terminadas_hoje": len(duracoes_min),
        "pacientes": pacientes_info
    })

@app.route("/api/pacientes_com_pulseira")
def api_pacientes_com_pulseira():
    atribuicoes_ativas = Atribuicao.query.filter_by(data_fim=None).all()
    pacientes = []
    for a in atribuicoes_ativas:
        paciente = db.session.get(Paciente, a.paciente_id)
        nome = paciente.nome if paciente else ""
        pacientes.append({"id": a.paciente_id, "pulseira_id": a.pulseira_id, "nome": nome})
    return jsonify({"pacientes": pacientes})

@app.route("/api/todos_pacientes")
def todos_pacientes():
    pacientes = Paciente.query.order_by(Paciente.data_registo.desc()).all()
    return jsonify({
        "pacientes": [
            {
                "id": p.id,
                "nome": p.nome or "",
                "data_registo": p.data_registo.strftime("%d/%m/%Y %H:%M") if p.data_registo else "—"
            }
            for p in pacientes
        ]
    })

@app.route("/api/dados_paciente")
def api_dados_paciente():
    paciente_id = request.args.get("id")
    if not paciente_id:
        return jsonify({"error": "Falta id"}), 400
    paciente = db.session.get(Paciente, paciente_id)
    if not paciente:
        return jsonify({"error": "Paciente não encontrado"}), 404
    return jsonify({
        "id": paciente.id,
        "nome": paciente.nome or "",
        "sexo": paciente.sexo or "",
        "idade": paciente.idade or "",
    })

@app.route("/api/foto_paciente")
def api_foto_paciente():
    paciente_id = request.args.get("id")
    if not paciente_id:
        return jsonify({"error": "Falta id"}), 400
    paciente = db.session.get(Paciente, paciente_id)
    if not paciente or not paciente.foto:
        return "", 404
    return jsonify({"foto": paciente.foto})

@app.route("/api/atribuicao_activa")
def api_atribuicao_activa():
    paciente_id = request.args.get("id")
    if not paciente_id:
        return jsonify({"error": "Falta id"}), 400
    atribuicao = Atribuicao.query.filter_by(paciente_id=paciente_id, data_fim=None).first()
    if not atribuicao:
        return jsonify({"data_inicio": None})
    return jsonify({"data_inicio": atribuicao.data_inicio.strftime("%Y-%m-%dT%H:%M")})

# ======================= INICIALIZAÇÃO =======================
db.init_app(app)

if __name__ == "__main__":
    if "--setup" in sys.argv:
        with app.app_context():
            db.create_all()
            Configuracao.get_limites()
            print("Tabelas criadas e configurações iniciais inseridas.")
    else:
        with app.app_context():
            Configuracao.get_limites()
        app.run(host="0.0.0.0", port=5000, debug=False)
