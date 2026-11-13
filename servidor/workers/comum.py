"""
comum.py — Núcleo partilhado pelos workers de processamento
===========================================================
Aqui vive tudo o que TEM de ser igual em todas as variantes do sistema:
a configuração, o esquema da base de dados, a via rápida (FC/SpO2/temp) e,
sobretudo, a cadeia de extracção e derivação de features.

Porquê um módulo próprio, em vez de cada worker ter a sua cópia: a tese
compara as variantes entre si a custo igualado, e afirma equivalência
numérica (r = 1,00000) entre a cadeia de treino e a de produção. Ambas as
afirmações só se sustentam se as variantes partilharem literalmente o mesmo
código de extracção — com cópias, uma correcção aplicada num sítio e
esquecida noutro faria a comparação medir a diferença entre as cópias em
vez da diferença entre os modelos, sem produzir erro nenhum.

Nenhum worker depende de outro: os três importam este módulo.

Dependências: pip install numpy scipy sqlalchemy psycopg2-binary
"""

import hashlib, json, logging, math, os, threading
from collections import deque
from datetime import datetime

import numpy as np
from scipy.signal import filtfilt, find_peaks, butter
from scipy.integrate import simpson
from scipy.stats import skew

import frequencia_respiratoria as frq
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Index
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

# ─────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────

DATABASE_URL     = os.environ.get("DATABASE_URL",     "postgresql://hospital:hospital_pass@localhost:5432/hospital")
MQTT_BROKER      = os.environ.get("MQTT_BROKER",      "localhost")
MQTT_PORT        = int(os.environ.get("MQTT_PORT",    "1883"))
MQTT_USER        = os.environ.get("MQTT_USER",        "worker")
MQTT_PASS        = os.environ.get("MQTT_PASS",        "worker_pass")
AUTH_TOKEN       = os.environ.get("AUTH_TOKEN",       "segredo123")
NUM_WORKERS      = int(os.environ.get("NUM_WORKERS",  "1"))
WORKER_ID        = int(os.environ.get("WORKER_ID",    "0"))
THREAD_POOL_SIZE = int(os.environ.get("THREAD_POOL_SIZE", "8"))   # igual em todos os workers (os de tendência leem-no daqui); 4 processos x 8 já ocupam os núcleos, não foi afinado

logging.basicConfig(level=logging.INFO,
    format=f"[worker-{WORKER_ID}] %(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# BASE DE DADOS
# ─────────────────────────────────────────────

engine  = create_engine(DATABASE_URL, pool_size=10, max_overflow=20, pool_pre_ping=True)
Base    = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))


class Medicao(Base):
    __tablename__  = "medicao"
    __table_args__ = (
        Index("ix_medicao_paciente_ts", "paciente_id", "timestamp"),
        Index("ix_medicao_pulseira_ts", "pulseira_id", "timestamp"),
    )
    id          = Column(Integer, primary_key=True, autoincrement=True)
    pulseira_id = Column(String, nullable=False)
    paciente_id = Column(String, nullable=False)
    bpm         = Column(Integer)
    spo2        = Column(Integer)
    temp        = Column(Float)
    # Frequência respiratória, derivada do PPG (ver frequencia_respiratoria.py).
    # Anulável e frequentemente nula: exige 60 s de histórico e é rejeitada
    # quando as duas derivações discordam. ATENÇÃO: numa base de dados já
    # existente, create_all() não acrescenta colunas — é preciso
    #   ALTER TABLE medicao ADD COLUMN resp DOUBLE PRECISION;
    resp        = Column(Float)
    timestamp   = Column(DateTime, default=datetime.utcnow)


class MedicaoAbsoluta(Base):
    """
    Uma linha por BLOCO (média de 3 janelas de 15 s, stride 5 s): só se grava quando o bloco
    fecha, e só depois de a janela de calibração fechar (~2,5 min).
    features_ml : as características que o modelo recebeu nesse bloco.
    sbp/dbp_predicted : previsão do modelo (nulo se modelo não carregado)
    """
    __tablename__  = "medicao_absoluta"
    __table_args__ = (
        Index("ix_ma_pulseira_ts", "pulseira_id", "timestamp"),
    )
    id                = Column(Integer, primary_key=True, autoincrement=True)
    pulseira_id       = Column(String, nullable=False)
    paciente_id       = Column(String, nullable=False)
    timestamp         = Column(DateTime, default=datetime.utcnow)
    features_ml       = Column(Text)   # JSON — features prontas para o modelo
    sbp_predicted     = Column(Float)
    dbp_predicted     = Column(Float)


class Pulseira(Base):
    __tablename__    = "pulseira"
    id               = Column(String,  primary_key=True)
    paciente_id      = Column(String)


class Paciente(Base):
    __tablename__ = "paciente"
    id            = Column(String,  primary_key=True)
    data_registo  = Column(DateTime, default=datetime.utcnow)
    idade         = Column(Integer)
    sexo          = Column(String)   # 'M' ou 'F' — nome real da coluna (server.py); o
                                      # modelo aqui chamava-lhe "genero" e nunca batia
                                      # certo com a tabela, rebentando get_paciente_info


class Atribuicao(Base):
    """Espelho mínimo da tabela do server.py. Uma atribuição ativa (sem data_fim)
    é uma sessão: o estado dos workers vale só dentro dela."""
    __tablename__ = "atribuicao"
    id          = Column(Integer, primary_key=True)
    pulseira_id = Column(String, nullable=False)
    paciente_id = Column(String, nullable=False)
    data_inicio = Column(DateTime, default=datetime.utcnow)
    data_fim    = Column(DateTime, nullable=True)


class Calibracao(Base):
    __tablename__ = "calibracao"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    pulseira_id = Column(String, nullable=False, index=True)
    paciente_id = Column(String, nullable=False)
    sbp         = Column(Integer, nullable=False)
    dbp         = Column(Integer, nullable=False)
    timestamp   = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────
# CACHES DE DB (reduz SELECTs repetidos)
# ─────────────────────────────────────────────

_cache_lock       = threading.Lock()
_pulseira_cache   : dict[str, tuple | None] = {}   # pid → (paciente_id, atribuicao_id)
_pulseira_cache_ts: dict[str, datetime] = {}
PULSEIRA_CACHE_TTL_S = 60   # re-verifica a associação pulseira-doente a cada 60 s
PULSEIRA_SEM_DOENTE_TTL_S = 3   # uma pulseira ainda sem doente volta a ser verificada em 3 s, para os valores aparecerem logo a seguir à associação
_paciente_cache   : dict[str, dict] = {}   # paciente_id → {idade, genero}
_cal_cache        : dict[str, dict] = {}   # pid → {sbp, dbp}
_cal_cache_ts     : dict[str, datetime] = {}
CAL_CACHE_TTL_S   = 60   # re-verifica calibração a cada 60 s


def get_sessao(pid: str) -> tuple | None:
    """Vai ao DB (com TTL de 60s) para saber a que doente a pulseira está
    atribuída e em que atribuição. Devolve (paciente_id, atribuicao_id), ou None
    sem doente. Sem esta cache, cada mensagem MQTT (4/s por pulseira) fazia
    uma consulta nova — a centenas de pulseiras, isto satura a pool de
    ligações antes de qualquer processamento de sinal começar."""
    agora = datetime.utcnow()
    with _cache_lock:
        ts = _pulseira_cache_ts.get(pid)
        if ts:
            ttl = PULSEIRA_CACHE_TTL_S if _pulseira_cache.get(pid) else PULSEIRA_SEM_DOENTE_TTL_S
            if (agora - ts).total_seconds() < ttl:
                return _pulseira_cache.get(pid)
    s = Session()
    try:
        p = s.get(Pulseira, pid)
        pac = p.paciente_id if p else None         # sem doente associado, paciente_id é nulo
        result = None
        if pac:
            a = (s.query(Atribuicao.id)
                 .filter_by(pulseira_id=pid, paciente_id=pac, data_fim=None)
                 .order_by(Atribuicao.data_inicio.desc())
                 .first())
            result = (pac, a.id if a else None)
        with _cache_lock:
            _pulseira_cache[pid]    = result
            _pulseira_cache_ts[pid] = agora
        return result
    finally:
        Session.remove()


def get_paciente_info(paciente_id: str) -> dict:
    with _cache_lock:
        if paciente_id in _paciente_cache:
            return _paciente_cache[paciente_id]
    s = Session()
    try:
        p = s.get(Paciente, paciente_id)
        info = {"idade": p.idade if p else None, "genero": p.sexo if p else None}
        with _cache_lock:
            _paciente_cache[paciente_id] = info
        return info
    finally:
        Session.remove()


def get_calibracao(pid: str) -> dict | None:
    """Vai ao DB (com TTL de 60s) para apanhar a calibração mais recente."""
    agora = datetime.utcnow()
    with _cache_lock:
        ts = _cal_cache_ts.get(pid)
        if ts and (agora - ts).total_seconds() < CAL_CACHE_TTL_S:
            return _cal_cache.get(pid)
    s = Session()
    try:
        cal = (s.query(Calibracao)
               .filter_by(pulseira_id=pid)
               .order_by(Calibracao.timestamp.desc())
               .first())
        if cal:
            result = {"sbp": cal.sbp, "dbp": cal.dbp}
        else:
            result = None
        with _cache_lock:
            _cal_cache[pid]     = result
            _cal_cache_ts[pid]  = agora
        return result
    finally:
        Session.remove()


# ─────────────────────────────────────────────
# ESTADO POR SESSÃO E DERIVAÇÃO DAS FEATURES
# ─────────────────────────────────────────────
# Porte de carregar_e_processar_todos_deltas + injetar_ajuste_variancia_fisiologica
# (algoritmos/codigo/2_estimacao_absoluta/modulos_partilhados/normalizacao_causal/derivacao.py) para streaming.
#
# A ORDEM DAS OPERAÇÕES NO TREINO IMPORTA e não é a óbvia. Verificada no
# notebook, por posição no ficheiro:
#
#   ANTES dos filtros de outliers, sobre TODOS os blocos:
#     tempo_desde_calib_seg  (= (bloco_id - min bloco_id) * 15, COM os buracos
#                             dos blocos que vão ser descartados)
#     pat_slope_inicial, pat_calib_stability, qtc_velocity,
#     pat_mean_baseline_adaptativo, pat_mean_diff_adaptativo, pat_mean_ewma_longo
#
#   DEPOIS dos filtros, só sobre os blocos sobreviventes:
#     pat_calib, pat_mean_z, hr_mean_z, pat_diff, pat_velocity, pat_trend,
#     pat_lag1, hr_lag1, *_global_z, pat_variance_ratio
#
# Daí as duas histórias: `blocos` (todos) e `ok` (os que passam o gate do PAT).
#
# ESTRUTURA: o treino agrupa 3 janelas de extração num BLOCO e tira a média
# (bloco_id = window_idx // 3). Uma linha do modelo é um bloco de 15 s.
#
# CALIBRAÇÃO: os z-scores usam a média/desvio dos primeiros blocos, congelados.
# Não há previsão nos primeiros ~2,5 min de monitorização.
#
# ÂNCORAS CAUSAIS: normalizar os *_global_z com a estatística do # doente INTEIRO usaria informação que não existe à cabeceira. Medido em 08_custo_normalizacao_causal.py:
# passar à janela de calibração custa +0,060 mmHg no holdout, IC95%
# [-0,247, +0,385], não demonstrável. É esta a variante servida.
#
# DIFERENÇA ASSUMIDA face ao treino, inevitável em produção: o filtro
# 70 < sbp_target < 190 não é aplicável (não há alvo à cabeceira), portanto
# blocos que o treino descartaria são aqui pontuados.
#
# Verificado contra a tabela do treino nas sessões do conjunto de teste
# (Absoluto_27/19_verificar_servidor.py).

BLOCOS_CALIB      = 10   # z-score de sessão e âncoras (n_calib do treino)
BLOCOS_SLOPE      = 8    # pat_slope_inicial e pat_calib_stability
JANELAS_POR_BLOCO = 3    # colapso temporal (window_idx // 3)
BLOCO_SEG         = 15
RHO_SANGUE        = 1060
PAT_MIN, PAT_MAX  = 0.15, 0.60      # gate do treino sobre a média do bloco

ANCORAS = ['alpha_womersley', 'qtc_interval', 'sdi_interaction', 'area_mean', 'w50',
           'qrs_duration', 'r_amplitude', 'qrs_amplitude', 'qrs_area',
           'ppg_crest_time', 'ppg_decay_time']
RENOMEAR = {'crest_time': 'ppg_crest_time', 'decay_time': 'ppg_decay_time'}

# z-score de calibração como no treino: divisor = desvio + EPS_Z, e quando o desvio
# é nulo ou indefinido o treino usa a mediana desse desvio entre as sessões da
# tabela. Valores de Absoluto_27/18_desvio_de_reserva.py (sd_reserva_calibracao.json).
EPS_Z = 1e-6
SD_RESERVA = {
    'pat_mean': 0.01930401512639275,
    'hr_mean': 2.324734089466652,
    'alpha_womersley': 4.2297829899304755,
    'qtc_interval': 0.010530590360152343,
    'sdi_interaction': 0.08874190103891168,
    'area_mean': 0.6024471199170068,
    'w50': 0.00834812021422255,
    'qrs_duration': 0.0021920310216782916,
    'r_amplitude': 0.01848794201635217,
    'qrs_amplitude': 0.022406008412097143,
    'qrs_area': 0.09328904762836533,
    'ppg_crest_time': 0.012553616000633358,
    'ppg_decay_time': 0.020003352265354227,
}

# As 27 do modelo (Absoluto_27/nucleo/config.py, FEATURES_V3), pela ordem do treino.
FEATURES_ML_ORDER = [
    'pat_diff', 'pat_mean_z', 'hr_mean_z', 'pat_velocity', 'pat_trend',
    'pat_lag1', 'hr_lag1', 'area_mean_global_z', 'w50_global_z',
    'alpha_womersley_global_z', 'qtc_interval_global_z', 'sdi_interaction_global_z',
    'ppg_crest_time_global_z', 'ppg_decay_time_global_z', 'qrs_duration_global_z',
    'r_amplitude_global_z', 'qrs_amplitude_global_z', 'qrs_area_global_z',
    'age', 'gender_bin', 'tempo_desde_calib_seg', 'pat_variance_ratio',
    'pat_mean_diff_adaptativo', 'pat_mean_ewma_longo',
    'pat_slope_inicial', 'qtc_velocity', 'pat_calib_stability',
]


def _baseline_ultra_robusta(vals: list) -> float:
    """Bloco de 5 valores mais estável nos primeiros 10 — get_ultra_robust_baseline."""
    h = [v for v in vals[:10] if v is not None]
    if len(h) < 5:
        return float(np.mean(h)) if h else 0.0
    melhor = float(np.mean(h[:5]))
    cv_min = float(np.std(h[:5]) / (melhor + 1e-6))
    for i in range(len(h) - 4):
        s = h[i:i + 5]
        m = float(np.mean(s))
        cv = float(np.std(s) / (m + 1e-6))
        if cv < cv_min:
            cv_min, melhor = cv, m
    return melhor


def _mu_sd(vals: list, reserva: float) -> tuple[float | None, float]:
    """Média e divisor do z-score de calibração, com as regras de normalizar_por_calibracao do treino.

    O desvio calcula-se como no pandas (algoritmo de Welford), que dá zero exato para
    valores iguais, onde o numpy deixa um resto da ordem de 1e-17. Desvio nulo ou
    indefinido passa à reserva, e o divisor leva mais EPS_Z. Sem nenhum valor, a média
    fica None, como o NaN do treino.
    """
    v = [float(x) for x in vals if x is not None]
    if not v:
        return None, 1.0
    n, media, m2 = 0, 0.0, 0.0
    for x in v:
        n += 1
        d = x - media
        media += d / n
        m2 += d * (x - media)
    sd = (m2 / (n - 1)) ** 0.5 if n > 1 else 0.0
    return math.fsum(v) / n, (sd if sd > 0 else reserva) + EPS_Z


class SessionState:
    """Duas histórias de blocos, porque o treino calcula em dois momentos.

    Em produção uma pulseira é um doente e uma sessão, por isso a calibração de
    sessão e a de doente coincidem — exactamente a variante causal validada.
    """

    def __init__(self):
        self.idx_janela = 0
        self.bloco_atual = 0
        # votos de polaridade do ECG e de orientação do PPG (negativos, positivos) desde o início da sessão: o treino decide
        # ambas no registo inteiro, o servidor decide pela maioria das janelas vistas até agora (causal)
        self.votos_orientacao: dict[str, list[int]] = {'ecg': [0, 0], 'ppg': [0, 0]}
        self.pendentes: list[dict] = []
        self.blocos: list[dict] = []      # TODOS (features anti-drift)
        self.ok: list[dict] = []          # só os que passam o gate do PAT
        self.cal_drift: dict | None = None   # congelada a BLOCOS_SLOPE blocos
        self.cal_z: dict | None = None       # congelada a BLOCOS_CALIB blocos ok
        self.ewma: float | None = None
        # inferência: âncora de offset e estado do Kalman (nível local)
        self.preds_iniciais: list[float] = []
        self.offset: float | None = None
        self.kf_x: float | None = None
        self.kf_P: float = 1e3


def _fechar_bloco(pendentes: list[dict]) -> dict | None:
    """Média das janelas do bloco — equivalente ao groupby(bloco_id).mean()."""
    if not pendentes:
        return None
    chaves = set()
    for j in pendentes:
        chaves |= set(j)
    b = {}
    for k in chaves:
        vals = [j[k] for j in pendentes if j.get(k) is not None]
        b[k] = float(np.mean(vals)) if vals else None
    for antigo, novo in RENOMEAR.items():
        if antigo in b:
            b[novo] = b.pop(antigo)
    return b


def _bfill(vals: list) -> list:
    """Preenche None com o próximo valor válido — pandas .bfill()."""
    out = list(vals)
    prox = None
    for i in range(len(out) - 1, -1, -1):
        if out[i] is None:
            out[i] = prox
        else:
            prox = out[i]
    return out


def _qtc_velocity(hist_qtc: list) -> float:
    """_rolling_slope_5 sobre qtc.shift(1).bfill(): declive de qtc[i-5..i-2]."""
    n = len(hist_qtc)
    if n < 5:
        return 0.0
    v = _bfill(hist_qtc)
    if any(x is None for x in v):
        return 0.0
    desl = [v[0]] + v[:-1]                 # shift(1) com o 1.º repetido
    y = np.array(desl[n - 5:n - 1], dtype=float)
    if len(y) < 4:
        return 0.0
    return float(np.polyfit(np.arange(4, dtype=float), y, 1)[0]) if np.std(y) > 1e-8 else 0.0


def _anti_drift(state: SessionState, b: dict) -> None:
    """Features calculadas ANTES dos filtros, sobre todos os blocos. Escreve em b."""
    blocos = state.blocos
    pat = b.get('pat_mean')
    b['_tempo'] = (b['_bid'] - blocos[0]['_bid']) * BLOCO_SEG   # bloco_id, com buracos

    if pat is not None:
        # baseline adaptativa: mediana dos 4 blocos anteriores (shift(1).rolling(4))
        ant = [x.get('pat_mean') for x in blocos[-5:-1]]
        ant = [v for v in ant if v is not None]
        base = float(np.median(ant)) if ant else pat
        b['pat_mean_diff_adaptativo'] = pat - base
        # EWMA causal span=12, adjust=False
        alfa = 2.0 / 13
        state.ewma = pat if state.ewma is None else alfa * pat + (1 - alfa) * state.ewma
        b['pat_mean_ewma_longo'] = state.ewma
    else:
        b['pat_mean_diff_adaptativo'] = None
        b['pat_mean_ewma_longo'] = state.ewma

    b['qtc_velocity'] = _qtc_velocity([x.get('qtc_interval') for x in blocos])

    # alpha_womersley é RECALCULADO sobre as médias do bloco (secção 7 do
    # carregar_e_processar), e não é a média dos alphas por batimento que a
    # extração devolve. Tem de ser feito aqui, ao fechar o bloco: se só se
    # recalculasse no bloco corrente, a janela de calibração ficaria com a outra
    # definição e o z-score saía errado.
    if (b.get('ppg_valley_intensity') is not None
            and b.get('ppg_ac_amplitude') is not None and b.get('hr_mean') is not None):
        b['alpha_womersley'] = float(
            b['ppg_valley_intensity']
            * np.sqrt((RHO_SANGUE * (b['hr_mean'] / 60.0)) / (b['ppg_ac_amplitude'] + 1e-6)))
    else:
        b['alpha_womersley'] = None       # no treino a coluna é substituída e fica NaN

    if state.cal_drift is None and len(blocos) >= BLOCOS_SLOPE:
        h = blocos[:BLOCOS_SLOPE]
        t = np.array([x['_tempo'] for x in h if x.get('pat_mean') is not None], dtype=float)
        p = np.array([x['pat_mean'] for x in h if x.get('pat_mean') is not None], dtype=float)
        slope = (float(np.polyfit(t, p, 1)[0])
                 if len(t) >= 4 and np.std(t) > 1e-6 else 0.0)
        state.cal_drift = {
            'pat_slope_inicial': slope,
            'pat_calib_stability': float(np.std(p, ddof=1)) if len(p) > 1 else 0.0,
        }

_session_states: dict[str, SessionState] = {}
_state_lock = threading.Lock()


def _get_state(pid: str) -> SessionState:
    with _state_lock:
        if pid not in _session_states:
            _session_states[pid] = SessionState()
        return _session_states[pid]


# ─────────────────────────────────────────────
# FILTROS (idênticos aos do treino)
# ─────────────────────────────────────────────

def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    b, a = butter(order, cutoff / (0.5 * fs), btype="low", analog=False)
    return filtfilt(b, a, data)


def butter_highpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Remove baseline wander do ECG (Estratégia 3 do notebook)."""
    b, a = butter(order, cutoff / (0.5 * fs), btype="high", analog=False)
    return filtfilt(b, a, data)


# ─────────────────────────────────────────────
# EXTRACÇÃO DE FEATURES RAW (port fiel do teste.ipynb)
# ─────────────────────────────────────────────

FS             = 100     # Hz — deve coincidir com o Arduino
FEATURE_WINDOW = 1500    # 15 s a 100 Hz (igual ao WINDOW_SIZE_SEC do notebook)
FEATURE_STRIDE = 500     # 5 s a 100 Hz  (igual ao STRIDE_SEC do notebook)

# Margem de filtragem de cada lado da janela. O notebook filtra o registo INTEIRO
# e só depois corta as janelas; aqui, em streaming, filtrar exactamente os 15 s
# deixa transientes de bordo do filtfilt dentro da janela. Isso desloca o mínimo
# da janela, que é a referência de ppg_valley_intensity — e daí de alpha.
# A diferença face ao notebook, filtrando 15 s isolados ou com 5 s de margem de cada
# lado, mede-se em 20_extracao_servidor_vs_treino.py (variante «filtro»).
# Custo: as features saem com 5 s de atraso (a janela usada é a central do
# buffer, não a mais recente). Irrelevante para um horizonte de 5 minutos.
FEATURE_MARGEM = 500     # 5 s a 100 Hz

_feature_counters: dict[str, int] = {}

# Razão sinal/ruído mínima do ECG para a janela ser aproveitada.
# Equivalente adimensional ao piso absoluto de 0.08 mV do notebook de treino,
# que não transporta para as contagens do ADC. Ver _detectar_r.
ECG_SNR_MIN = 3.0


def remover_picos_isolados(sinal: np.ndarray, frac: float = 0.18) -> np.ndarray:
    """Suaviza as amostras isoladas do ECG antes do highpass (pensado para falhas de UMA amostra).

    Marca a amostra i só quando AMBOS os vizinhos divergem muito dela mas concordam ENTRE si e substitui-a pela média dos vizinhos.
    Um filtro de mediana com janela maior cortaria os próprios picos R.

    O notebook de treino usa um limiar absoluto de 0.15 mV. Aqui o sinal vem em
    contagens do ADC, por isso o limiar é relativo à amplitude do próprio sinal
    (frac x (p99-p1)); com frac = 0.18 equivale, em registos do MIMIC, aos 0.15 mV do
    notebook (comparação em 20_extracao_servidor_vs_treino.py, variante «glitch»).
    """
    s = np.asarray(sinal, dtype=float)
    if len(s) < 3:
        return s
    escala = np.percentile(s, 99) - np.percentile(s, 1)
    if not np.isfinite(escala) or escala <= 0:
        return s
    limiar = frac * escala
    d_ant, d_prox = np.abs(s[1:-1] - s[:-2]), np.abs(s[1:-1] - s[2:])
    d_viz = np.abs(s[:-2] - s[2:])
    idx = np.where((d_ant > limiar) & (d_prox > limiar) & (d_viz < limiar * 0.5))[0] + 1
    limpo = s.copy()
    limpo[idx] = (s[idx - 1] + s[idx + 1]) / 2
    return limpo


def _envelope_qrs(win_ecg: np.ndarray, fs: int) -> np.ndarray:
    """Envelope tipo Pan-Tompkins: derivada² integrada em ~80 ms.

    Realça a subida/descida rápida do QRS e suprime a onda T, que é lenta.
    Serve para desempatar candidatos a R sem usar amplitude.
    """
    deriv = np.diff(win_ecg, prepend=win_ecg[0]) * fs
    w = max(1, int(0.08 * fs))
    return np.convolve(deriv ** 2, np.ones(w) / w, mode="same")


def _detectar_r(win_ecg: np.ndarray, fs: int) -> np.ndarray | None:
    """Deteção HÍBRIDA de picos R — porte de processar_janela_teste1.

    Candidatos por AMPLITUDE (picos ≥15% do máximo da janela), desempate por
    ENVELOPE quando ficam a <0.4 s um do outro. Um detetor só por amplitude
    marca a onda T como R sempre que o R está atenuado; um detetor só por
    envelope perde R genuínos quando o máximo do envelope está inflado por uma
    onda S abrupta noutro ponto da janela.

    Devolve os índices dos R, ou None se a janela não for aproveitável.
    """
    envelope = _envelope_qrs(win_ecg, fs)

    def env_local(p):
        lo, hi = max(0, p - int(0.05 * fs)), min(len(envelope), p + int(0.05 * fs))
        return np.max(envelope[lo:hi])

    candidatos, _ = find_peaks(win_ecg, height=0.15 * np.max(win_ecg))
    r_peaks: list[int] = []
    for p in candidatos:
        if r_peaks and (p - r_peaks[-1]) < int(fs * 0.4):
            env_p, env_last = env_local(p), env_local(r_peaks[-1])
            # Só troca por envelope quando a diferença é clara (>15%): é o caso T-vs-R,
            # em que a T tem envelope muito menor por ser lenta. Com envelopes próximos
            # os dois candidatos pertencem à MESMA subida do QRS e decide a amplitude,
            # porque é o topo que se procura.
            if env_p > env_last * 1.15 or (env_p > env_last / 1.15
                                           and win_ecg[p] > win_ecg[r_peaks[-1]]):
                r_peaks[-1] = int(p)
        else:
            r_peaks.append(int(p))
    r = np.array(r_peaks, dtype=int)

    # Onda T isolada que escapou ao desempate (intervalo T-R ≥ 0.4 s) e ficou como
    # "R solteiro". Só rejeita quando amplitude E envelope estão ambos muito abaixo
    # da mediana da janela — uma T alta nunca é afiada, mas um batimento aberrante
    # genuíno mantém a amplitude, por isso uma condição só não chega.
    if len(r) >= 3:
        envs = np.array([env_local(p) for p in r])
        amps = win_ecg[r]
        med_env, med_amp = np.median(envs), np.median(amps)
        if med_env > 0 and med_amp > 0:
            r = r[~((envs < 0.15 * med_env) & (amps < 0.40 * med_amp))]

    if len(r) < 8:
        return None

    amp_r = float(np.median(win_ecg[r]))
    if amp_r <= 0:
        return None

    # ── Piso de qualidade ────────────────────────────────────────────────────
    # O notebook de treino usa um piso ABSOLUTO (0.08 mV, e 0.15 mV na segunda
    # camada) porque o ECG do MIMIC vem em mV. Aqui o ECG vem em contagens do ADC
    # de 12 bits do ESP32, com o ganho do AD8232 pelo meio — o valor absoluto não
    # transporta. O que o piso do treino mede, na prática, é se o R se distingue
    # do ruído de fundo; por isso porta-se a INTENÇÃO com uma razão sinal/ruído,
    # que é adimensional e vale em qualquer hardware.
    #
    # Ruído medido no segmento isoelétrico TP (55–75% do RR), onde não há P nem T.
    # O piso de 3.0 rejeita as janelas em que o R mal se distingue do ruído de fundo:
    # com SNR < 3 não há deteção fiducial fiável, e uma feature extraída desses
    # sinais é ruído com nome.
    rr_amostras = np.diff(r)
    segs = []
    for i in range(len(r) - 1):
        L = int(rr_amostras[i])
        if L < int(0.4 * fs):
            continue
        s = win_ecg[r[i] + int(0.55 * L): r[i] + int(0.75 * L)]
        if len(s) > 5:
            segs.append(s - np.mean(s))
    if segs:
        ruido = float(np.std(np.concatenate(segs)))
        if ruido > 0 and (amp_r / ruido) < ECG_SNR_MIN:
            return None

    # Irregularidade do RR: batimentos genuínos têm RR consistente (CV ≤ 0.10 nas
    # janelas validadas); ruído puro deu CV ≥ 0.35. Só rejeita quando a amplitude
    # também é modesta face ao ruído — amplitude claramente forte não é descartada
    # só por variabilidade de RR (arritmia é fisiológica).
    rr = rr_amostras / fs
    cv_rr = float(np.std(rr) / np.mean(rr))
    if segs and ruido > 0 and (amp_r / ruido) < 2 * ECG_SNR_MIN and cv_rr > 0.25:
        return None

    # Excursão anómala (artefacto de movimento/elétrodo: dura dezenas de ms, não é
    # um glitch de 1 amostra). Critério relativo à própria janela, porque a
    # amplitude varia com o registo e o ganho.
    if np.max(np.abs(win_ecg)) > 4 * amp_r:
        return None

    return r


def _inverter(estado, canal: str, negativo: bool) -> bool:
    """Inverte o sinal? Sem estado, se esta janela o pede. Com estado, pela maioria das janelas da sessão (a atual incluída)."""
    if estado is None:
        return negativo
    v = estado.votos_orientacao[canal]
    v[0 if negativo else 1] += 1
    return v[0] > v[1]


def processar_janela_absoluto(win_ecg: np.ndarray, win_ppg: np.ndarray, fs: int = FS, detectar_r=None) -> dict | None:
    """Features de UMA janela de 15 s JÁ filtrada e orientada (ECG com o R para cima, PPG com o pico sistólico para cima).

    É o núcleo de extrair_features_janela, sem a filtragem, a correção de polaridade e os portões do sensor, para se poder
    medir o servidor contra o treino (processar_janela_teste1 do Extração_features.ipynb) sobre os mesmos sinais.
    `detectar_r` troca o detetor de picos R (por omissão o do servidor, com o piso de relação sinal/ruído).
    """
    win_ppg_pos = win_ppg - np.min(win_ppg) + 0.01   # positividade p/ alpha
    r_peaks = (detectar_r or _detectar_r)(win_ecg, fs)
    if r_peaks is None:
        return None

    # Picos sistólicos GLOBAIS do PPG. A versão anterior procurava o 1.º pico
    # dentro de [R+0.15s, R+0.65s]: com PAT curto essa janela começa depois do
    # pico sistólico e o 1.º pico encontrado é o entalhe dicrótico.
    prom_min = (np.max(win_ppg_pos) - np.min(win_ppg_pos)) * 0.15
    ppg_peaks_all, _ = find_peaks(win_ppg_pos, distance=int(fs * 0.30),
                                  prominence=prom_min)

    rho_blood = 1060
    beats = []
    num_r = len(r_peaks)

    for i in range(num_r):
        r_now  = int(r_peaks[i])
        r_next = int(r_peaks[i + 1]) if i + 1 < num_r else None

        # ── Fiduciais do ECG ────────────────────────────────────────────────
        q_start = max(0, r_now - int(0.06 * fs))
        q_idx = int(q_start + np.argmin(win_ecg[q_start:r_now])) if q_start < r_now else r_now

        s_end = min(r_now + int(0.08 * fs), len(win_ecg))
        s_idx = int(r_now + np.argmin(win_ecg[r_now:s_end])) if r_now < s_end else r_now

        t_idx = None
        if r_next is not None:
            # margem de 0.08 s antes do R seguinte para o T não invadir o QRS
            t_start = r_now + int(0.12 * fs)
            t_end   = min(r_now + int(0.45 * fs), r_next - int(0.08 * fs))
            if t_start < t_end and t_start < len(win_ecg):
                t_idx = int(t_start + np.argmax(win_ecg[t_start:t_end]))

        qn_idx = None
        if r_next is not None:
            qn_start = max(r_now, r_next - int(0.06 * fs))
            if qn_start < r_next and qn_start < len(win_ecg):
                qn_idx = int(qn_start + np.argmin(win_ecg[qn_start:r_next]))

        # ── QRS ─────────────────────────────────────────────────────────────
        qrs_duration = qrs_amplitude = qrs_area = r_amplitude = None
        qrs_ok = False
        if q_idx < s_idx < len(win_ecg):
            qrs_duration  = round((s_idx - q_idx) / fs, 4)
            qrs_ok        = 0.04 < qrs_duration < 0.16
            r_amplitude   = round(float(win_ecg[r_now]), 4)
            qrs_amplitude = round(float(win_ecg[r_now] - min(win_ecg[q_idx], win_ecg[s_idx])), 4)
            seg = win_ecg[q_idx:s_idx + 1] - win_ecg[q_idx]
            qrs_area = round(float(simpson(y=seg)), 4) if len(seg) > 1 else 0.0

        # ── QT / TQ / QTc / SDI ─────────────────────────────────────────────
        qt = tq = qtc = sdi = None
        if t_idx is not None and q_idx < t_idx < len(win_ecg):
            qt = round((t_idx - q_idx) / fs, 4)
            if r_next is not None:
                rr_sec = (r_next - r_now) / fs
                qtc = round(qt / np.sqrt(rr_sec + 1e-6), 4)
        if t_idx is not None and qn_idx is not None and t_idx < qn_idx < len(win_ecg):
            tq = round((qn_idx - t_idx) / fs, 4)
        if qt is not None and tq is not None:
            sdi = round(min(qt / (tq + 1e-6), 10.0), 4)

        hr = round((60 * fs) / (r_next - r_now), 2) if r_next is not None else None

        # ── Pico sistólico: 1.º pico GLOBAL depois do R ─────────────────────
        lim = r_next if r_next is not None else min(r_now + int(0.9 * fs), len(win_ppg_pos))
        cand = ppg_peaks_all[(ppg_peaks_all > r_now) & (ppg_peaks_all <= lim)]
        if len(cand) == 0:
            beats.append({"pat": None, "hr": hr, "pat_kept": False, "qrs_ok": qrs_ok,
                          "qrs_duration": qrs_duration, "r_amplitude": r_amplitude,
                          "qrs_amplitude": qrs_amplitude, "qrs_area": qrs_area,
                          "qt": qt, "tq": tq, "qtc": qtc, "sdi": sdi,
                          "crest_time": None, "decay_time": None,
                          "alpha": None, "alpha_b": None, "pulse_area": None, "w50": None,
                          "ppg_valley_intensity": None, "ppg_ac_amplitude": None})
            continue
        ppg_pk = int(cand[0])
        pat = round((ppg_pk - r_now) / fs, 4)

        # ── Vale/pé do PPG: mínimo antes do pico, limitado pelo pico PPG
        # ANTERIOR e nunca pelo R. Usar r_now falha nos dois sentidos: com PAT
        # curto cai depois do pé verdadeiro; com PAT longo arrisca apanhar o vale
        # do ciclo anterior. O pé pertence à estrutura do próprio PPG.
        picos_antes = ppg_peaks_all[ppg_peaks_all < ppg_pk]
        pe_ini = max(int(picos_antes[-1]) if len(picos_antes) > 0 else 0,
                     ppg_pk - int(0.4 * fs), 0)
        ppg_vly = (pe_ini + int(np.argmin(win_ppg_pos[pe_ini:ppg_pk]))
                   if pe_ini < ppg_pk else max(0, ppg_pk - int(0.2 * fs)))

        vly_val = float(win_ppg_pos[ppg_vly])
        ppg_ac  = float(win_ppg_pos[ppg_pk] - win_ppg_pos[ppg_vly])
        omega   = (hr / 60.0) if hr is not None else 1.2
        alpha   = float(vly_val * np.sqrt((rho_blood * omega) / (ppg_ac + 1e-6)))

        upslope   = win_ppg_pos[ppg_vly:ppg_pk + 1]
        deriv_ppg = np.diff(upslope) * fs
        max_slope = float(np.max(deriv_ppg)) if len(deriv_ppg) > 0 else 0.0
        b_coef    = 1.0 / (max_slope + 1e-6)
        alpha_b   = float(vly_val * np.sqrt((omega * rho_blood) / (b_coef + 1e-6)))

        crest_time = round((ppg_pk - ppg_vly) / fs, 4)

        # ── Fim do pulso: 1.º mínimo local após o pico, limitado pelo PRÓXIMO
        # PICO PPG (não pelo R seguinte, que assumia PAT curto).
        prox = ppg_peaks_all[ppg_peaks_all > ppg_pk]
        limite = int(prox[0]) if len(prox) > 0 else min(ppg_pk + int(0.9 * fs), len(win_ppg_pos))
        pe_start = ppg_pk + int(0.04 * fs)
        pe_end   = min(ppg_pk + int(0.50 * fs), len(win_ppg_pos), limite)
        if pe_start < pe_end:
            seg_pe = win_ppg_pos[pe_start:pe_end]
            mins, _ = find_peaks(-seg_pe, distance=int(fs * 0.05))
            pe = int(pe_start + (mins[0] if len(mins) > 0 else np.argmin(seg_pe)))
            decay_time = round((pe - ppg_pk) / fs, 4)
        else:
            decay_time = None

        # ── Geometria do pulso ──────────────────────────────────────────────
        p_low  = max(0, ppg_pk - int(0.1 * fs))
        p_high = min(len(win_ppg_pos), ppg_pk + int(0.3 * fs))
        pulso  = win_ppg_pos[p_low:p_high]
        pulse_area = w50 = None
        if len(pulso) > 1 and (np.max(pulso) - np.min(pulso)) > 0.01:
            norm_p = (pulso - np.min(pulso)) / (np.max(pulso) - np.min(pulso))
            pulse_area = round(float(simpson(y=norm_p)), 4)
            idx50 = np.where(norm_p >= 0.5)[0]
            if len(idx50) > 1:
                w50 = round(float((idx50[-1] - idx50[0]) / fs), 4)

        beats.append({"pat": pat, "hr": hr, "pat_kept": False, "qrs_ok": qrs_ok,
                      "qrs_duration": qrs_duration, "r_amplitude": r_amplitude,
                      "qrs_amplitude": qrs_amplitude, "qrs_area": qrs_area,
                      "qt": qt, "tq": tq, "qtc": qtc, "sdi": sdi,
                      "crest_time": crest_time, "decay_time": decay_time,
                      "alpha": round(alpha, 4), "alpha_b": round(alpha_b, 4),
                      "pulse_area": pulse_area, "w50": w50,
                      "ppg_valley_intensity": round(vly_val, 4),
                      "ppg_ac_amplitude": round(ppg_ac, 4)})

    # ── Janela de PAT ADAPTATIVA ────────────────────────────────────────────
    # Substitui o intervalo fisiológico fixo 0.19–0.50 s. O atraso R→pico do PPG
    # inclui a latência de processamento do próprio oxímetro, consistente dentro
    # de uma sessão mas muito variável entre aparelhos. Em vez de um intervalo
    # global, aceita-se o agrupamento dominante e apertado DESTA sessão, em bins
    # de 40 ms, seja qual for o valor absoluto.
    pats_ok = np.array([b["pat"] for b in beats
                        if b["pat"] is not None and b["hr"] is not None
                        and 45 < b["hr"] < 140])
    if len(pats_ok) >= 6:
        bins = np.arange(0, pats_ok.max() + 0.04, 0.04)
        hist, edges = np.histogram(pats_ok, bins=bins)
        i_max = int(np.argmax(hist))
        cl_min, cl_max = edges[i_max] - 0.04, edges[i_max + 1] + 0.04
        for b in beats:
            if b["pat"] is not None and b["hr"] is not None and 45 < b["hr"] < 140:
                b["pat_kept"] = bool(cl_min <= b["pat"] <= cl_max)

    # ── Filtragem IQR sobre os PAT aceites ──────────────────────────────────
    pats = np.array([b["pat"] for b in beats if b["pat"] is not None and b["pat_kept"]])
    if len(pats) >= 4:
        q1, q3 = np.percentile(pats, [25, 75])
        iqr = (q3 - q1) if (q3 - q1) != 0 else 0.05
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for b in beats:
            if b["pat"] is not None and b["pat_kept"]:
                b["pat_kept"] = bool(lo <= b["pat"] <= hi)

    kept = [b for b in beats if b["pat_kept"] and b["pat"] is not None]
    if len(kept) < 6 or np.std([b["pat"] for b in kept]) >= 0.05:
        return None

    def med(lst, key):
        vals = [b[key] for b in lst if b.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    qrs_ok_beats = [b for b in kept if b["qrs_ok"]]
    return {
        # ── as 12 features que derivar_features_ml consome (nomes inalterados) ──
        "pat_mean":             med(kept, "pat"),
        "hr_mean":              med(kept, "hr"),
        "w50":                  med(kept, "w50"),
        "area_mean":            med(kept, "pulse_area"),
        "ppg_valley_intensity": med(kept, "ppg_valley_intensity"),
        "ppg_ac_amplitude":     med(kept, "ppg_ac_amplitude"),
        "alpha_womersley":      med(kept, "alpha"),
        "alpha_womersley_b":    med(kept, "alpha_b"),
        "qt_interval":          med(kept, "qt"),
        "tq_interval":          med(kept, "tq"),
        "qtc_interval":         med(kept, "qtc"),
        "sdi_interaction":      med(kept, "sdi"),
        # ── morfologia que o modelo de TENDÊNCIA usa e que não era calculada ──
        "crest_time":           med(kept, "crest_time"),
        "decay_time":           med(kept, "decay_time"),
        "qrs_duration":         med(qrs_ok_beats, "qrs_duration"),
        "r_amplitude":          med(qrs_ok_beats, "r_amplitude"),
        "qrs_amplitude":        med(qrs_ok_beats, "qrs_amplitude"),
        "qrs_area":             med(qrs_ok_beats, "qrs_area"),
    }


def extrair_features_janela(snap: list, fs: int = FS, estado: 'SessionState | None' = None) -> dict | None:
    """Extração de features de UMA janela de 15 s.

    Porte de processar_janela_teste1 do Extração_features.ipynb (variante
    baseline), que é o pipeline com que o modelo foi treinado. Tem de o
    reproduzir, senão o modelo recebe em produção features de uma distribuição
    diferente da do treino. Pontos que contam:

      1. picos R por amplitude+envelope (não find_peaks com percentil 75)
      2. remoção da onda T isolada marcada como R
      3. pisos de qualidade da janela (SNR, CV do RR, excursão anómala)
      4. pico sistólico do PPG por deteção GLOBAL (não o 1.º pico dentro de
         [R+0.15s, R+0.65s], que marcava o entalhe dicrótico quando o PAT era
         curto)
      5. janela de PAT ADAPTATIVA por sessão (não o intervalo fixo 0.19–0.50 s)

    Entrada : lista de dicts {ecg, ir, red}, com pelo menos FEATURE_WINDOW elementos. Com `estado` (o da sessão), a polaridade
              do ECG e a orientação do PPG decidem-se pela maioria das janelas da sessão até agora, como o treino faz sobre o
              registo inteiro; sem ele, janela a janela (ensaios e testes).
    Saída   : dict de features, ou None se a janela for inválida.
    Usa o canal IR como PPG (equivalente ao PLETH do MIMIC).
    """
    if len(snap) < FEATURE_WINDOW:
        return None

    # Filtra o buffer TODO e usa a janela central: as margens absorvem os
    # transientes de bordo do filtfilt, aproximando-se de filtrar o registo
    # inteiro como o notebook faz. Se o buffer não tiver margem (testes), usa a
    # janela mais recente e filtra só isso.
    margem = (len(snap) - FEATURE_WINDOW) // 2
    if margem >= 1:
        bloco = snap
    else:
        bloco, margem = snap[-FEATURE_WINDOW:], 0

    ecg_raw = np.array([s["ecg"] for s in bloco], dtype=float)
    ir_raw  = np.array([s["ir"]  for s in bloco], dtype=float)

    # Sensor provavelmente não está no dedo: a intensidade do IR é muito menor sem
    # dedo do que com o dedo bem colocado, e 10 000 fica entre as duas.
    if np.mean(ir_raw) < 10_000:
        return None

    # Suaviza as amostras isoladas ANTES do highpass, senão o filtro espalha o
    # glitch por várias amostras e cria um falso pico afiado.
    ecg_raw = remover_picos_isolados(ecg_raw)

    win_ppg = butter_lowpass_filter(ir_raw, 8.0, fs)
    win_ecg = butter_highpass_filter(ecg_raw, 0.5, fs)

    if margem:
        win_ppg = win_ppg[margem:margem + FEATURE_WINDOW]
        win_ecg = win_ecg[margem:margem + FEATURE_WINDOW]

    # Correção de inversão: com os elétrodos trocados o R aponta para baixo e
    # toda a deteção falha em silêncio. A assimetria do ECG é positiva quando o R
    # está para cima, por construção.
    if _inverter(estado, 'ecg', skew(win_ecg) < 0):
        win_ecg = -win_ecg

    # Correção de inversão do PPG. O MAX30102 devolve intensidade de luz
    # RECEBIDA: mais sangue -> mais absorção -> menos luz, portanto o pico
    # sistólico sai como MÍNIMO. O PLETH do MIMIC, com que o modelo foi
    # treinado, tem a convenção oposta. Sem isto toda a morfologia de PPG entra
    # espelhada (crest_time, decay_time, area_mean, w50, alpha_womersley): a
    # subida sistólica sai mais lenta do que a descida, o que é fisiologicamente
    # impossível.
    # Critério: a subida sistólica é muito mais abrupta que a descida
    # diastólica, por isso a derivada tem cauda positiva e skew(diff) > 0 num
    # sinal bem orientado; sem correção, o MAX30102 dá skew(diff) negativo.
    # É um no-op para dados do MIMIC, por isso não invalida o treino.
    if _inverter(estado, 'ppg', skew(np.diff(win_ppg)) < 0):
        win_ppg = -win_ppg

    win_ppg_pos = win_ppg - np.min(win_ppg) + 0.01   # positividade p/ alpha
    if np.std(win_ppg_pos) < 10:                     # sinal plano (sem dedo)
        return None

    return processar_janela_absoluto(win_ecg, win_ppg, fs)


# ─────────────────────────────────────────────
# FEATURES DERIVADAS (equivalente a carregar_e_processar_todos_deltas)
# ─────────────────────────────────────────────

def derivar_features_ml(raw: dict | None, state: SessionState,
                        paciente_info: dict) -> dict | None:
    """Acumula uma janela; devolve as 27 features quando um bloco fecha e é válido.

    `raw` pode ser None (janela rejeitada pela extração): a chamada conta para o
    alinhamento dos blocos, que é como o treino numera as janelas.
    """
    state.idx_janela += 1
    bid = (state.idx_janela - 1) // JANELAS_POR_BLOCO
    fechou = None

    if bid != state.bloco_atual:
        b = _fechar_bloco(state.pendentes)
        state.pendentes = []
        if b is not None:
            b['_bid'] = state.bloco_atual
            state.blocos.append(b)
            _anti_drift(state, b)
            if b.get('pat_mean') is not None and PAT_MIN < b['pat_mean'] < PAT_MAX:
                state.ok.append(b)
                fechou = b
        state.bloco_atual = bid

    if raw is not None and raw.get('pat_mean') is not None:
        state.pendentes.append(raw)

    if fechou is None:
        return None                       # nenhum bloco válido fechou nesta chamada
    if state.cal_drift is None:
        return None                       # janela de drift ainda a encher
    if len(state.ok) < BLOCOS_CALIB:
        return None                       # janela de calibração ainda a encher

    ok = state.ok
    if state.cal_z is None:
        h = ok[:BLOCOS_CALIB]
        cal = {'pat_calib': _baseline_ultra_robusta([x.get('pat_mean') for x in h])}
        for nome in ('pat_mean', 'hr_mean', *ANCORAS):
            cal[f'mu_{nome}'], cal[f'sd_{nome}'] = _mu_sd([x.get(nome) for x in h], SD_RESERVA[nome])
        state.cal_z = cal
    cal = state.cal_z
    if cal['mu_pat_mean'] is None or cal['mu_hr_mean'] is None:
        return None

    b = fechou
    i = len(ok) - 1
    if i < 3:
        return None                       # pat_trend precisa de 3 blocos anteriores
    pat, hr = b.get('pat_mean'), b.get('hr_mean')
    if pat is None or hr is None:
        return None

    hp = [x.get('pat_mean') for x in ok]
    hh = [x.get('hr_mean') for x in ok]
    if any(v is None for v in hp[-4:]) or hh[-2] is None:
        return None

    pat_z = (pat - cal['mu_pat_mean']) / cal['sd_pat_mean']
    hr_z = (hr - cal['mu_hr_mean']) / cal['sd_hr_mean']

    anc = {}
    for a in ANCORAS:
        v = b.get(a)
        anc[f'{a}_global_z'] = (None if v is None or cal[f'mu_{a}'] is None
                                else (float(v) - cal[f'mu_{a}']) / cal[f'sd_{a}'])
    if any(v is None for v in anc.values()):
        return None

    pat_velocity = pat - hp[-2]
    pat_trend = pat - hp[-4]
    pat_diff = pat - cal['pat_calib']

    # pat_variance_ratio: expanding std do pat_diff / std dos 5 primeiros. O treino
    # calcula-o no fim, só sobre as linhas que ficam na tabela. Não entram os 3
    # primeiros blocos válidos (sem pat_trend) nem os blocos a que falta uma âncora.
    hd = [x['pat_mean'] - cal['pat_calib'] for x in ok[3:]
          if all(x.get(a) is not None for a in ANCORAS)]
    if len(hd) >= 5:
        ini = float(np.std(hd[:5], ddof=1))
        if not np.isfinite(ini) or ini < 1e-5:
            ini = 1.0
        pat_variance_ratio = float(np.std(hd, ddof=1)) / ini
    else:
        pat_variance_ratio = 1e-5

    idade = paciente_info.get('idade') or 0
    genero = paciente_info.get('genero') or 'M'

    return {
        'pat_diff': pat_diff,
        'pat_mean_z': pat_z,
        'hr_mean_z': hr_z,
        'pat_velocity': pat_velocity,
        'pat_trend': pat_trend,
        'pat_lag1': (hp[-2] - cal['mu_pat_mean']) / cal['sd_pat_mean'],
        'hr_lag1': (hh[-2] - cal['mu_hr_mean']) / cal['sd_hr_mean'],
        **anc,
        'age': idade,
        'gender_bin': 0 if genero == 'M' else 1,      # o treino faz {'M': 0, 'F': 1}
        'tempo_desde_calib_seg': b['_tempo'],
        'pat_variance_ratio': pat_variance_ratio,
        'pat_mean_diff_adaptativo': b['pat_mean_diff_adaptativo'],
        'pat_mean_ewma_longo': b['pat_mean_ewma_longo'],
        'pat_slope_inicial': state.cal_drift['pat_slope_inicial'],
        'qtc_velocity': b['qtc_velocity'],
        'pat_calib_stability': state.cal_drift['pat_calib_stability'],
        '_bid': b['_bid'],                            # para rastreio/testes
    }


# ─────────────────────────────────────────────
# BUFFER DESLIZANTE + PROCESSAMENTO BPM/SpO2
# ─────────────────────────────────────────────

BUFFER_MAX        = FEATURE_WINDOW + 2 * FEATURE_MARGEM   # 2500 amostras (25 s)
FILTER_W          = 5
MIN_AMOSTRAS      = 800   # 8 s a 100 Hz — alinhado com o tempo de médio típico dos oxímetros clínicos (4-8 s)
MEDICAO_INTERVALO = 1.0   # s

# ── Frequência respiratória ──
# A FR precisa de ~60 s de histórico, e o buffer de amostras só tem 25 s. Em vez
# de o alargar (o que multiplicaria a memória por doente), guarda-se a LINHA DE
# BASE decimada: um ponto por lote MQTT, ~4 Hz. 450 pontos cobrem ~108 s, com
# margem sobre a janela de 90 s que a estimativa usa.
LINHA_BASE_MAX = 450
_linha_base: dict[str, deque] = {}
_fr_ultimo_estado: dict[str, str] = {}
_fr_lock = threading.Lock()
# FC da via rápida (instante do dispositivo, bpm), para a FR tirar do espectro a
# zona do pulso (frequencia_respiratoria.MARGEM_FC). Uma por segundo; 150 cobrem
# a janela de 90 s com folga.
FC_RECENTE_MAX = 150
_fc_recente: dict[str, deque] = {}

# ── Cadência real de amostragem ──
# O firmware pede 100 Hz ao MAX30102 mas o oscilador do sensor desvia: a cadência real
# fica alguns por cento acima. Assumir 100 enviesa a FC na mesma proporção.
# Mede-se a cadência de cada pulseira a partir dos carimbos do próprio
# dispositivo, dentro de cada lote — entre lotes há mensagens perdidas (QoS 0)
# que falseariam a contagem.
_fs_estado: dict[str, dict] = {}
_fs_lock = threading.Lock()
FS_LOTES_MIN = 20          # lotes a acumular antes de confiar na medição


def fs_medida(pulseira_id: str, amostras: list) -> float:
    """Cadência real da pulseira, em Hz. Devolve FS enquanto não for fiável."""
    if len(amostras) < 5:
        return float(FS)
    try:
        dt = (float(amostras[-1]["t"]) - float(amostras[0]["t"])) / 1000.0
    except (KeyError, TypeError, ValueError):
        return float(FS)
    if dt <= 0:
        return float(FS)
    fs_lote = (len(amostras) - 1) / dt
    if not 50.0 <= fs_lote <= 200.0:
        return float(FS)
    with _fs_lock:
        est = _fs_estado.get(pulseira_id)
        if est is None:
            est = _fs_estado[pulseira_id] = {"soma": 0.0, "n": 0}
        est["soma"] += fs_lote
        est["n"] += 1
        return float(FS) if est["n"] < FS_LOTES_MIN else est["soma"] / est["n"]

_buffers   : dict[str, deque] = {}
_buf_lock  = threading.Lock()
_last_save : dict[str, datetime] = {}
_save_lock = threading.Lock()


def _get_buf(pid: str) -> deque:
    with _buf_lock:
        if pid not in _buffers:
            _buffers[pid] = deque(maxlen=BUFFER_MAX)
        return _buffers[pid]


def _ma_filter(arr: np.ndarray) -> np.ndarray:
    """Moving average zero-phase (rápido, para BPM/SpO2)."""
    k = np.ones(FILTER_W) / FILTER_W
    return filtfilt(k, [1.0], arr)


def _bpm(ir_f: np.ndarray, fs: float) -> int | None:
    # A proeminência baseada no desvio-padrão deixava passar o entalhe
    # dicrótico como um batimento à parte, dobrando o BPM (visto em bancada: o BPM
    # saltava para cerca do dobro com SpO2 e temperatura estáveis, logo sem perda
    # de contacto). Distância de 0,30 s + proeminência de 15% da amplitude da
    # janela é a combinação já validada no treino para o mesmo
    # problema.
    amplitude = float(np.max(ir_f) - np.min(ir_f))
    if amplitude <= 0:
        return None
    dist     = int(fs * 0.30)
    prom     = amplitude * 0.15
    peaks, _ = find_peaks(ir_f, distance=dist, prominence=prom)
    if len(peaks) < 2:
        return None
    bpm = 60.0 / (np.mean(np.diff(peaks)) / fs)
    return int(round(bpm)) if 40 < bpm < 180 else None


def _spo2(ir_f: np.ndarray, red_f: np.ndarray) -> int | None:
    dc_ir, dc_red = np.mean(ir_f), np.mean(red_f)
    if dc_ir < 10_000 or dc_red < 10_000:
        return None
    ac_ir, ac_red = np.std(ir_f), np.std(red_f)
    if ac_ir < 10:
        return None
    R    = (ac_red / dc_red) / (ac_ir / dc_ir)
    spo2 = -45.060 * R**2 + 30.354 * R + 94.845
    return int(round(spo2)) if 80 <= spo2 <= 100 else None


# ─────────────────────────────────────────────
# VIA RÁPIDA (partilhada por todos os workers)
# ─────────────────────────────────────────────

def acumular_linha_de_base(pulseira_id: str, amostras: list) -> None:
    """Acrescenta um ponto da linha de base do PPG, um por lote recebido.

    Tem de correr a CADA lote, e não só quando o limitador de cadência das
    medições deixa passar: a ~4 Hz a série cobre até 54 resp/min, mas a 1 Hz
    ficaria limitada a 30 e perdia precisamente a taquipneia.

    Nunca levanta exceção: a FR é um extra do painel e não pode derrubar a via
    rápida, que alimenta a FC, a SpO2 e a temperatura.
    """
    try:
        t_ms = np.array([a["t"] for a in amostras], dtype=float)
        ir = np.array([a["ir"] for a in amostras], dtype=float)
        if len(ir) == 0:
            return
        ponto = frq.ponto_de_linha_de_base(t_ms, ir)
        with _fr_lock:
            serie = _linha_base.get(pulseira_id)
            if serie is None:
                serie = _linha_base[pulseira_id] = deque(maxlen=LINHA_BASE_MAX)
            # O millis() da pulseira reinicia num reboot, e aí a série antiga
            # deixa de fazer sentido. Mas os lotes são processados por um pool
            # de threads e chegam fora de ordem, por isso só um salto GRANDE
            # para trás é reboot — comparar com o ponto anterior apagaria a
            # série a toda a hora e a FR nunca chegaria aos 60 s mínimos.
            if serie and ponto[0] < serie[-1][0] - frq.JANELA_MAX_S:
                serie.clear()
                _fc_recente.pop(pulseira_id, None)
            serie.append(ponto)
    except Exception as e:
        log.warning("linha de base %s: %r", pulseira_id, e)


def _estimar_fr(pulseira_id: str) -> float | None:
    """FR a partir da linha de base acumulada, ou None se não for fiável."""
    try:
        with _fr_lock:
            serie = _linha_base.get(pulseira_id)
            # Ordenar: o pool de threads insere os lotes fora de ordem, e o
            # espectro precisa de uma base de tempo crescente.
            janela = sorted(serie, key=lambda p: p[0]) if serie else []
        if len(janela) < 16:
            return None
        # Descarta o que exceda a janela máxima, para a estimativa não arrastar
        # respiração de há três minutos. Guarda-se uma amostra a mais do que a
        # janela, senão a cobertura fica sempre uns décimos abaixo de
        # JANELA_MAX_S e um mínimo igual à janela nunca seria atingido.
        corte = janela[-1][0] - frq.JANELA_MAX_S - 1.0 / frq.FS_SERIE
        janela = [p for p in janela if p[0] >= corte]
        if len(janela) < 16:
            return None
        t = np.array([p[0] for p in janela], dtype=float)
        v = np.array([p[1] for p in janela], dtype=float)
        with _fr_lock:
            fcs = [b for (s, b) in _fc_recente.get(pulseira_id, ()) if s >= t[0]]
        r = frq.estimar(t, v, fcs)
        # Só se regista a MUDANÇA de estado: a estimativa corre uma vez por
        # segundo e por pulseira, e registar cada rejeição encheria o journal.
        # O motivo traz a percentagem, que muda a cada segundo; só interessa o
        # tipo de motivo para decidir se houve mudança de estado.
        estado = r["motivo"].split("(")[0].strip() or "ok"
        if _fr_ultimo_estado.get(pulseira_id) != estado:
            _fr_ultimo_estado[pulseira_id] = estado
            if r["fr"] is None:
                log.info("FR %s sem estimativa: %s (pontos=%d, cobertura=%.0fs)",
                         pulseira_id, r["motivo"], len(janela), t[-1] - t[0])
            else:
                log.info("FR %s a estimar de novo: %.1f resp/min (qualidade %.2f)",
                         pulseira_id, r["fr"], r["qualidade"])
        return r["fr"]
    except Exception as e:                       # nunca derrubar a via rápida
        log.warning("FR %s: %r", pulseira_id, e)
        return None


# ── Dump opcional do sinal em bruto ──
# Desligado por omissão: gravar 104 Hz de 200 pulseiras não se justifica em
# produção. Serve para diagnosticar um episódio específico (ex.: a rajada de
# BPM ou o reboot de hoje, dos quais só ficaram os resumos já processados).
# Ligar só durante o teste, para uma ou poucas pulseiras:
#   DUMP_BRUTO_DIR=/home/filipe/dump_ppg
# nas três unidades pulseiras-worker@{0,1,2,3} e reiniciar. Cada lote fica tal
# como chegou por MQTT, um por linha, em <DUMP_BRUTO_DIR>/<pulseira_id>.jsonl
# — mesmo formato que os ficheiros de captura usados hoje, por isso os scripts
# de análise já escritos servem sem alteração. Ficheiro de acrescento simples,
# sem rotação nem limite de tamanho: é para sessões curtas de diagnóstico, não
# para produção contínua — desligar a variável e apagar o ficheiro depois de
# recolher o que interessa.
DUMP_BRUTO_DIR = os.environ.get("DUMP_BRUTO_DIR", "").strip()
_dump_lock = threading.Lock()


def _dump_bruto(pulseira_id: str, amostras: list) -> None:
    if not DUMP_BRUTO_DIR:
        return
    try:
        os.makedirs(DUMP_BRUTO_DIR, exist_ok=True)
        caminho = os.path.join(DUMP_BRUTO_DIR, f"{pulseira_id}.jsonl")
        linha = json.dumps({"pulseira_id": pulseira_id, "amostras": amostras})
        with _dump_lock, open(caminho, "a") as f:
            f.write(linha + "\n")
    except Exception as e:
        log.warning("dump bruto %s: %r", pulseira_id, e)


def processar_via_rapida(pulseira_id: str, pac_id: str, amostras: list,
                         snap: list, sessao=None) -> None:
    """FC, SpO2 e temperatura a ~1 Hz, para o painel de monitorização.

    Só lê os canais ir/red e a temperatura, nunca o ecg: funciona na mesma com
    os elétrodos desligados. Está aqui em função própria, e não embutida no
    processar_lote, porque as variantes do sistema (PPG+ECG e só-PPG) são
    implantações ALTERNATIVAS: corre-se uma ou outra, e a que estiver a correr
    tem de alimentar o painel sozinha. Sem isto, uma implantação só-PPG deixava
    o painel sem sinais vitais nenhuns.

    `sessao` permite ao chamador usar a sua própria ligação à base de dados;
    por omissão usa a deste módulo.
    """
    _dump_bruto(pulseira_id, amostras)

    S = sessao if sessao is not None else Session

    # A linha de base da FR e a medição da cadência têm de crescer a CADA lote,
    # antes do limitador de cadência abaixo: a estimativa corre uma vez por
    # segundo, mas a série de que ela depende tem de ser amostrada a ~4 Hz.
    acumular_linha_de_base(pulseira_id, amostras)
    fs = fs_medida(pulseira_id, amostras)

    temps = [float(a["temp"]) for a in amostras
             if a.get("temp") is not None and float(a["temp"]) > -100]
    temp_media = float(np.mean(temps)) if temps else None

    # O lock só protege o limitador de cadência. Libertá-lo antes do cálculo e
    # da escrita evita serializar todas as pulseiras atrás de um lock global.
    agora = datetime.utcnow()
    with _save_lock:
        ultimo = _last_save.get(pulseira_id)
        if ultimo is not None and (agora - ultimo).total_seconds() < MEDICAO_INTERVALO:
            return
        _last_save[pulseira_id] = agora

    # _bpm() não verifica o nível DC do ir, ao contrário do _spo2(): sem dedo,
    # o find_peaks encontra "picos" no próprio ruído e devolve um BPM plausível
    # (40-180) sobre um sinal que não é PPG nenhum. Aplicar aqui o mesmo
    # limiar de presença de dedo usado em extrair_features_janela/_spo2.
    if len(snap) >= MIN_AMOSTRAS and np.mean([s["ir"] for s in snap[-MIN_AMOSTRAS:]]) >= 10_000:
        ir_arr  = np.array([s["ir"]  for s in snap[-MIN_AMOSTRAS:]], dtype=float)
        red_arr = np.array([s["red"] for s in snap[-MIN_AMOSTRAS:]], dtype=float)
        bpm_val  = _bpm(_ma_filter(ir_arr), fs)
        spo2_val = _spo2(_ma_filter(ir_arr), _ma_filter(red_arr))
    else:
        bpm_val = spo2_val = None

    # a FR precisa da FC da mesma janela para tirar o pulso do espectro; o
    # instante é o do relógio do dispositivo, como o da linha de base
    t_lote = [float(a["t"]) for a in amostras if a.get("t") is not None]
    if bpm_val and t_lote:
        with _fr_lock:
            _fc_recente.setdefault(pulseira_id, deque(maxlen=FC_RECENTE_MAX)).append(
                (max(t_lote) / 1000.0, bpm_val))

    resp_val = _estimar_fr(pulseira_id)

    if not (bpm_val or spo2_val or temp_media or resp_val):
        return

    s = S()
    try:
        s.add(Medicao(pulseira_id=pulseira_id, paciente_id=pac_id,
                      bpm=bpm_val, spo2=spo2_val, resp=resp_val,
                      temp=round(temp_media, 2) if temp_media else None))
        s.commit()
    except Exception as e:
        s.rollback(); log.error("DB Medicao %s: %s", pulseira_id, e)
    finally:
        S.remove()


# ─────────────────────────────────────────────
# ESTADO POR DOENTE E POR SESSÃO
# ─────────────────────────────────────────────
# Todo o estado acima está indexado pela pulseira, mas só vale para um doente
# numa sessão: a calibração congelada nos primeiros blocos, o histórico dos
# declives, o buffer de amostras e a linha de base da FR. Se a pulseira passar
# para outro doente (ou o mesmo doente tiver uma nova atribuição), o estado
# antigo normalizaria o novo doente com a calibração do anterior. A cadência
# medida (_fs_estado) fica, porque é do oscilador da pulseira e não do doente.
_sessao_ativa: dict[str, tuple] = {}
_sessao_lock = threading.Lock()


def nova_sessao(pid: str, sessao: tuple) -> bool:
    """Regista a sessão (paciente_id, atribuicao_id) em que chegou este lote.
    Se mudou, apaga o estado da pulseira e devolve True (uma só vez por mudança,
    mesmo com vários lotes da mesma pulseira em threads diferentes), para o
    worker apagar também o estado que é só dele."""
    with _sessao_lock:
        anterior = _sessao_ativa.get(pid)
        if anterior == sessao:
            return False
        _sessao_ativa[pid] = sessao
    with _state_lock:
        _session_states.pop(pid, None)
    with _buf_lock:
        _buffers.pop(pid, None)
        _feature_counters.pop(pid, None)
    with _fr_lock:
        _linha_base.pop(pid, None)
        _fr_ultimo_estado.pop(pid, None)
        _fc_recente.pop(pid, None)
    with _save_lock:
        _last_save.pop(pid, None)
    with _cache_lock:
        _cal_cache.pop(pid, None)          # a calibração do novo doente é gravada no /atribuir
        _cal_cache_ts.pop(pid, None)
    if anterior is not None:
        log.info("%s: nova sessão %s (antes %s), estado reiniciado", pid, sessao, anterior)
    return True


# ─────────────────────────────────────────────
# REPARTIÇÃO HORIZONTAL ENTRE INSTÂNCIAS
# ─────────────────────────────────────────────

def is_mine(pid: str) -> bool:
    if NUM_WORKERS == 1:
        return True
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % NUM_WORKERS == WORKER_ID
