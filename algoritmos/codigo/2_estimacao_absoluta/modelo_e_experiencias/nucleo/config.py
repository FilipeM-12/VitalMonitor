"""Configuração única do absoluto: caminhos, constantes do protocolo, listas de características e variantes.

Tudo o que antes estava copiado à mão em cerca de dez scripts (a lista de características, o tamanho da janela,
o número de dobras, os caminhos) vive aqui, uma só vez. Os restantes módulos e scripts importam daqui.

Protocolo (igual para todos os resultados do absoluto):
  - validação cruzada agrupada por sujeito com N_FOLDS = 3 dobras, só no conjunto de desenvolvimento;
  - o conjunto de teste reservado (holdout) é lido uma só vez por configuração, no fim;
  - normalização CAUSAL: as âncoras morfológicas e o PAT/FC usam só os primeiros N_BLOCOS_CALIBRACAO blocos da sessão
    (janela de calibração), como na tendência (11_construir_causal_ppg_puro.py), e a suavização é só para a frente (Kalman).
"""
import os
from dataclasses import dataclass

# ── caminhos ───────────────────────────────────────────────────────────────────────────────────────
RAIZ = "/home/filipe/Desktop/Tese/algoritmo-ml"
AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))           # .../absoluto_causal
DIR_FEATURES = "/media/filipe/Dados/Features"                                   # CSV de características (MIMIC)
IDS_HOLDOUT = f"{RAIZ}/RESULTADOS_RELATORIO/codigo_dependencias/particao_sujeitos/pacientes_holdout.txt"
CSV_CLINICAS = f"{RAIZ}/06_FEATURES_DATASETS/features_clinicas_extra.csv"      # temperatura, ventilação, RASS, vasoativos
CSV_DIAGNOSTICO = f"{RAIZ}/06_FEATURES_DATASETS/flags_diagnostico.csv"         # is_cardiac, is_respiratory
DIR_CACHE = f"{AQUI}/cache"                                                     # tabelas já derivadas (pickle)
DIR_OPTUNA = f"{AQUI}/optuna"                                                   # um estudo Optuna (sqlite) por modelo e variante
DIR_RESULTADOS = f"{AQUI}/resultados"

# ── protocolo ──────────────────────────────────────────────────────────────────────────────────────
ALVO = "sbp_delta"               # o modelo prevê o desvio face à calibração com manguito, PA = sbp_calib + desvio
N_FOLDS = 3                      # dobras da validação cruzada por sujeito
JANELA_SEQ = 5                   # blocos consecutivos (5 x 15 s) que as redes veem
MIN_BLOCOS_SESSAO = 20           # sessões com menos blocos são descartadas (20 x 15 s = 5 min), como na tendência
N_BLOCOS_CALIBRACAO = 10         # janela de calibração da SESSÃO: estatísticas congeladas a partir destes blocos (como na tendência)
SEMENTE = 42
N_TRIALS = 20                    # tentativas do Optuna, igual para todas as famílias e variantes e à comparação da tendência
SUAVIZACAO = "kalman"            # kalman | ewma | none
KALMAN = dict(q_over_r=0.12, trend=False, robust_k=4.0)
TIMEOUT_TRIAL_S = 3600           # um trial que passe disto conta como falha (valor 999)
VALOR_FALHA = 999.0

# ── características ────────────────────────────────────────────────────────────────────────────────
# Ordem das características (conta para o XGBoost: a escolha de colunas por árvore depende da posição).
FEATURES_V3 = (
    'pat_diff', 'pat_mean_z', 'hr_mean_z', 'pat_velocity', 'pat_trend', 'pat_lag1', 'hr_lag1',

    'area_mean_global_z', 'w50_global_z', 'alpha_womersley_global_z', 'qtc_interval_global_z',
    'sdi_interaction_global_z', 'ppg_crest_time_global_z', 'ppg_decay_time_global_z', 'qrs_duration_global_z',
    'r_amplitude_global_z', 'qrs_amplitude_global_z', 'qrs_area_global_z',
    'age', 'gender_bin',
    'tempo_desde_calib_seg', 'pat_variance_ratio', 'pat_mean_diff_adaptativo', 'pat_mean_ewma_longo',
    'pat_slope_inicial', 'qtc_velocity', 'pat_calib_stability',
)
assert len(FEATURES_V3) == 27

# Versões incrementais do Capítulo 6 ("Conceção das Características"). A v3 é a referência.
_PPG = {'area_mean_global_z', 'w50_global_z', 'alpha_womersley_global_z', 'ppg_crest_time_global_z',
        'ppg_decay_time_global_z', 'sdi_interaction_global_z'}
_ECG = {'qtc_interval_global_z', 'qrs_duration_global_z', 'r_amplitude_global_z', 'qrs_amplitude_global_z',
        'qrs_area_global_z', 'qtc_velocity'}
FEATURES_V1 = tuple(f for f in FEATURES_V3 if f not in _PPG | _ECG)            # PAT, idade, FC e anti-deriva (15)
FEATURES_V2 = tuple(f for f in FEATURES_V3 if f not in _ECG)                   # v1 + morfologia do PPG (21)
FEATURES_V4 = FEATURES_V3 + ('is_cardiac', 'is_respiratory')                   # v3 + grupo nosológico (29)
assert (len(FEATURES_V1), len(FEATURES_V2), len(FEATURES_V4)) == (15, 21, 29)
VERSOES = {"v1": FEATURES_V1, "v2": FEATURES_V2, "v3": FEATURES_V3, "v4": FEATURES_V4}

# Âncoras morfológicas: ganham uma coluna `<nome>_global_z`, normalizada pela janela de calibração do DOENTE.
ANCORAS = ('alpha_womersley', 'qtc_interval', 'sdi_interaction', 'area_mean', 'w50', 'qrs_duration', 'r_amplitude',
           'qrs_amplitude', 'qrs_area', 'ppg_crest_time', 'ppg_decay_time', 'alpha_womersley_b')

COLUNAS_CLINICAS = ('temperatura_c', 'ventilacao_mecanica', 'rass', 'vasopressor_ativo', 'n_vasoativos_ativos')


@dataclass(frozen=True)
class Variante:
    """Uma variante de dados do absoluto. Só o que muda de variante para variante está aqui."""
    nome: str
    padrao: str                              # ficheiros em DIR_FEATURES
    features: tuple = FEATURES_V3
    ancoras_extra: tuple = ()                # colunas além de ANCORAS que também levam `_global_z`
    opcionais: tuple = ()                    # características que podem faltar (o XGBoost trata o NaN)
    fundir: str = ""                         # "" | "clinicas" | "diagnostico": CSV juntado antes de colapsar
    descricao: str = ""


_APG = ('apg_b_a', 'apg_c_a', 'apg_d_a', 'apg_aging')
_FFT = tuple(f"ecg_fft_mag_{i:02d}" for i in range(24))

# NK2: as âncoras ppg_crest_time e ppg_decay_time não existem nos CSV e dão lugar a quatro descritores da onda APG,
# nas mesmas posições da lista.
_F_NK2 = tuple(x for f in FEATURES_V3 for x in
               ((f"{a}_global_z" for a in _APG) if f == 'ppg_crest_time_global_z'
                else () if f == 'ppg_decay_time_global_z' else (f,)))

VARIANTES = {v.nome: v for v in (
    Variante("baseline", "features_*_baseline.csv", descricao="extração própria (referência)"),
    Variante("nk", "features_*_com_nk.csv", descricao="NeuroKit, filtragem agressiva"),
    Variante("nk2", "features_*_com_nk2.csv", features=_F_NK2, ancoras_extra=_APG,
             descricao="NeuroKit2 parcial (APG no lugar de crest/decay)"),
    Variante("fft", "features_*_fft.csv", features=FEATURES_V3 + tuple(f"{c}_global_z" for c in _FFT),
             ancoras_extra=_FFT, descricao="espectral (24 magnitudes FFT do ECG)"),
    Variante("clinicas", "features_*_baseline.csv", features=FEATURES_V3 + COLUNAS_CLINICAS,
             opcionais=COLUNAS_CLINICAS, fundir="clinicas", descricao="covariáveis clínicas (temperatura, ventilação, RASS, vasoativos)"),
    Variante("diagnostico", "features_*_baseline.csv", features=FEATURES_V4, fundir="diagnostico",
             descricao="grupo nosológico (v4)"),
)}
