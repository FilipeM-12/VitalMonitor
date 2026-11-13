"""derivacao_ppg.py — derivação causal por bloco, só com PPG, com a janela de 5 min ancorada no tempo e as 43 características do modelo.

Porte para streaming da tabela de treino da tendência (3_tendencia_modelo_final/tendencia_tempo_fixo/construir_tabela_ppg_morf.py:
colapso, z-scores e linhas da tabela).

Não se pode reaproveitar `derivar_features_ml` do worker_absoluto.py: essa função está
construída à volta do PAT e devolve None sem ele.

ORDEM DAS OPERAÇÕES — tem de bater certo com o treino:
  1. 3 janelas (passo de 5 s) colapsam num BLOCO de 15 s pela média.
  2. alpha_womersley é RECALCULADO sobre as médias do bloco (None se faltar uma
     entrada, como o NaN do treino).
  3. Calibração congelada nos 10 primeiros blocos da sessão, todos, mesmo os que
     depois não entram na tabela. Média e desvio como normalizar_causal_calibracao:
     desvio do pandas (Welford), desvio nulo ou indefinido passa à mediana entre as
     sessões da tabela (SD_RESERVA) e o divisor leva + 1e-6.
  4. hr_lag1 é o hr_mean_z do bloco anterior, seja ele qual for (no treino
     calcula-se antes de se deitarem linhas fora).
  5. Só entram na tabela os blocos com hr_lag1 e com o z das cinco âncoras. Há mais 7 âncoras
     opcionais (d/a, b/a, w25, w75, inclinação, reflection index), normalizadas como as outras
     mas sem excluírem o bloco se faltarem.
  6. As variações a 2/5/10 min vão buscar a linha dos blocos bid-8/-20/-40 (instante exato) e
     ficam em falta (None) se esse bloco não está na tabela; a volatilidade a 5 min só existe
     se houver as 20 linhas dos blocos bid-19..bid. Não se espera por 10 min de linhas: o
     score pode sair logo a seguir à calibração.
O treino deita fora os blocos fora de 70 < PAS < 190 antes de tudo isto. À
cabeceira não há PAS, por isso aqui esses blocos contam.

Verificação contra a tabela do treino no conjunto de teste inteiro, bloco a bloco:
3_tendencia_modelo_final/servidor_tempo/verificar_servidor_tempo.py. As medianas de SD_RESERVA
vêm do 06_desvio_reserva_morf.py (sd_reserva_ppg_morf.json).
"""
import math

import numpy as np

BLOCO_SEG = 15
JANELAS_POR_BLOCO = 3
BLOCOS_CALIB = 10                      # n_calib do treino
EPS_Z = 1e-6                           # eps de normalizar_causal_calibracao
ESCALAS_MIN = [2, 5, 10]
RHO_SANGUE = 1060
W5_BLOCOS = int(5 * 60 / BLOCO_SEG)    # 20 linhas = janela da volatilidade
N_MAX = int(10 * 60 / BLOCO_SEG)       # 40 linhas = maior desfasamento

# Âncoras morfológicas normalizadas (FEATURES_ANCORA de construir_tabela_ppg.py)
ANCORAS = ['alpha_womersley', 'area_mean', 'w50', 'ppg_crest_time', 'ppg_decay_time']          # exigidas: sem o z de uma delas o bloco sai da tabela
OPCIONAIS = ['apg_b_a', 'apg_d_a', 'w25', 'w75', 'ppg_max_slope', 'ppg_max_slope_norm', 'reflection_index']   # z em falta (None) não exclui o bloco
TODAS_ANCORAS = ANCORAS + OPCIONAIS
# famílias de variações a 2 e 5 min das características novas (como em comparar_caracteristicas_novas.py / tempo_fixo.py)
MORF = ['w50_global_z', 'ppg_crest_time_global_z', 'ppg_decay_time_global_z']
APG = ['apg_b_a_global_z', 'apg_d_a_global_z']
NOVAS = ['w25_global_z', 'w75_global_z', 'ppg_max_slope_norm_global_z', 'reflection_index_global_z']
ESCALAS_NOVAS = [2, 5]

SLOPE_COLS = ['hr_mean_z', 'area_mean_global_z', 'alpha_womersley_global_z']
VOL_COL = 'hr_mean_z'

# Mediana, entre as sessões da tabela, do desvio da janela de calibração
SD_RESERVA = {
    'hr_mean': 1.529912296622039,
    'alpha_womersley': 2.9763542759689265,
    'area_mean': 0.3859974135154677,
    'w50': 0.00491838482120816,
    'ppg_crest_time': 0.0040327021397326985,
    'ppg_decay_time': 0.013186126685963789,
    # novas (sd_reserva_ppg_morf.json, 06_desvio_reserva_morf.py)
    'apg_b_a': 0.03514147170017321,
    'apg_d_a': 0.030776941875933155,
    'w25': 0.006515124416097769,
    'w75': 0.0033885883284779277,
    'ppg_max_slope': 0.2703413773680756,
    'ppg_max_slope_norm': 0.18221490511056565,
    'reflection_index': 0.0659023520065862,
}


def _mu_sd(vals, reserva):
    """Média e divisor do z-score de calibração, com as regras de normalizar_causal_calibracao.

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


class EstadoSessaoPPG:
    """Estado por pulseira: acumula janelas, fecha blocos, congela a calibração."""

    def __init__(self):
        self.idx_janela = 0
        self.bloco_atual = 0
        self.pendentes = []              # janelas do bloco a fechar
        self.espera = []                 # blocos fechados antes de a calibração congelar
        self.cal = None                  # médias e divisores congelados da calibração
        self.hr_z_anterior = None        # hr_mean_z do bloco anterior (hr_lag1)
        self.linhas = {}                 # bid -> z do bloco, só os últimos N_MAX + 1 blocos (a busca é por instante, não por posição)


def _fechar_bloco(pendentes):
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

    # alpha_womersley recalculado sobre as médias do bloco (ver cabeçalho)
    if (b.get('ppg_valley_intensity') is not None
            and b.get('ppg_ac_amplitude') is not None
            and b.get('hr_mean') is not None):
        b['alpha_womersley'] = float(
            b['ppg_valley_intensity']
            * np.sqrt((RHO_SANGUE * (b['hr_mean'] / 60.0)) / (b['ppg_ac_amplitude'] + 1e-6)))
    else:
        b['alpha_womersley'] = None
    return b


def _linha(estado, b):
    """z-scores do bloco; acrescenta-o às linhas da tabela se o treino o guardasse."""
    cal = estado.cal
    z = {'hr_mean_z': (b['hr_mean'] - cal['mu_hr_mean']) / cal['sd_hr_mean']}
    z['hr_lag1'], estado.hr_z_anterior = estado.hr_z_anterior, z['hr_mean_z']
    for a in ANCORAS:
        v = b.get(a)
        z[f'{a}_global_z'] = (None if v is None or cal[f'mu_{a}'] is None
                              else (float(v) - cal[f'mu_{a}']) / cal[f'sd_{a}'])
    for a in OPCIONAIS:
        v = b.get(a)
        z[f'{a}_global_z'] = (None if v is None or cal[f'mu_{a}'] is None
                              else (float(v) - cal[f'mu_{a}']) / cal[f'sd_{a}'])
    z['_bid'] = b['_bid']
    exigidas = ['hr_mean_z', 'hr_lag1'] + [f'{a}_global_z' for a in ANCORAS]
    if any(z[k] is None for k in exigidas):
        return None
    estado.linhas[z['_bid']] = z
    for k in [k for k in estado.linhas if k < z['_bid'] - N_MAX]:
        del estado.linhas[k]
    return z


def derivar_features_ppg(raw, estado, paciente_info):
    """Acumula uma janela; devolve as features do modelo quando um bloco fecha
    e há histórico suficiente. `raw` pode ser None (janela rejeitada) — a
    chamada conta na mesma para o alinhamento dos blocos, tal como no treino,
    onde uma janela rejeitada continua a ocupar o seu window_idx.
    """
    estado.idx_janela += 1
    bid = (estado.idx_janela - 1) // JANELAS_POR_BLOCO
    fechou = None

    if bid != estado.bloco_atual:
        b = _fechar_bloco(estado.pendentes)
        estado.pendentes = []
        if b is not None and b.get('hr_mean') is not None:
            b['_bid'] = estado.bloco_atual
            fechou = b
        estado.bloco_atual = bid

    if raw is not None:
        estado.pendentes.append(raw)

    if fechou is None:
        return None

    # ── calibração congelada nos primeiros BLOCOS_CALIB blocos ───────────────
    if estado.cal is None:
        estado.espera.append(fechou)
        if len(estado.espera) < BLOCOS_CALIB:
            return None                   # janela de calibração ainda a encher
        cal = {}
        for nome in ('hr_mean', *TODAS_ANCORAS):
            cal[f'mu_{nome}'], cal[f'sd_{nome}'] = _mu_sd(
                [x.get(nome) for x in estado.espera], SD_RESERVA[nome])
        estado.cal = cal
        for b in estado.espera[:-1]:      # os blocos de calibração também são linhas
            _linha(estado, b)
        estado.espera = []

    atual = _linha(estado, fechou)
    if atual is None:
        return None                       # bloco fora da tabela

    linhas, bid_atual = estado.linhas, atual['_bid']
    out = {k: atual[k] for k in ('hr_mean_z', 'hr_lag1', '_bid', *(f'{a}_global_z' for a in TODAS_ANCORAS))}
    out['age'] = paciente_info.get('idade') or 0
    genero = paciente_info.get('genero') or 'M'
    out['gender_bin'] = 0 if genero == 'M' else 1

    def atras(coluna, minutos):
        """Variação face ao bloco a EXATAMENTE `minutos` atrás; em falta (None) se esse bloco não está na tabela ou não tem a coluna."""
        antes = linhas.get(bid_atual - int(minutos * 60 / BLOCO_SEG))
        if antes is None or antes[coluna] is None or atual[coluna] is None:
            return None
        return atual[coluna] - antes[coluna]

    for c in SLOPE_COLS:
        for m in ESCALAS_MIN:
            out[f'd_{c}_{m}m'] = atras(c, m)
    for c in MORF + APG + NOVAS:
        for m in ESCALAS_NOVAS:
            out[f'd_{c}_{m}m'] = atras(c, m)
    # volatilidade: só com os 20 blocos seguidos (instantes bid-19..bid), como rolling(20, min_periods=20) na grade
    janela = [linhas.get(bid_atual - i) for i in range(W5_BLOCOS)]
    out[f'vol_{VOL_COL}_5m'] = (float(np.std([x[VOL_COL] for x in janela], ddof=1)) if all(x is not None for x in janela) else None)
    return out
