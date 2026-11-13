"""
derivacao_ppg_classificador.py — derivação causal por bloco, só com PPG.

Porte para streaming da tabela única da tendência,
  1_dados_e_caracteristicas/tabelas_de_treino_e_teste_da_tendencia/construir_tabela_ppg.py   (colapso, z-scores, linhas da tabela)
e das variações e da volatilidade calculadas nessa tabela. Os dois modelos de tendência
(regressão do delta e classificador) foram treinados e avaliados nesta mesma tabela.

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
  5. Só entram na tabela os blocos com hr_lag1 e com o z das cinco âncoras.
  6. Variações a 2/5/10 min e volatilidade a 5 min sobre as linhas da tabela, por
     posição: as últimas 8/20/40 linhas, como o .shift(N) e o .rolling(N) do pandas.
     São 2/5/10 min quando nenhum bloco é rejeitado.
O treino deita fora os blocos fora de 70 < PAS < 190 antes de tudo isto. À
cabeceira não há PAS, por isso aqui esses blocos contam.

Verificação contra a tabela do treino no conjunto de teste inteiro, bloco a bloco:
5_servidor_e_carga/verificacao_da_extracao_e_derivacao_contra_o_treino/05_verificar_servidor_teste_completo.py.
As medianas de SD_RESERVA vêm do 06_desvio_de_reserva_tendencia.py.
"""
import math
from collections import deque

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
ANCORAS = ['alpha_womersley', 'area_mean', 'w50', 'ppg_crest_time', 'ppg_decay_time']

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
        self.linhas = deque(maxlen=N_MAX + 1)   # z das últimas linhas da tabela


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
    z['_bid'] = b['_bid']
    if any(v is None for v in z.values()):
        return None
    estado.linhas.append(z)
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
        for nome in ('hr_mean', *ANCORAS):
            cal[f'mu_{nome}'], cal[f'sd_{nome}'] = _mu_sd(
                [x.get(nome) for x in estado.espera], SD_RESERVA[nome])
        estado.cal = cal
        for b in estado.espera[:-1]:      # os blocos de calibração também são linhas
            _linha(estado, b)
        estado.espera = []

    atual = _linha(estado, fechou)
    if atual is None or len(estado.linhas) <= N_MAX:
        return None                       # bloco fora da tabela, ou 10 min de linhas ainda a encher

    linhas = estado.linhas
    out = {k: atual[k] for k in ('hr_mean_z', 'hr_lag1', '_bid', *(f'{a}_global_z' for a in ANCORAS))}
    out['age'] = paciente_info.get('idade') or 0
    genero = paciente_info.get('genero') or 'M'
    out['gender_bin'] = 0 if genero == 'M' else 1

    # ── variações multi-escala / volatilidade, por posição nas linhas ────────
    for c in SLOPE_COLS:
        for m in ESCALAS_MIN:
            n = int(m * 60 / BLOCO_SEG)
            out[f'd_{c}_{m}m'] = atual[c] - linhas[-1 - n][c]
    out[f'vol_{VOL_COL}_5m'] = float(np.std([x[VOL_COL] for x in list(linhas)[-W5_BLOCOS:]], ddof=1))
    return out
