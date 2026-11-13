#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulador_pulseira.py — substitui o ESP32-C3 para testar o servidor + worker.

Publica no broker MQTT exactamente o mesmo payload que o firmware
ECG_e_PPG.ino: lotes de 25 amostras a 100 Hz (250 ms por lote), no
tópico  pulseiras/raw/<PULSEIRA_ID>,  com o formato

    {"pulseira_id": "...", "token": "...",
     "amostras": [{"t": 12345, "ecg": 2231, "ir": 37402, "red": 55910,
                   "temp": 36.51}, ...]}

Como o firmware, NÃO filtra nada — o sinal vai em contagens do ADC, tal como sai
dos sensores. Toda a filtragem, extracção de features e inferência acontecem do
lado do worker.

DUAS FONTES DE SINAL
────────────────────
  --fonte sintetico  (omissão)  gera ECG e PPG a partir de um modelo de ondas.
                                Duração ilimitada e HR/PAT controláveis.
  --fonte csv                   reproduz as aquisições reais da bancada
                                (Dados/ECG+PPG/Configuração 1/*_bruto_*.csv),
                                em ciclo. Morfologia autêntica, ~62 s de
                                conteúdo por ficheiro.

As amplitudes do modo sintético (as constantes ECG_*, IR_* e RED_* abaixo) vêm dessas
aquisições reais. É por isso que o sinal gerado atravessa os mesmos portões de
qualidade do worker que o sinal verdadeiro.

O QUE ESTE SIMULADOR PROVA — E O QUE NÃO PROVA
──────────────────────────────────────────────
PROVA: que a cadeia MQTT → buffer → filtros → extracção → derivação → modelo →
base de dados corre de ponta a ponta, sem hardware; que os portões de qualidade
aceitam sinal bom e rejeitam sinal mau; e que o alinhamento dos blocos aguenta
janelas rejeitadas (usar --artefactos para o exercitar).

NÃO PROVA: que a previsão de PA está certa. A relação PAT↔PA usada aqui é
inventada (ver PAT_POR_MMHG); o modelo foi treinado em MIMIC e não tem razão
nenhuma para concordar com ela. Uma subida imposta de PA faz o valor previsto
mexer-se, mas a magnitude não tem significado clínico. Para medir exactidão o
único caminho é o holdout do MIMIC.

EXEMPLOS
────────
  # 6 minutos em repouso, o suficiente para passar a calibração (~2,5 min)
  python3 simulador_pulseira.py --duracao 360

  # rampa de PA, 8× mais depressa, para ver a previsão a reagir
  python3 simulador_pulseira.py --perfil rampa --velocidade 8 --duracao 600

  # com artefactos de movimento e dedo fora do sensor
  python3 simulador_pulseira.py --artefactos 0.15 --duracao 600

  # sinal real da bancada em vez de sintético
  python3 simulador_pulseira.py --fonte csv --duracao 600

  # sem broker: escreve o payload para ficheiro, para inspecção
  python3 simulador_pulseira.py --duracao 30 --sem-mqtt --guardar lote.jsonl

  # a acompanhar o que chega à base de dados
  python3 simulador_pulseira.py --duracao 600 --monitor

Variáveis de ambiente reconhecidas (mesmos nomes do worker):
  MQTT_BROKER  MQTT_PORT  AUTH_TOKEN  DATABASE_URL
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np

# O paho 2.x avisa que a API de callbacks v1 está obsoleta. Usa-se a v1 de
# propósito, porque é a que os workers do servidor usam: um simulador que ligasse ao broker
# de maneira diferente do worker deixaria de testar o mesmo caminho.
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=r".*Callback API version 1.*")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DO FIRMWARE  (têm de coincidir com ECG_e_PPG.ino)
# ─────────────────────────────────────────────────────────────────────────────

FS         = 100     # Hz
BATCH_SIZE = 25      # amostras por publicação → 250 ms por lote
TOPICO_FMT = "pulseiras/raw/{}"

# Quando é que a primeira previsão pode aparecer, em segundos de sinal.
#   25 s  o worker só extrai a 1.ª janela com o buffer cheio (BUFFER_MAX = 2500
#         amostras), porque precisa das margens de filtragem dos dois lados
#   10 s  o 1.º bloco só fecha ao fim de 3 janelas com passo de 5 s
#  150 s  a janela de calibração ocupa BLOCOS_CALIB = 10 blocos de 15 s
# Com sinal limpo dá 185 s. Com janelas rejeitadas é mais
# tarde: os blocos inválidos não contam para a calibração.
T_PRIMEIRA_PREVISAO = 25 + 10 + 10 * 15          # 185 s

PROGRESSO_S = 30     # de quantos em quantos segundos de SINAL se imprime o estado

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRAÇÃO DAS AMPLITUDES  (a partir das aquisições de bancada)
# ─────────────────────────────────────────────────────────────────────────────

ECG_BASE      = 2184.0    # linha de base do AD8232 no ADC de 12 bits
ECG_R_AMP     = 410.0     # altura do pico R acima da linha de base
ECG_RUIDO     = 8.0       # ruído branco (contagens)
ECG_REDE_AMP  = 4.0       # interferência da rede a 50 Hz
ECG_WANDER    = 25.0      # ondulação respiratória da linha de base

IR_DC         = 37300.0
IR_AC         = 594.0     # pico a pico do componente pulsátil
RED_DC        = 55850.0
PPG_RUIDO     = 12.0

TEMP_BASE     = 36.5
TEMP_RUIDO    = 0.05

# Relação PAT↔PA. INVENTADA — serve só para o sinal mexer de forma coerente
# quando o perfil pede uma variação de PA. Ver o aviso no cabeçalho.
PAT_BASE      = 0.250     # s, a SBP_BASE
SBP_BASE      = 120.0     # mmHg
PAT_POR_MMHG  = 0.0015    # s por mmHg (PA sobe → PAT encurta)
PAT_MIN       = 0.130
PAT_MAX       = 0.420


# ─────────────────────────────────────────────────────────────────────────────
# PERFIS FISIOLÓGICOS
# ─────────────────────────────────────────────────────────────────────────────

def perfil_valores(nome: str, t: float, duracao: float) -> tuple[float, float]:
    """Devolve (hr_media, sbp) no instante t. Ambos em unidades clínicas."""
    f = min(max(t / duracao, 0.0), 1.0) if duracao > 0 else 0.0

    if nome == "repouso":
        return 65.0, 120.0

    if nome == "rampa":
        # PA sobe 40 mmHg de forma contínua; HR acompanha discretamente
        return 65.0 + 8.0 * f, 120.0 + 40.0 * f

    if nome == "degrau":
        # salto a meio da sessão — o Kalman deve segui-lo com algum atraso
        return (68.0, 150.0) if f >= 0.5 else (65.0, 120.0)

    if nome == "exercicio":
        # esforço progressivo: taquicardia e hipertensão sistólica
        return 65.0 + 45.0 * f, 120.0 + 35.0 * f

    if nome == "hipotensao":
        # deterioração lenta — o cenário que a tese quer detectar
        return 70.0 + 20.0 * f, 125.0 - 35.0 * f

    raise ValueError(f"perfil desconhecido: {nome}")


PERFIS = ["repouso", "rampa", "degrau", "exercicio", "hipotensao"]


# ─────────────────────────────────────────────────────────────────────────────
# GERADOR SINTÉTICO
# ─────────────────────────────────────────────────────────────────────────────

# Ondas do ECG: (amplitude relativa ao R, desvio face ao R em s, largura em s).
# Q e S estão a ±30 ms para o worker medir uma duração de QRS de ~60 ms, dentro
# do intervalo 0,04–0,16 s que ele exige. A onda T a +200 ms cai dentro da
# janela de procura [R+120 ms, R+450 ms] usada na extracção.
ONDAS_ECG = [
    (0.10, -0.200, 0.028),   # P
    (-0.15, -0.030, 0.009),  # Q
    (1.00,  0.000, 0.014),   # R
    (-0.25,  0.030, 0.012),  # S
    (0.25,  0.200, 0.045),   # T
]

# PPG: sístole assimétrica (subida rápida, descida lenta) + onda dicrótica.
# A dicrótica fica a 280 ms do pico sistólico, dentro da distância mínima de
# 300 ms do find_peaks do worker, para nunca ser confundida com o pico.
PPG_SUBIDA   = 0.050   # s, largura do flanco ascendente
PPG_DESCIDA  = 0.090   # s, largura do flanco descendente
PPG_DIC_AMP  = 0.35
PPG_DIC_POS  = 0.280
PPG_DIC_LARG = 0.075


class GeradorSintetico:
    """Gera ECG + PPG amostra a amostra, com batimentos agendados.

    O agendamento é preciso: cada batimento tem o seu instante de R e o seu PAT,
    e o pico sistólico do PPG é colocado exactamente em (t_R + PAT). É isso que
    torna o PAT observável — é a grandeza que o worker mede e que alimenta o
    modelo.
    """

    LOOKAHEAD = 0.8   # s de batimentos agendados para lá do bloco actual

    def __init__(self, perfil: str, duracao: float, semente: int = 42,
                 artefactos: float = 0.0, spo2_alvo: float = 98.0):
        self.rng        = np.random.default_rng(semente)
        self.perfil     = perfil
        self.duracao    = duracao
        self.artefactos = artefactos
        self.spo2_alvo  = spo2_alvo

        self.n          = 0          # amostras já geradas
        self.batimentos = []         # (t_r, pat, amp_ppg)
        self.proximo_r  = 0.50       # s — instante do próximo R

        # estado dos artefactos: (amostras restantes, tipo)
        self.art_restantes = 0
        self.art_tipo      = None

        # último par (hr, sbp) verdadeiros, para o relatório
        self.hr_actual  = 65.0
        self.sbp_actual = 120.0

    # ── agendamento ──────────────────────────────────────────────────────────

    def _agendar_ate(self, t_fim: float) -> None:
        while self.proximo_r < t_fim:
            hr_med, sbp = perfil_valores(self.perfil, self.proximo_r, self.duracao)
            self.hr_actual, self.sbp_actual = hr_med, sbp

            # Arritmia sinusal respiratória: o RR encurta na inspiração.
            resp = np.sin(2 * np.pi * 0.25 * self.proximo_r)
            hr   = hr_med * (1.0 + 0.04 * resp) + self.rng.normal(0, 0.8)
            hr   = float(np.clip(hr, 40.0, 175.0))
            rr   = 60.0 / hr

            pat = PAT_BASE - (sbp - SBP_BASE) * PAT_POR_MMHG
            pat += self.rng.normal(0, 0.003)          # jitter batimento a batimento
            pat = float(np.clip(pat, PAT_MIN, PAT_MAX))

            # A amplitude do PPG também é modulada pela respiração
            amp = 1.0 + 0.10 * resp + self.rng.normal(0, 0.02)

            self.batimentos.append((self.proximo_r, pat, amp))
            self.proximo_r += rr

        # descarta batimentos que já não influenciam o sinal
        limite = t_fim - 2.0
        self.batimentos = [b for b in self.batimentos if b[0] > limite]

    # ── formas de onda ───────────────────────────────────────────────────────

    @staticmethod
    def _forma_ecg(dt: np.ndarray) -> np.ndarray:
        y = np.zeros_like(dt)
        for amp, mu, sigma in ONDAS_ECG:
            y += amp * np.exp(-0.5 * ((dt - mu) / sigma) ** 2)
        return y

    @staticmethod
    def _forma_ppg(dt: np.ndarray) -> np.ndarray:
        # sístole assimétrica: largura diferente antes e depois do pico
        larg = np.where(dt < 0, PPG_SUBIDA, PPG_DESCIDA)
        y = np.exp(-0.5 * (dt / larg) ** 2)
        y += PPG_DIC_AMP * np.exp(-0.5 * ((dt - PPG_DIC_POS) / PPG_DIC_LARG) ** 2)
        return y

    # ── geração ──────────────────────────────────────────────────────────────

    def gerar(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Devolve k amostras de (ecg, ir, red) em contagens do ADC."""
        t = (self.n + np.arange(k)) / FS
        self.n += k
        self._agendar_ate(t[-1] + self.LOOKAHEAD)

        ecg = np.zeros(k)
        ppg = np.zeros(k)
        for t_r, pat, amp in self.batimentos:
            dt_e = t - t_r
            m = np.abs(dt_e) < 0.6
            if m.any():
                ecg[m] += self._forma_ecg(dt_e[m])
            dt_p = t - (t_r + pat)
            m = (dt_p > -0.35) & (dt_p < 0.65)
            if m.any():
                ppg[m] += amp * self._forma_ppg(dt_p[m])

        # ── ECG em contagens ────────────────────────────────────────────────
        wander = ECG_WANDER * np.sin(2 * np.pi * 0.25 * t + 0.7)
        rede   = ECG_REDE_AMP * np.sin(2 * np.pi * 50.0 * t)
        ecg_c  = (ECG_BASE + ECG_R_AMP * ecg + wander + rede
                  + self.rng.normal(0, ECG_RUIDO, k))

        # ── PPG em contagens ────────────────────────────────────────────────
        # Tudo o que varia — pulso e modulação respiratória — entra nos dois
        # canais com a MESMA forma, e o RED leva o factor k_red. É isso que faz
        # a razão R = (AC_red/DC_red)/(AC_ir/DC_ir) sair no valor pedido: o
        # worker calcula AC com um desvio-padrão sobre a janela inteira, por isso
        # qualquer componente que exista só num dos canais desvia a SpO2.
        resp = 120.0 * np.sin(2 * np.pi * 0.25 * t)
        pulsatil = IR_AC * (ppg - 0.5) + resp

        ir_c = IR_DC + pulsatil + self.rng.normal(0, PPG_RUIDO, k)

        k_red = _razao_red(self.spo2_alvo) * RED_DC / IR_DC
        red_c = RED_DC + pulsatil * k_red + self.rng.normal(0, PPG_RUIDO, k)

        if self.artefactos > 0:
            ecg_c, ir_c, red_c = self._injectar_artefactos(ecg_c, ir_c, red_c, k)

        return ecg_c, ir_c, red_c

    # ── artefactos ───────────────────────────────────────────────────────────

    def _injectar_artefactos(self, ecg, ir, red, k):
        """Corrompe o sinal de vez em quando, como acontece na realidade.

        Serve para exercitar dois caminhos do worker que o sinal limpo nunca
        toca: a rejeição de janelas (mean(ir) < 5000, SNR do ECG, CV do RR) e o
        alinhamento dos blocos quando há janelas em falta.
        """
        if self.art_restantes <= 0:
            # em média um episódio por lote com probabilidade `artefactos`
            if self.rng.random() < self.artefactos * (k / FS):
                self.art_tipo      = self.rng.choice(["movimento", "dedo_fora"])
                self.art_restantes = int(self.rng.uniform(1.5, 4.0) * FS)
            else:
                return ecg, ir, red

        n = min(self.art_restantes, k)
        self.art_restantes -= n

        if self.art_tipo == "dedo_fora":
            # IR muito baixo, como nas aquisições reais sem dedo
            ir[:n]  = self.rng.normal(550, 120, n)
            red[:n] = self.rng.normal(1035, 200, n)
            ecg[:n] = ECG_BASE + self.rng.normal(0, 300, n)
        else:  # movimento
            desvio = np.cumsum(self.rng.normal(0, 40, n))
            ecg[:n] += desvio
            ir[:n]  += np.cumsum(self.rng.normal(0, 90, n))
            red[:n] += np.cumsum(self.rng.normal(0, 90, n))

        return ecg, ir, red


def _razao_red(spo2_alvo: float) -> float:
    """Inverte a curva de SpO2 do worker para achar a razão R correspondente.

    O worker calcula  spo2 = -45.060 R² + 30.354 R + 94.845.  Das duas raízes
    interessa a maior, que é a fisiológica (R ≈ 0,5 para 98 %).
    """
    a, b, c = -45.060, 30.354, 94.845 - spo2_alvo
    disc = b * b - 4 * a * c
    if disc < 0:
        return 0.5
    r1 = (-b + np.sqrt(disc)) / (2 * a)
    r2 = (-b - np.sqrt(disc)) / (2 * a)
    return float(max(r1, r2))


# ─────────────────────────────────────────────────────────────────────────────
# FONTE CSV — reprodução das aquisições reais
# ─────────────────────────────────────────────────────────────────────────────

CSV_PADRAO = Path(__file__).resolve().parents[1] / "Dados" / "ECG+PPG" / "Configuração 1"


class GeradorCSV:
    """Reproduz em ciclo um CSV bruto da bancada (colunas time_ms,ecg,ir,red,...).

    O ciclo é feito por reflexão (ida e volta) em vez de reinício, para não
    criar uma descontinuidade brusca na junção — que o worker leria como um
    artefacto e faria rejeitar uma janela a cada volta.
    """

    def __init__(self, caminho: Path):
        import csv as _csv
        ecg, ir, red = [], [], []
        with open(caminho, newline="", encoding="utf-8") as fh:
            for linha in _csv.DictReader(fh):
                try:
                    ecg.append(float(linha["ecg"]))
                    ir.append(float(linha["ir"]))
                    red.append(float(linha["red"]))
                except (KeyError, ValueError, TypeError):
                    continue
        if len(ecg) < FS * 10:
            raise ValueError(f"{caminho.name}: só {len(ecg)} amostras utilizáveis")

        e = np.array(ecg); i = np.array(ir); r = np.array(red)
        self.ecg = np.concatenate([e, e[-2:0:-1]])
        self.ir  = np.concatenate([i, i[-2:0:-1]])
        self.red = np.concatenate([r, r[-2:0:-1]])
        self.pos = 0
        self.nome = caminho.name
        self.n_orig = len(e)
        self.hr_actual = self.sbp_actual = float("nan")

    def gerar(self, k):
        idx = (self.pos + np.arange(k)) % len(self.ecg)
        self.pos = (self.pos + k) % len(self.ecg)
        return self.ecg[idx], self.ir[idx], self.red[idx]


def escolher_csv(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if not p.exists():
            raise SystemExit(f"ficheiro não encontrado: {p}")
        return p
    if not CSV_PADRAO.is_dir():
        raise SystemExit(f"pasta de aquisições não encontrada: {CSV_PADRAO}\n"
                         "usar --csv <ficheiro> ou --fonte sintetico")
    cands = sorted(CSV_PADRAO.glob("*bruto*.csv"),
                   key=lambda p: p.stat().st_size, reverse=True)
    if not cands:
        raise SystemExit(f"sem ficheiros *bruto*.csv em {CSV_PADRAO}")
    return cands[0]


# ─────────────────────────────────────────────────────────────────────────────
# MONITOR DA BASE DE DADOS
# ─────────────────────────────────────────────────────────────────────────────

class Monitor(threading.Thread):
    """Mostra o que vai aparecendo nas tabelas, para se ver o efeito ao vivo."""

    daemon = True

    def __init__(self, url: str, pulseira: str, intervalo: float = 10.0):
        super().__init__()
        self.url       = url
        self.pulseira  = pulseira
        self.intervalo = intervalo
        self.parar     = threading.Event()
        self.erro      = None

    def run(self):
        try:
            from sqlalchemy import create_engine, text
        except ImportError:
            self.erro = "sqlalchemy não instalado — monitor desligado"
            return
        try:
            eng = create_engine(self.url, pool_pre_ping=True)
            with eng.connect() as c:
                c.execute(text("SELECT 1"))
        except ModuleNotFoundError as e:
            # caso típico: correr fora do venv do servidor, sem psycopg2
            self.erro = (f"falta o driver da base de dados ({e.name}) — monitor "
                         f"desligado. Correr dentro do venv do servidor, ou "
                         f"pip install psycopg2-binary")
            return
        except Exception as e:
            self.erro = (f"sem ligação a {self.url.split('@')[-1]} "
                         f"({type(e).__name__}) — monitor desligado")
            return

        from sqlalchemy import text
        t0 = time.time()
        while not self.parar.wait(self.intervalo):
            try:
                with eng.connect() as c:
                    med = c.execute(text(
                        "SELECT count(*), max(bpm), max(spo2) FROM medicao "
                        "WHERE pulseira_id = :p"), {"p": self.pulseira}).first()
                    feat = c.execute(text(
                        "SELECT count(*), "
                        "count(features_ml), "
                        "count(sbp_predicted), "
                        "max(sbp_predicted) FROM medicao_absoluta "
                        "WHERE pulseira_id = :p"), {"p": self.pulseira}).first()
                    ult = c.execute(text(
                        "SELECT sbp_predicted FROM medicao_absoluta "
                        "WHERE pulseira_id = :p AND sbp_predicted IS NOT NULL "
                        "ORDER BY id DESC LIMIT 1"), {"p": self.pulseira}).first()
            except Exception as e:
                print(f"  [BD] erro: {e}", flush=True)
                continue

            ultimo = (f"última {ult[0]:.1f}" if ult else
                      "ainda sem previsão")
            print(f"  [BD +{time.time()-t0:4.0f}s] medicao={med[0]} "
                  f"(bpm≤{med[1]}, spo2≤{med[2]}) | "
                  f"medicao_absoluta={feat[0]} blocos={feat[1]} "
                  f"previsões={feat[2]} | {ultimo}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLICAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def construir_payload(pulseira: str, token: str, t_ms: np.ndarray,
                      ecg, ir, red, temp: float) -> str:
    """Reproduz o buildJson() do firmware, com os mesmos tipos e arredondamentos.

    O ESP32 envia ecg como int, ir/red como long e temp com 2 casas decimais.
    Manter isso importa: se o simulador enviasse floats, um erro de conversão do
    lado do worker passaria despercebido aqui e apareceria só com hardware real.
    """
    amostras = [
        {"t": int(t_ms[i]), "ecg": int(round(ecg[i])),
         "ir": int(round(ir[i])), "red": int(round(red[i])),
         "temp": round(temp, 2)}
        for i in range(len(t_ms))
    ]
    return json.dumps({"pulseira_id": pulseira, "token": token,
                       "amostras": amostras}, separators=(",", ":"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Simulador da pulseira ESP32-C3 (publica MQTT como o firmware)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Ver o cabeçalho do ficheiro para exemplos.")

    g = ap.add_argument_group("identidade")
    g.add_argument("--pulseira", default="PULSEIRA_01",
                   help="ID da pulseira (tem de estar atribuída a um doente)")
    g.add_argument("--token", default=os.environ.get("AUTH_TOKEN", "segredo123"))

    g = ap.add_argument_group("broker")
    g.add_argument("--broker", default=os.environ.get("MQTT_BROKER", "localhost"))
    g.add_argument("--porta", type=int, default=int(os.environ.get("MQTT_PORT", "1883")))
    g.add_argument("--utilizador", default="pulseira")
    g.add_argument("--password", default="pulseira_pass")
    g.add_argument("--sem-mqtt", action="store_true",
                   help="não publica; só gera (usar com --guardar)")

    g = ap.add_argument_group("sinal")
    g.add_argument("--fonte", choices=["sintetico", "csv"], default="sintetico")
    g.add_argument("--csv", help="ficheiro bruto a reproduzir (implica --fonte csv)")
    g.add_argument("--perfil", choices=PERFIS, default="repouso")
    g.add_argument("--artefactos", type=float, default=0.0, metavar="P",
                   help="taxa de episódios de movimento/dedo fora, por segundo "
                        "(ex.: 0.05). Exercita a rejeição de janelas.")
    g.add_argument("--spo2", type=float, default=98.0, metavar="PCT",
                   help="SpO2 que o worker deve calcular (80–100, omissão 98)")
    g.add_argument("--semente", type=int, default=42)
    g.add_argument("--sem-ecg", action="store_true",
                   help="publica o canal ecg a zeros, como se o AD8232 estivesse "
                        "desligado. Serve para provar que a variante ppg_puro do "
                        "worker de tendência prevê sem elétrodos — as variantes "
                        "que dependem dos picos R deixam de produzir seja o que for.")

    g = ap.add_argument_group("tempo")
    g.add_argument("--duracao", type=float, default=360.0,
                   help="segundos de sinal a gerar (360 = 6 min)")
    g.add_argument("--velocidade", type=float, default=1.0, metavar="N",
                   help="envia N× mais depressa do que o tempo real. As features "
                        "dependem da contagem de amostras e saem na mesma; as "
                        "linhas de medicao (BPM/SpO2) são limitadas a 1/s de "
                        "tempo real e saem em menor número.")

    g = ap.add_argument_group("saída")
    g.add_argument("--guardar", metavar="FICHEIRO",
                   help="grava cada payload publicado, uma linha JSON por lote")
    g.add_argument("--monitor", action="store_true",
                   help="consulta a base de dados a cada 10 s e mostra o estado")
    g.add_argument("--bd", default=os.environ.get(
        "DATABASE_URL", "postgresql://hospital:hospital_pass@localhost:5432/hospital"))
    g.add_argument("--silencioso", action="store_true")

    a = ap.parse_args()
    if a.csv:
        a.fonte = "csv"

    # ── fonte ───────────────────────────────────────────────────────────────
    if a.fonte == "csv":
        caminho = escolher_csv(a.csv)
        ger = GeradorCSV(caminho)
        desc = f"CSV real «{ger.nome}» ({ger.n_orig/FS:.0f} s, em ciclo)"
        if a.sem_ecg:
            desc += " — SEM ECG (canal a zeros)"
    else:
        ger = GeradorSintetico(a.perfil, a.duracao, a.semente, a.artefactos, a.spo2)
        desc = f"sintético, perfil «{a.perfil}», SpO2 {a.spo2:g}%"
        if a.artefactos:
            desc += f", artefactos {a.artefactos}/s"

    # ── broker ──────────────────────────────────────────────────────────────
    cliente = None
    if not a.sem_mqtt:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            print("erro: falta o paho-mqtt  (pip install paho-mqtt)", file=sys.stderr)
            return 2
        # O CONNACK chega depois do connect() e é ele que traz a recusa de
        # autenticação. Sem esperar por ele, um broker com allow_anonymous false
        # deixa o simulador a "publicar" para o vazio, com rc=0 nas primeiras
        # chamadas — que foi exactamente o que aconteceu ao testar contra o
        # hospital.conf. Daí este handshake explícito.
        estado_ligacao = {"rc": None}
        ligou = threading.Event()

        def _on_connect(_c, _u, _f, rc):
            estado_ligacao["rc"] = rc
            ligou.set()

        cliente = mqtt.Client(client_id=a.pulseira, clean_session=True)
        cliente.username_pw_set(a.utilizador, a.password)
        cliente.on_connect = _on_connect
        try:
            cliente.connect(a.broker, a.porta, keepalive=30)
        except Exception as e:
            print(f"erro: não liguei a {a.broker}:{a.porta} — {e}", file=sys.stderr)
            print("      o mosquitto está a correr?  systemctl status mosquitto",
                  file=sys.stderr)
            return 2
        cliente.loop_start()

        if not ligou.wait(5.0):
            print(f"erro: {a.broker}:{a.porta} não respondeu ao CONNECT em 5 s",
                  file=sys.stderr)
            cliente.loop_stop()
            return 2
        if estado_ligacao["rc"] != 0:
            motivos = {1: "versão do protocolo recusada",
                       2: "client_id recusado",
                       3: "broker indisponível",
                       4: "utilizador ou password errados",
                       5: "não autorizado"}
            motivo = motivos.get(estado_ligacao["rc"], "razão desconhecida")
            print(f"erro: o broker recusou a ligação — rc={estado_ligacao['rc']} "
                  f"({motivo})", file=sys.stderr)
            if estado_ligacao["rc"] in (4, 5):
                print("      o listener 1883 tem allow_anonymous false; usar "
                      "--utilizador/--password (por omissão pulseira/pulseira_pass)",
                      file=sys.stderr)
            cliente.loop_stop()
            return 2

    topico = TOPICO_FMT.format(a.pulseira)
    fh = open(a.guardar, "w", encoding="utf-8") if a.guardar else None

    mon = None
    if a.monitor:
        mon = Monitor(a.bd, a.pulseira)
        mon.start()
        time.sleep(0.5)
        if mon.erro:
            print(f"  aviso: {mon.erro}", flush=True)

    # ── cabeçalho ───────────────────────────────────────────────────────────
    n_lotes = int(a.duracao * FS / BATCH_SIZE)
    print("=" * 72)
    print(f"SIMULADOR DE PULSEIRA  ·  {a.pulseira}")
    print("=" * 72)
    print(f"  sinal      : {desc}")
    print(f"  destino    : {'(nenhum, --sem-mqtt)' if a.sem_mqtt else f'{a.broker}:{a.porta} → {topico}'}")
    print(f"  duração    : {a.duracao:.0f} s de sinal em {a.duracao/a.velocidade:.0f} s "
          f"reais ({a.velocidade:g}×)")
    print(f"  lotes      : {n_lotes} × {BATCH_SIZE} amostras a {FS} Hz")
    print(f"  1.ª previsão esperada por volta dos {T_PRIMEIRA_PREVISAO:.0f} s de "
          f"sinal (25 s a encher o buffer, 10 s até ao 1.º bloco, 10 blocos de "
          f"calibração)")
    if a.duracao < T_PRIMEIRA_PREVISAO:
        print(f"  AVISO: {a.duracao:.0f} s não chega para haver previsão nenhuma.")
    print("=" * 72, flush=True)

    # ── ciclo principal ─────────────────────────────────────────────────────
    periodo   = (BATCH_SIZE / FS) / a.velocidade
    t_arranque = time.time()
    t_ms0     = int(time.time() * 1000) % (2 ** 32)   # imita o millis() do ESP32
    gerados = enviados = falhados = 0
    temp = TEMP_BASE

    try:
        for lote in range(n_lotes):
            ecg, ir, red = ger.gerar(BATCH_SIZE)
            if a.sem_ecg:
                ecg = np.zeros_like(ecg)
            base = lote * BATCH_SIZE
            t_ms = (t_ms0 + (base + np.arange(BATCH_SIZE)) * (1000 // FS)) % (2 ** 32)
            temp += np.random.normal(0, TEMP_RUIDO * 0.1)
            temp = float(np.clip(temp, 35.5, 37.5))

            payload = construir_payload(a.pulseira, a.token, t_ms, ecg, ir, red, temp)

            gerados += 1
            if fh:
                fh.write(payload + "\n")

            if cliente is not None:
                info = cliente.publish(topico, payload, qos=0)
                if info.rc == 0:
                    enviados += 1
                else:
                    falhados += 1

            # progresso a cada PROGRESSO_S de SINAL, não de tempo real: assim a
            # densidade de linhas não depende da velocidade escolhida
            if not a.silencioso and lote % max(1, int(PROGRESSO_S * FS / BATCH_SIZE)) == 0:
                t_sinal = base / FS
                extra = ""
                if a.fonte == "sintetico":
                    pat = np.clip(PAT_BASE - (ger.sbp_actual - SBP_BASE) * PAT_POR_MMHG,
                                  PAT_MIN, PAT_MAX)
                    extra = (f" | HR {ger.hr_actual:5.1f}  SBP_imposta "
                             f"{ger.sbp_actual:5.1f}  PAT {pat:.3f}s")
                print(f"  t={t_sinal:6.1f}s  lotes={gerados:5d}"
                      f"  {len(payload):4d} B{extra}", flush=True)

            # espera até ao instante teórico do lote seguinte (evita deriva)
            alvo = t_arranque + (lote + 1) * periodo
            atraso = alvo - time.time()
            if atraso > 0:
                time.sleep(atraso)

    except KeyboardInterrupt:
        print("\n  interrompido pelo utilizador", flush=True)

    finally:
        if mon:
            mon.parar.set()
        if cliente is not None:
            time.sleep(0.5)          # deixa drenar a fila de publicação
            cliente.loop_stop()
            cliente.disconnect()
        if fh:
            fh.close()

    # ── relatório ───────────────────────────────────────────────────────────
    decorrido = time.time() - t_arranque
    print("=" * 72)
    print(f"  lotes gerados    : {gerados}   ({gerados * BATCH_SIZE} amostras, "
          f"{gerados * BATCH_SIZE / FS:.0f} s de sinal)")
    if a.sem_mqtt:
        print("  publicados       : nenhum (--sem-mqtt)")
    else:
        print(f"  publicados       : {enviados}"
              + (f"   FALHADOS {falhados}" if falhados else ""))
    print(f"  tempo real       : {decorrido:.0f} s")
    if a.guardar:
        print(f"  payloads em      : {a.guardar}")
    if not a.sem_mqtt:
        print()
        print("  Confirmar do lado do servidor:")
        print(f"    SELECT count(*) FROM medicao WHERE pulseira_id='{a.pulseira}';")
        print("    SELECT timestamp, sbp_predicted FROM medicao_absoluta")
        print(f"     WHERE pulseira_id='{a.pulseira}' AND sbp_predicted IS NOT NULL")
        print("     ORDER BY id DESC LIMIT 10;")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
