"""
frequencia_respiratoria.py — estimação da frequência respiratória a partir do PPG.

PORQUÊ EXISTE
    A respiração modula o PPG de três formas conhecidas:
      RIIV  oscilação da linha de base (intensidade contínua)
      RIAV  modulação da amplitude do pulso
      RIFV  modulação do intervalo entre batimentos (arritmia sinusal)

    Este módulo usa o RIIV. Nas aquisições de bancada com o dedo (MAX30102) o RIIV
    mostrou um pico espectral concentrado que acompanhou a contagem manual de
    respirações, enquanto o RIAV e o RIFV deram um espectro plano, em que a
    escolha do pico é só ruído. O RIAV e o RIFV discordam do RIIV, mas também
    não estavam validados.

LIMITE CONHECIDO — HIPERVENTILAÇÃO
    Num registo com respiração voluntariamente acelerada, nenhuma derivação
    acertou. A componente respiratória existe no espectro mas fica abaixo de uma
    oscilação vasomotora lenta, de frequência muito menor, que concentra a maior
    parte da potência da banda. A hipocapnia provocada pela hiperventilação gera essa
    vasomotricidade, e respirações rápidas são mais superficiais: o artefacto
    cresce ao mesmo tempo que o sinal encolhe.

    Não se sabe se o mesmo acontece na taquipneia clínica, que não é voluntária
    e cursa com CO₂ normal ou alto. É para este tipo de caso que existe o portão
    de qualidade: em vez de devolver o pico errado, o módulo cala-se.

O QUE ESTE MÓDULO NÃO É
    Não está validado num grupo de doentes nem contra capnografia. O que está
    medido é a concordância com contagem manual, num só sujeito, em repouso.

DESENHO
    A estimativa consome a linha de base decimada — um ponto por lote MQTT,
    cerca de 4 Hz — e não as amostras em bruto. Isso permite manter 120 s de
    histórico com ~500 pontos por doente, em vez de alargar o buffer de amostras
    de 25 s para 120 s.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, find_peaks, welch

# Banda respiratória: 6 a 54 resp/min. O tecto vai bem acima dos 30 porque a
# taquipneia é precisamente o sinal de deterioração que interessa apanhar.
# ATENÇÃO: as oscilações vasomotoras que enganam o método (hiperventilação,
# pressão do dedo) vivem nos 6-9 resp/min — dentro desta banda. O chão a 6
# resp/min é uma decisão explícita, para não perder bradipneia, e deixa o risco de
# aceitar uma dessas oscilações como FR válida.
BANDA_LO, BANDA_HI = 0.1, 0.9

# Histórico mínimo para arriscar uma estimativa: a janela inteira. Os limiares
# foram afinados em janelas de 90 s; com janelas mais curtas o portão
# aceita falsos positivos. O chamador tem de guardar uma amostra a mais do que
# a janela ao cortar a série, senão a cobertura fica sempre uns décimos abaixo
# disto. Custa 30 s a mais de aquecimento no início de cada sessão.
JANELA_MIN_S = 90.0

# Comprimento máximo da janela. Mais curta, aceita mais estimativas
# mas deixa passar falsos positivos; mais longa, deixa de os ter mas aceita menos,
# porque a frequência varia ao longo do registo e o pico espalha-se. 90 s é o
# compromisso.
JANELA_MAX_S = 90.0

# Cadência da série de linha de base (um ponto por lote de 25 amostras).
# Nyquist a 2 Hz, muito acima do tecto de 0,9 Hz da banda.
FS_SERIE = 4.0

# Um pico só é aceite se cumprir DUAS condições, porque cada uma falha sozinha
# de forma complementar:
#   - concentração: fração da potência da banda no bin do pico. Sozinha,
#     rejeita picos verdadeiros mas largos (ritmo a vaguear) e picos
#     estreitos afogados por uma bossa vasomotora.
#   - proeminência: potência do pico sobre o chão local (mediana móvel de 9
#     bins). Sozinha, premeia agulhas de 1% da banda sobre um chão quase nulo
#     e a própria bossa vasomotora.
# Os limiares são 15% e 2,0 sobre janelas de 90 s.
# Limiares mais baixos aceitam mais mas inventam valores
# na hiperventilação.
CONCENTRACAO_MIN = 0.15
PROEMINENCIA_MIN = 2.0

# Zona cardíaca excluída. Com a FC abaixo de cerca de 54 bpm a frequência do
# pulso cai dentro da banda (até 0,9 Hz), e a média de 25 amostras quase não a
# atenua. O pico do coração é estreito e forte, passa os dois portões e saía
# como FR: num ensaio com FC baixa a estimativa dava o valor da
# própria FC. A zona à volta da mediana das FC da
# janela, com esta fração de cada lado, sai da procura do pico e da potência da
# banda usada na concentração. A mediana e não o mínimo e o máximo: a FC da via
# rápida tem valores soltos que alargariam a zona até tapar a própria respiração.
MARGEM_FC = 0.10


def _passa_banda(x: np.ndarray, fs: float) -> np.ndarray:
    ny = 0.5 * fs
    b, a = butter(2, [max(BANDA_LO / ny, 1e-6), min(BANDA_HI / ny, 0.99)],
                  btype='band')
    return filtfilt(b, a, x)


def _pico_espectral(x: np.ndarray, fs: float,
                    excluir: tuple[float, float] | None = None) -> tuple[float, float, float, bool]:
    """Pico respiratório mais credível: (resp/min, concentração, proeminência,
    aceite).

    Entre os máximos locais que cumprem CONCENTRACAO_MIN e PROEMINENCIA_MIN,
    escolhe o mais proeminente. Se nenhum cumprir, devolve o mais proeminente
    de todos com `aceite=False`, para o chamador poder dizer porquê. `excluir`
    é a zona cardíaca (Hz) que fica de fora da banda.
    """
    if len(x) < 16 or not np.isfinite(x).all() or np.std(x) < 1e-12:
        return float('nan'), 0.0, 0.0, False
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(fs * 60)))
    m = (f >= BANDA_LO) & (f <= BANDA_HI)
    if excluir is not None:
        m &= ~((f >= excluir[0]) & (f <= excluir[1]))
    if not m.any() or p[m].sum() <= 0:
        return float('nan'), 0.0, 0.0, False
    f_banda, p_banda = f[m], p[m]
    conc = p_banda / p_banda.sum()
    proem = p_banda / (median_filter(p_banda, size=9, mode='nearest') + 1e-12)
    picos, _ = find_peaks(proem)
    if len(picos) == 0:
        picos = np.array([int(np.argmax(p_banda))])
    bons = [i for i in picos if conc[i] >= CONCENTRACAO_MIN and proem[i] >= PROEMINENCIA_MIN]
    if bons:
        i = max(bons, key=lambda k: proem[k])
        return float(f_banda[i] * 60), float(conc[i]), float(proem[i]), True
    i = max(picos, key=lambda k: proem[k])
    return float(f_banda[i] * 60), float(conc[i]), float(proem[i]), False


def ponto_de_linha_de_base(t_ms: np.ndarray, ir: np.ndarray) -> tuple[float, float]:
    """Reduz um lote de amostras a um ponto da série de linha de base.

    A média do lote é já uma decimação passa-baixo: com 25 amostras a ~104 Hz a
    janela é de 240 ms, e o que interessa (≤ 0,9 Hz) atravessa-a intacto.

    O instante vem do carimbo do próprio dispositivo, não do relógio do
    servidor nem do contador de amostras — com QoS 0 há mensagens perdidas, e
    tanto o jitter de rede como as perdas deformariam a base de tempo do
    espectro.
    """
    return float(np.mean(t_ms)) / 1000.0, float(np.mean(ir))


def zona_cardiaca(fcs) -> tuple[float, float] | None:
    """Zona do espectro (Hz) ocupada pelo pulso, a partir das FC (bpm) da janela."""
    fcs = [float(x) for x in (fcs or []) if x]
    if not fcs:
        return None
    fc = float(np.median(fcs)) / 60.0
    return fc * (1 - MARGEM_FC), fc * (1 + MARGEM_FC)


def estimar(t: np.ndarray, v: np.ndarray, fcs=None) -> dict:
    """FR a partir da série de linha de base acumulada.

    `t` em segundos (relógio do dispositivo) e `v` a linha de base, ordenados no
    tempo. O chamador é responsável por manter a janela (ver JANELA_MAX_S).
    `fcs` são as FC (bpm) medidas na mesma janela: a zona do pulso fica de fora
    (ver MARGEM_FC). Sem elas a banda fica inteira.

    Devolve {fr, qualidade, proeminencia, motivo}. `fr` é None quando não há
    estimativa fiável, e `motivo` diz porquê. `qualidade` é a concentração do
    pico na banda.
    """
    vazio: dict = dict(fr=None, qualidade=None, proeminencia=None, motivo="")
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if len(t) < 16 or len(t) != len(v):
        return {**vazio, "motivo": "série curta"}
    if not (np.isfinite(t).all() and np.isfinite(v).all()):
        return {**vazio, "motivo": "série com valores inválidos"}

    duracao = float(t[-1] - t[0])
    if duracao < JANELA_MIN_S:
        return {**vazio, "motivo": f"histórico curto ({duracao:.0f} s)"}

    # Reamostragem uniforme: os lotes chegam a cadência constante no relógio do
    # dispositivo, mas as perdas de QoS 0 abrem buracos que o espectro não
    # tolera.
    t_u = np.arange(t[0], t[-1], 1.0 / FS_SERIE)
    if len(t_u) < 40:
        return {**vazio, "motivo": "série uniforme demasiado curta"}
    vi = np.interp(t_u, t, v)

    fr, conc, proem, aceite = _pico_espectral(
        _passa_banda(vi - vi.mean(), FS_SERIE), FS_SERIE, zona_cardiaca(fcs))
    if not np.isfinite(fr):
        return {**vazio, "motivo": "sem pico na banda respiratória"}
    if not aceite:
        return {**vazio, "qualidade": round(conc, 3), "proeminencia": round(proem, 2),
                "motivo": f"pico mal definido ({conc * 100:.0f}% da banda, "
                          f"proeminência {proem:.1f})"}

    return dict(fr=round(float(fr), 1), qualidade=round(conc, 3),
                proeminencia=round(proem, 2), motivo="")
