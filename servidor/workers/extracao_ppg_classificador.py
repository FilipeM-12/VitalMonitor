"""
extracao_ppg_classificador.py — extração de features de UMA janela usando SÓ o PPG.

Porte para streaming de
  1_dados_e_caracteristicas/extracao_ppg_e_segunda_derivada_da_tendencia/09_extrair_features_ppg_apg.py
sem a APG e sem o alpha_womersley_b, que nenhum modelo usa. É o código que extraiu as
janelas da tabela única da tendência (1_dados_e_caracteristicas/tabelas_de_treino_e_teste_da_tendencia/construir_tabela_ppg.py).

Existe para o worker de tendência "só-PPG" poder correr sem AD8232: a
`extrair_features_janela` do worker_absoluto.py deteta primeiro os picos R e devolve
None se o ECG for mau, por isso um worker que a use nunca é verdadeiramente
independente do ECG, mesmo que o MODELO não use nenhuma feature de ECG.

Tem de ficar byte-a-byte equivalente ao script de treino: qualquer divergência
faz o modelo receber em produção features de uma distribuição diferente
daquela em que foi treinado. As constantes e a ordem das operações são as
mesmas; a única diferença é a origem do sinal (buffer MQTT em vez de wfdb).

Canal usado: IR do MAX30102 (equivalente ao PLETH do MIMIC).
"""
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.integrate import simpson
from scipy.stats import skew

RHO_SANGUE = 1060

# Portões — TÊM de ser os mesmos do script de treino.
MIN_BATIMENTOS = 8
MIN_BATIMENTOS_KEPT = 6
CV_RITMO_MAX = 0.25
DISPERSAO_MAX = 0.05
HR_MIN, HR_MAX = 45, 140
CREST_MIN, CREST_MAX = 0.05, 0.40


def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)
    return filtfilt(b, a, data)


def corrigir_orientacao(win_ppg):
    """Repõe a convenção do MIMIC PLETH (pico sistólico = MÁXIMO).

    O MAX30102 devolve intensidade de luz recebida: mais sangue -> mais
    absorção -> MENOS luz, portanto o pico sistólico sai como MÍNIMO. O PLETH
    do MIMIC, com que os modelos foram treinados, tem a convenção oposta. Sem
    esta correção toda a morfologia entra espelhada: a subida sistólica sai mais
    lenta do que a descida, o que é fisiologicamente impossível.

    Critério: numa onda de PPG bem orientada a subida sistólica é muito mais
    abrupta do que a descida diastólica, por isso a derivada tem cauda positiva
    longa e skew(diff) > 0. É o análogo do `skew(win_ecg) < 0` que o worker_absoluto.py
    já usava para os elétrodos trocados.

    O critério separa o MAX30102 sem correção (skew negativo) do PLETH do MIMIC
    (skew positivo).
    """
    return -win_ppg if skew(np.diff(win_ppg)) < 0 else win_ppg


def processar_janela_so_ppg(win_ppg, fs):
    """Segmenta batimentos e extrai features usando apenas o PPG.

    `win_ppg` é o PPG JÁ filtrado (passa-baixo 8 Hz), como no treino.
    Devolve um dict de features ou None se a janela não passar os portões.
    """
    win_ppg = corrigir_orientacao(win_ppg)
    win = win_ppg - np.min(win_ppg) + 0.01
    amplitude_janela = float(np.max(win) - np.min(win))
    if amplitude_janela <= 0:
        return None

    prom_min = amplitude_janela * 0.15
    picos, _ = find_peaks(win, distance=int(fs * 0.30), prominence=prom_min)
    if len(picos) < MIN_BATIMENTOS:
        return None

    pp = np.diff(picos) / fs
    if len(pp) < 2:
        return None
    cv_ritmo = float(np.std(pp) / (np.mean(pp) + 1e-9))
    hr_mediana = 60.0 / (float(np.median(pp)) + 1e-9)
    if not (HR_MIN <= hr_mediana <= HR_MAX):
        return None
    if cv_ritmo > CV_RITMO_MAX:
        return None

    amp_pulsatil = float(np.median([win[p] for p in picos])) - float(np.median(win))
    if amp_pulsatil <= 0:
        return None
    if float(np.max(np.abs(win - np.median(win)))) > 4 * amp_pulsatil:
        return None

    batimentos = []
    n = len(picos)
    for i in range(n):
        pk = int(picos[i])
        pk_next = int(picos[i + 1]) if i + 1 < n else None
        pk_prev = int(picos[i - 1]) if i > 0 else None

        hr = round((60 * fs) / (pk_next - pk), 2) if pk_next is not None else None

        ini = max(pk_prev if pk_prev is not None else 0, pk - int(0.4 * fs), 0)
        if ini >= pk:
            continue
        vly = ini + int(np.argmin(win[ini:pk]))

        vly_val = float(win[vly])
        ac = float(win[pk] - win[vly])
        if ac <= 0:
            continue

        crest_time = round((pk - vly) / fs, 4)
        if not (CREST_MIN <= crest_time <= CREST_MAX):
            continue

        omega = (hr / 60.0) if hr is not None else 1.2
        alpha = float(vly_val * np.sqrt((RHO_SANGUE * omega) / (ac + 1e-6)))

        limite = pk_next if pk_next is not None else min(pk + int(0.9 * fs), len(win))
        pe_ini = pk + int(0.04 * fs)
        pe_fim = min(pk + int(0.50 * fs), len(win), limite)
        if pe_ini < pe_fim:
            seg = win[pe_ini:pe_fim]
            mins, _ = find_peaks(-seg, distance=int(fs * 0.05))
            pe = int(pe_ini + (mins[0] if len(mins) > 0 else np.argmin(seg)))
            decay_time = round((pe - pk) / fs, 4)
        else:
            decay_time = None

        p_low = max(0, pk - int(0.1 * fs))
        p_high = min(len(win), pk + int(0.3 * fs))
        pulso = win[p_low:p_high]
        pulse_area = w50 = None
        if len(pulso) > 1 and (np.max(pulso) - np.min(pulso)) > 0.01:
            norm_p = (pulso - np.min(pulso)) / (np.max(pulso) - np.min(pulso))
            pulse_area = round(float(simpson(y=norm_p)), 4)
            idx50 = np.where(norm_p >= 0.5)[0]
            if len(idx50) > 1:
                w50 = round(float((idx50[-1] - idx50[0]) / fs), 4)

        batimentos.append({
            'hr': hr, 'crest_time': crest_time, 'decay_time': decay_time,
            'alpha': round(alpha, 4),
            'pulse_area': pulse_area, 'w50': w50,
            'ppg_valley_intensity': round(vly_val, 4),
            'ppg_ac_amplitude': round(ac, 4),
        })

    crests = np.array([b['crest_time'] for b in batimentos], dtype=float)
    if len(crests) >= 4:
        q1, q3 = np.percentile(crests, [25, 75])
        iqr = (q3 - q1) if (q3 - q1) != 0 else 0.05
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        kept = [b for b in batimentos if lo <= b['crest_time'] <= hi]
    else:
        kept = list(batimentos)

    kept = [b for b in kept if b['hr'] is None or HR_MIN < b['hr'] < HR_MAX]
    if len(kept) < MIN_BATIMENTOS_KEPT:
        return None
    if float(np.std([b['crest_time'] for b in kept])) >= DISPERSAO_MAX:
        return None

    def med(key):
        vals = [b[key] for b in kept if b.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        'hr_mean': med('hr'),
        'w50': med('w50'),
        'area_mean': med('pulse_area'),
        'ppg_valley_intensity': med('ppg_valley_intensity'),
        'ppg_ac_amplitude': med('ppg_ac_amplitude'),
        'ppg_crest_time': med('crest_time'),
        'ppg_decay_time': med('decay_time'),
        'alpha_womersley': med('alpha'),
    }


def extrair_features_janela_ppg(snap, fs, feature_window, feature_margem):
    """Interface equivalente à `extrair_features_janela` do worker_absoluto.py, mas só PPG.

    `snap` é a lista de dicts do buffer ({ecg, ir, red}); só o campo `ir` é lido
    — o `ecg` é ignorado de propósito, para o worker funcionar com o AD8232
    desligado.

    Mantém a estratégia de margens do worker_absoluto.py: filtra o buffer todo e usa a
    janela central, para os transientes de bordo do filtfilt ficarem fora da
    janela analisada.
    """
    if len(snap) < feature_window:
        return None

    margem = (len(snap) - feature_window) // 2
    if margem >= 1:
        bloco = snap
    else:
        bloco, margem = snap[-feature_window:], 0

    ir_raw = np.array([s["ir"] for s in bloco], dtype=float)

    # Sensor provavelmente não está no dedo (mesmo critério do worker_absoluto.py)
    if np.mean(ir_raw) < 5_000:
        return None

    win_ppg = butter_lowpass_filter(ir_raw, 8.0, fs)
    if margem:
        win_ppg = win_ppg[margem:margem + feature_window]

    if np.std(win_ppg) < 10:      # sinal plano (sem dedo)
        return None

    return processar_janela_so_ppg(win_ppg, fs)
