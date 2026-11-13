"""extracao_ppg.py — extração de características de UMA janela usando SÓ o PPG, com as de forma
(w25, w75, ppg_max_slope(_norm), reflection_index) e a APG (b/a, d/a).

Porte para streaming de 3_tendencia_modelo_final/extracao_morfologia/09b_extrair_features_ppg_morfologia.py, o código que extraiu
as janelas da tabela de treino da tendência. Verificação: 3_tendencia_modelo_final/servidor_tempo/verificar_extracao_morf.py.

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
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
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


def extrair_apg_features(pulso, fs):
    """Ondas a,b,c,d,e da 2ª derivada (APG) de UM pulso PPG e os rácios b/a, c/a,
    d/a + índice de envelhecimento (b-c-d-e)/a. Método Elgendi/Takazawa: NÃO precisa
    do pé nem do entalhe dicrótico (mais robusto que crest/decay times). None se não
    fiável.

    Cópia literal da função do 09_extrair_features_ppg_apg.py (código próprio em scipy, método
    Takazawa/Elgendi, sem NeuroKit) — não alterar, senão deixa de ser comparável.
    """
    if len(pulso) < int(0.3 * fs):
        return None
    win_sg = max(5, int(0.05 * fs) | 1)   # ímpar, ~50ms
    if len(pulso) <= win_sg:
        return None
    suave = savgol_filter(pulso, win_sg, 3)
    apg = np.gradient(np.gradient(suave))

    metade = len(apg) // 2
    picos_pos, _ = find_peaks(apg[:metade])
    if len(picos_pos) == 0:
        return None
    a_idx = picos_pos[int(np.argmax(apg[picos_pos]))]
    a_val = apg[a_idx]
    if a_val <= 0:
        return None

    resto_neg, _ = find_peaks(-apg[a_idx:])
    if len(resto_neg) == 0:
        return None
    b_idx = a_idx + resto_neg[int(np.argmax(-apg[a_idx + resto_neg]))]
    b_val = apg[b_idx]

    lim_dic = min(len(apg), b_idx + int(0.45 * fs))

    def _prox_extremo(sig, start, fim):
        if start >= fim:
            return None
        p, _ = find_peaks(sig[start:fim])
        return start + p[0] if len(p) > 0 else None

    c_idx = _prox_extremo(apg, b_idx + 1, lim_dic)
    c_val = apg[c_idx] if c_idx is not None else np.nan
    d_idx = _prox_extremo(-apg, (c_idx + 1) if c_idx is not None else b_idx + 1, lim_dic)
    d_val = apg[d_idx] if d_idx is not None else np.nan
    e_idx = _prox_extremo(apg, (d_idx + 1) if d_idx is not None else b_idx + 1, lim_dic)
    e_val = apg[e_idx] if e_idx is not None else np.nan

    c0 = 0.0 if np.isnan(c_val) else c_val
    d0 = 0.0 if np.isnan(d_val) else d_val
    e0 = 0.0 if np.isnan(e_val) else e_val
    return {
        'apg_b_a': b_val / a_val,
        'apg_c_a': (c_val / a_val) if not np.isnan(c_val) else np.nan,
        'apg_d_a': (d_val / a_val) if not np.isnan(d_val) else np.nan,
        'apg_aging': (b_val - c0 - d0 - e0) / a_val,
    }


def largura_a(norm_p, fs, frac):
    """Largura (s) do pulso a `frac` da amplitude, como o w50: do primeiro ao último ponto >= frac. Devolve (largura, truncada),
    em que truncada = o último ponto acima do limiar é o último da janela (o pulso não desceu abaixo do limiar a tempo)."""
    idx = np.where(norm_p >= frac)[0]
    if len(idx) <= 1:
        return None, False
    return round(float((idx[-1] - idx[0]) / fs), 4), bool(idx[-1] >= len(norm_p) - 1)


def indice_reflexao(win, pk, vly, pe, limite, ac, fs):
    """(pico diastólico - pé) / (pico sistólico - pé). O pico diastólico é o máximo local depois do entalhe `pe` (fim do pulso do
    próprio pipeline) até 0,35 s depois ou ao pico seguinte, o que vier primeiro, com proeminência de pelo menos 2 % da amplitude.
    None se não houver pico distinto (frequente: a onda refletida pode ficar fundida na descida)."""
    if pe is None:
        return None
    fim = min(pe + int(0.35 * fs), limite, len(win))
    if fim - pe < 5:
        return None
    seg = win[pe:fim]
    p, props = find_peaks(seg, prominence=0.02 * ac)
    if len(p) == 0:
        return None
    dp = pe + int(p[int(np.argmax(props['prominences']))])
    ri = (float(win[dp]) - float(win[vly])) / ac
    return round(ri, 4) if 0.0 < ri < 1.0 else None


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

        # inclinação sistólica máxima (como no 09b: a do alpha_b, que o servidor não usava)
        deriv = np.diff(win[vly:pk + 1]) * fs
        max_slope = float(np.max(deriv)) if len(deriv) > 0 else 0.0

        limite = pk_next if pk_next is not None else min(pk + int(0.9 * fs), len(win))
        pe_ini = pk + int(0.04 * fs)
        pe_fim = min(pk + int(0.50 * fs), len(win), limite)
        pe = None
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
        w25 = w75 = None
        w25_t = w75_t = False
        if len(pulso) > 1 and (np.max(pulso) - np.min(pulso)) > 0.01:
            norm_p = (pulso - np.min(pulso)) / (np.max(pulso) - np.min(pulso))
            pulse_area = round(float(simpson(y=norm_p)), 4)
            idx50 = np.where(norm_p >= 0.5)[0]
            if len(idx50) > 1:
                w50 = round(float((idx50[-1] - idx50[0]) / fs), 4)
            w25, w25_t = largura_a(norm_p, fs, 0.25)
            w75, w75_t = largura_a(norm_p, fs, 0.75)
        ri = indice_reflexao(win, pk, vly, pe, limite, ac, fs)

        # APG: 2.ª derivada de um pulso completo a partir do pé, fim do pulso a 1,1 x o intervalo pico-a-pico
        if pk_next is not None:
            fim_pulso = min(vly + int(1.1 * (pk_next - pk)), len(win))
        else:
            fim_pulso = min(vly + int(1.1 * fs * 60.0 / (hr_mediana + 1e-9)), len(win))
        apg = extrair_apg_features(win[vly:fim_pulso], fs)

        batimentos.append({
            'hr': hr, 'crest_time': crest_time, 'decay_time': decay_time,
            'alpha': round(alpha, 4),
            'pulse_area': pulse_area, 'w50': w50,
            'ppg_valley_intensity': round(vly_val, 4),
            'ppg_ac_amplitude': round(ac, 4),
            'w25': w25, 'w75': w75, 'w25_trunc': float(w25_t) if w25 is not None else None,
            'w75_trunc': float(w75_t) if w75 is not None else None,
            'ppg_max_slope': round(max_slope, 4), 'ppg_max_slope_norm': round(max_slope / ac, 4),
            'reflection_index': ri,
            'apg_b_a': apg['apg_b_a'] if apg else None, 'apg_d_a': apg['apg_d_a'] if apg else None,
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
        # isfinite, como no 09: a APG devolve NaN (não None) quando não localiza as ondas c/d/e
        vals = [b[key] for b in kept if b.get(key) is not None and np.isfinite(b[key])]
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
        'w25': med('w25'), 'w75': med('w75'),
        'ppg_max_slope': med('ppg_max_slope'), 'ppg_max_slope_norm': med('ppg_max_slope_norm'),
        'reflection_index': med('reflection_index'),
        'apg_b_a': med('apg_b_a'), 'apg_d_a': med('apg_d_a'),
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
