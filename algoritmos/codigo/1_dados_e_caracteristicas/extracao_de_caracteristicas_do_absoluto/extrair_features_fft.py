import os
# 1 thread por worker (a FFT/scipy usa BLAS multi-thread; com N_JOBS=10 os workers
# competiriam). Tem de ser antes de importar numpy.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import pandas as pd
import wfdb
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.integrate import simpson
from scipy.stats import skew
from joblib import Parallel, delayed
import multiprocessing

# --- CONFIGURAÇÃO ---
# Variante "fft": reextrai as MESMAS janelas/batimentos do Teste 1 (baseline) e acrescenta
# ecg_fft_mag_00..23 (espetro de frequência do batimento) — ver extrair_espectro_fft_batimento().
PATH_DATASET = "/media/filipe/Dados/MIMIC_DATASET_CONSOLIDADO"   # onde está o dataset bruto (leitura)
INPUT_CSVS = [f"{PATH_DATASET}/dataset_para_extração_features.csv"]
PATH_FEATURES_OUT = "/media/filipe/Dados/Features"               # onde gravar as features (escrita)
WINDOW_SIZE_SEC = 15
STRIDE_SEC = 5  # avanço de 5 s: as janelas de 15 s sobrepõem-se

# Nº de processos paralelos. O pico de RAM ≈ N_JOBS × (registo maior a ser processado). Alguns
# registos têm >1.5 GB; usar todos os cores (16) fazia colidir vários registos grandes e esgotar a
# RAM. 8 é um compromisso seguro (≈ metade dos cores). Pode subir-se com RAM folgada
# e descer-se se voltar a estourar. `None` = todos os cores (comportamento antigo, arriscado com este dataset).
N_JOBS = 10

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def butter_highpass_filter(data, cutoff, fs, order=4):
    """Remove flutuações respiratórias da linha de base do ECG (Baseline Wander)"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def remover_picos_isolados(sinal, limiar=0.15):
    """Marca a amostra i só se AMBOS os vizinhos imediatos divergem muito dela mas concordam ENTRE si (pensado para falhas de UMA amostra)
    e substitui-a pela média dos vizinhos. Um filtro de mediana/MAD com janela maior cortaria os próprios picos R."""
    s = np.asarray(sinal, dtype=float)
    if len(s) < 3:
        return s
    d_ant = np.abs(s[1:-1] - s[:-2])
    d_prox = np.abs(s[1:-1] - s[2:])
    d_vizinhos = np.abs(s[:-2] - s[2:])
    glitch = (d_ant > limiar) & (d_prox > limiar) & (d_vizinhos < limiar * 0.5)
    limpo = s.copy()
    idx = np.where(glitch)[0] + 1
    limpo[idx] = (s[idx - 1] + s[idx + 1]) / 2
    return limpo

# Canal de ECG: EXCLUSIVAMENTE a derivação II (nome exato, não substring — 'V'/'I' soltos
# apanhavam CVP/ICP/IC2 por conterem essas letras). Nada de fallback para outras derivações: o
# setup real do projeto (AD8232, 3 elétrodos) mede Lead II, e treinar com a morfologia de QRS/T
# de outra derivação criaria uma discrepância treino↔uso real. Neste dataset todos os 9646
# registos têm II, por isso esta restrição não perde nenhum paciente.
def selecionar_ecg(sig_names):
    return sig_names.index('II') if 'II' in sig_names else None

def extrair_targets_abp_robusto(win_abp, fs):
    sbp_idxs, _ = find_peaks(win_abp, distance=int(fs*0.4), height=np.percentile(win_abp, 50))
    dbp_idxs, _ = find_peaks(-win_abp, distance=int(fs*0.4))

    if len(sbp_idxs) < 3 or len(dbp_idxs) < 3:
        return None, None, None, None

    sbp_vals = win_abp[sbp_idxs]
    sbp_mean = np.mean(sbp_vals)
    sbp_std = np.std(sbp_vals)
    dbp_mean = np.mean(win_abp[dbp_idxs])
    pulse_pressure = sbp_mean - dbp_mean

    return sbp_mean, dbp_mean, sbp_std, pulse_pressure


# ========== BEAT VAZIO ==========
def beat_vazio(r, q, s, t):
    return {
        'r_idx': int(r), 'r_next_idx': None, 'q_idx': q, 's_idx': s, 't_idx': t, 'qn_idx': None,
        'ppg_pk_idx': None, 'ppg_vly_idx': None, 'pat': None, 'hr': None, 'pat_kept': False, 'qrs_ok': False,
        'qrs_duration': None, 'r_amplitude': None, 'qrs_amplitude': None, 'qrs_area': None,
        'qt_interval': None, 'tq_interval': None, 'qtc_interval': None, 'sdi_interaction': None,
        'crest_time': None, 'decay_time': None, 'alpha': None, 'alpha_b': None, 'pulse_area': None, 'w50': None,
        'ppg_valley_intensity': None, 'ppg_ac_amplitude': None
    }


# ========== PIPELINE TESTE 1 (SEM NEUROKIT) — partilhado entre baseline/com_sqi/sem_sqi ==========
def processar_janela_teste1(win_ecg, win_ppg, fs):
    """Processa UMA janela com a lógica do Teste 1 (scipy, sem NeuroKit).
        win_ecg : ECG já filtrado (highpass 0.5 Hz, remove baseline wander)
        win_ppg : PPG já filtrado (lowpass 8 Hz)
    Devolve (beats, features, win_ecg, win_ppg). Os índices dos batimentos são relativos à janela.
    'r_next_idx' guarda o fim do pulso PPG (pe)."""
    win_ppg_pos = win_ppg - np.min(win_ppg) + 0.01   # positividade p/ alpha (não move os índices)

    # Deteção de picos R — abordagem HÍBRIDA (amplitude do ECG + envelope de Pan-Tompkins):
    #  • Candidatos por AMPLITUDE no próprio ECG (picos positivos ≥15% do máximo). Isto apanha
    #    de forma fiável todos os R genuínos, mesmo os de QRS menos afiado — um detetor puramente
    #    baseado no envelope (threshold global 0.3*max) perdia R verdadeiros quando o máximo do
    #    envelope estava inflado por uma onda S muito abrupta noutro ponto da janela.
    #  • O envelope (derivada²+integração ~80ms, realça a subida/descida rápida do QRS e suprime a
    #    onda T, mais lenta) serve para DESEMPATAR: quando dois candidatos ficam a <0.4s, fica o de
    #    maior envelope local (o R, sempre mais afiado) — mesmo que a onda T seja MAIS ALTA que o R
    #    (R atenuado). Assim resolve-se o problema clássico de marcar a T como R sem usar amplitude.
    deriv = np.diff(win_ecg, prepend=win_ecg[0]) * fs
    win_integ = max(1, int(0.08 * fs))
    envelope = np.convolve(deriv ** 2, np.ones(win_integ) / win_integ, mode='same')

    def _env_local(p):
        lo, hi = max(0, p - int(0.05 * fs)), min(len(envelope), p + int(0.05 * fs))
        return np.max(envelope[lo:hi])

    candidatos, _ = find_peaks(win_ecg, height=0.15 * np.max(win_ecg))
    r_peaks = []
    for p in candidatos:
        if r_peaks and (p - r_peaks[-1]) < int(fs * 0.4):
            if _env_local(p) > _env_local(r_peaks[-1]):
                r_peaks[-1] = p
        else:
            r_peaks.append(int(p))
    r_peaks = np.array(r_peaks)
    if len(r_peaks) < 8:
        return None

    # Piso de amplitude mínima do R — o threshold relativo acima sempre encontra um "top 15%"
    # mesmo em ruído puro, porque é relativo ao próprio sinal. Um R genuíno (highpass 0.5Hz) tem
    # tipicamente amplitude bem acima de 0.1; abaixo de ~0.08 já não se distingue do ruído de
    # fundo, mesmo que a forma/distância pareçam plausíveis.
    amp_r_mediana = np.median(win_ecg[r_peaks])
    if amp_r_mediana < 0.08:
        return None

    # Segunda camada: a deteção híbrida encontra sempre "picos" mesmo em ruído puro, porque o
    # threshold é relativo ao máximo da janela — em ruído esses picos podem empurrar
    # amp_r_mediana para pouco acima do piso absoluto. A irregularidade do RR é o sinal mais
    # fiável de que não é um ritmo cardíaco real: batimentos genuínos têm RR consistente (mesmo
    # com arritmia leve, o CV é baixo); o ruído puro dá um CV muito maior. Só rejeita se
    # amplitude AINDA modesta (<0.15) E RR muito irregular — amplitude claramente forte não é
    # descartada só por variabilidade de RR mais alta.
    rr = np.diff(r_peaks) / fs
    cv_rr = np.std(rr) / np.mean(rr)
    if amp_r_mediana < 0.15 and cv_rr > 0.25:
        return None

    # Piso contra excursões anómalas na janela (artefacto de movimento/elétrodo — não é glitch de
    # 1 amostra, dura dezenas de ms, por isso `remover_picos_isolados` não o apanha): se algum
    # ponto da janela filtrada ultrapassa 4x a amplitude R típica DESTA janela, não é fisiológico.
    # Relativo à própria janela, não um valor fixo, porque a amplitude varia por registo/ganho.
    if np.max(np.abs(win_ecg)) > 4 * amp_r_mediana:
        return None

    prom_min = (np.max(win_ppg_pos) - np.min(win_ppg_pos)) * 0.15
    ppg_peaks_all, _ = find_peaks(win_ppg_pos, distance=int(fs * 0.30), prominence=prom_min)  # picos sistólicos globais
    rho_blood = 1060
    beats = []
    num_r = len(r_peaks)

    for i in range(num_r):
        r_now  = int(r_peaks[i])
        r_next = int(r_peaks[i + 1]) if i + 1 < num_r else None

        # --- Fiduciais ECG (Teste 1: argmin/argmax em janelas fixas) ---
        q_start = max(0, r_now - int(0.06 * fs))
        q_idx = int(q_start + np.argmin(win_ecg[q_start:r_now])) if q_start < r_now else r_now

        s_end_local = min(r_now + int(0.08 * fs), len(win_ecg))
        s_idx = int(r_now + np.argmin(win_ecg[r_now:s_end_local])) if r_now < s_end_local else r_now

        t_idx = None
        if r_next is not None:
            # margem antes do r_next (igual à do qn_idx) para o T nunca invadir a subida do QRS seguinte
            t_start = r_now + int(0.12 * fs)
            t_end = min(r_now + int(0.45 * fs), r_next - int(0.08 * fs))
            if t_start < t_end and t_start < len(win_ecg):
                t_idx = int(t_start + np.argmax(win_ecg[t_start:t_end]))

        qn_idx = None
        if r_next is not None:
            qn_start = max(r_now, r_next - int(0.06 * fs))
            if qn_start < r_next and qn_start < len(win_ecg):
                qn_idx = int(qn_start + np.argmin(win_ecg[qn_start:r_next]))

        # --- QRS ---
        qrs_duration = qrs_amplitude = qrs_area = r_amplitude = None
        qrs_ok = False
        if q_idx < s_idx < len(win_ecg):
            qrs_duration  = round((s_idx - q_idx) / fs, 4)
            qrs_ok        = 0.04 < qrs_duration < 0.16
            r_amplitude   = round(float(win_ecg[r_now]), 4)
            qrs_amplitude = round(float(win_ecg[r_now] - min(win_ecg[q_idx], win_ecg[s_idx])), 4)
            seg = win_ecg[q_idx:s_idx + 1] - win_ecg[q_idx]
            qrs_area = round(float(simpson(y=seg)), 4) if len(seg) > 1 else 0.0

        # --- QT / TQ / QTc / SDI ---
        qt_interval = tq_interval = qtc_interval = sdi_interaction = None
        if t_idx is not None and q_idx < t_idx < len(win_ecg):
            qt_interval = round((t_idx - q_idx) / fs, 4)
            if r_next is not None:
                rr_sec = (r_next - r_now) / fs
                qtc_interval = round(qt_interval / np.sqrt(rr_sec + 1e-6), 4)
        if t_idx is not None and qn_idx is not None and t_idx < qn_idx < len(win_ecg):
            tq_interval = round((qn_idx - t_idx) / fs, 4)
        if qt_interval is not None and tq_interval is not None:
            sdi_interaction = round(min(qt_interval / (tq_interval + 1e-6), 10.0), 4)

        # --- HR ---
        hr = round((60 * fs) / (r_next - r_now), 2) if r_next is not None else None

        # --- Pico sistólico PPG: 1º pico global depois do R (robusto a PAT curto/longo) ---
        lim = r_next if r_next is not None else min(r_now + int(0.9 * fs), len(win_ppg_pos))
        cand = ppg_peaks_all[(ppg_peaks_all > r_now) & (ppg_peaks_all <= lim)]
        ppg_pk = int(cand[0]) if len(cand) > 0 else None

        if ppg_pk is None:
            beats.append(beat_vazio(r_now, q_idx, s_idx, t_idx))
            continue

        pat = round((ppg_pk - r_now) / fs, 4)

        # --- Vale/pé PPG: mínimo ANTES do pico, limitado pelo pico PPG anterior (nunca pelo R) ---
        # Usar r_now como fronteira de início falha nos dois sentidos: com PAT curto, r_now cai
        # DEPOIS do pé verdadeiro (já na subida) e a busca nunca o alcança; com PAT grande
        # (latência do aparelho — ver janela de PAT adaptativa abaixo), r_now cai cedo demais e a
        # busca arrisca apanhar o vale de um ciclo anterior. O pé pertence à estrutura do PRÓPRIO PPG.
        picos_antes = ppg_peaks_all[ppg_peaks_all < ppg_pk]
        pe_ini = max(int(picos_antes[-1]) if len(picos_antes) > 0 else 0, ppg_pk - int(0.4 * fs), 0)
        ppg_vly = pe_ini + int(np.argmin(win_ppg_pos[pe_ini:ppg_pk])) if pe_ini < ppg_pk else max(0, ppg_pk - int(0.2 * fs))

        ppg_valley_intensity = float(win_ppg_pos[ppg_vly])
        ppg_ac_amplitude     = float(win_ppg_pos[ppg_pk] - win_ppg_pos[ppg_vly])
        omega     = (hr / 60.0) if hr is not None else 1.2
        alpha     = float(ppg_valley_intensity * np.sqrt((rho_blood * omega) / (ppg_ac_amplitude + 1e-6)))
        upslope   = win_ppg_pos[ppg_vly:ppg_pk + 1]
        deriv_ppg = np.diff(upslope) * fs
        max_slope = float(np.max(deriv_ppg)) if len(deriv_ppg) > 0 else 0.0
        b_coef    = 1.0 / (max_slope + 1e-6)
        alpha_b   = float(ppg_valley_intensity * np.sqrt((omega * rho_blood) / (b_coef + 1e-6)))

        crest_time = round((ppg_pk - ppg_vly) / fs, 4)

        # --- Fim do pulso PPG (Teste 1: primeiro mínimo local após o pico) ---
        # Limita a busca pelo PRÓXIMO PICO PPG (não pelo próximo R): usar r_next assumia PAT
        # curto, mas o PAT observado neste dataset pode ocupar quase todo o RR (latência do
        # aparelho de oximetria — ver janela de PAT adaptativa abaixo), o que deixava ppg_pk muito
        # perto de r_next e esganava esta busca (entalhe a colapsar em cima do próprio pico).
        prox_pico_ppg = ppg_peaks_all[ppg_peaks_all > ppg_pk]
        limite_pulso = int(prox_pico_ppg[0]) if len(prox_pico_ppg) > 0 else min(ppg_pk + int(0.9 * fs), len(win_ppg_pos))
        pe_start = ppg_pk + int(0.04 * fs)
        pe_end   = min(ppg_pk + int(0.50 * fs), len(win_ppg_pos), limite_pulso)
        if pe_start < pe_end:
            seg_pe = win_ppg_pos[pe_start:pe_end]
            mins, _ = find_peaks(-seg_pe, distance=int(fs * 0.05))
            pe = int(pe_start + mins[0]) if len(mins) > 0 else int(pe_start + np.argmin(seg_pe))
            decay_time = round((pe - ppg_pk) / fs, 4)
        else:
            pe = ppg_pk
            decay_time = None

        # --- Geometria do pulso ---
        p_low, p_high = max(0, ppg_pk - int(0.1 * fs)), min(len(win_ppg_pos), ppg_pk + int(0.3 * fs))
        pulso = win_ppg_pos[p_low:p_high]
        pulse_area = w50 = None
        if len(pulso) > 1 and (np.max(pulso) - np.min(pulso)) > 0.01:
            norm_p = (pulso - np.min(pulso)) / (np.max(pulso) - np.min(pulso))
            pulse_area = round(float(simpson(y=norm_p)), 4)
            idx50 = np.where(norm_p >= 0.5)[0]
            if len(idx50) > 1:
                w50 = round(float((idx50[-1] - idx50[0]) / fs), 4)

        # pat_kept definido depois do loop (janela de PAT adaptativa) — placeholder aqui
        pat_kept = False

        beats.append({
            'r_idx': r_now, 'r_next_idx': int(pe), 'q_idx': q_idx, 's_idx': s_idx, 't_idx': t_idx, 'qn_idx': qn_idx,
            'ppg_pk_idx': ppg_pk, 'ppg_vly_idx': ppg_vly, 'pat': pat, 'hr': hr,
            'pat_kept': pat_kept, 'qrs_ok': qrs_ok, 'qrs_duration': qrs_duration, 'r_amplitude': r_amplitude,
            'qrs_amplitude': qrs_amplitude, 'qrs_area': qrs_area, 'qt_interval': qt_interval, 'tq_interval': tq_interval,
            'qtc_interval': qtc_interval, 'sdi_interaction': sdi_interaction, 'crest_time': crest_time, 'decay_time': decay_time,
            'alpha': round(alpha, 4), 'alpha_b': round(alpha_b, 4), 'pulse_area': pulse_area, 'w50': w50,
            'ppg_valley_intensity': round(ppg_valley_intensity, 4), 'ppg_ac_amplitude': round(ppg_ac_amplitude, 4)
        })

    # --- Janela de PAT adaptativa por registo (substitui o intervalo fisiológico fixo 0.19-0.50s) ---
    # O PLETH tem latência de processamento própria do aparelho (média de vários batimentos p/
    # estabilizar o SpO2), consistente DENTRO do mesmo registo mas muito variável ENTRE registos —
    # confirmado por cross-check com o atraso R->ABP (medição direta, sem processamento de oxímetro),
    # que é apertado e estável, ao contrário do PAT do PPG cru. Em vez de um intervalo global fixo, aceitamos o agrupamento dominante e apertado
    # DESTE registo — seja qual for o valor absoluto — em bins de 40ms.
    pats_hr_ok = np.array([
        b['pat'] for b in beats
        if b['pat'] is not None and b['hr'] is not None and 45 < b['hr'] < 140
    ])
    if len(pats_hr_ok) >= 6:
        bins = np.arange(0, pats_hr_ok.max() + 0.04, 0.04)
        hist, edges = np.histogram(pats_hr_ok, bins=bins)
        i_max = int(np.argmax(hist))
        cl_min, cl_max = edges[i_max] - 0.04, edges[i_max + 1] + 0.04
        for b in beats:
            if b['pat'] is not None and b['hr'] is not None and 45 < b['hr'] < 140:
                b['pat_kept'] = bool(cl_min <= b['pat'] <= cl_max)

    # --- Filtragem IQR sobre os PATs aceites (Teste 1: remove outliers batimento a batimento) ---
    pats = np.array([b['pat'] for b in beats if b['pat'] is not None and b['pat_kept']])
    if len(pats) >= 4:
        q1, q3 = np.percentile(pats, [25, 75])
        iqr = (q3 - q1) if (q3 - q1) != 0 else 0.05
        lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for b in beats:
            if b['pat'] is not None and b['pat_kept']:
                b['pat_kept'] = bool(lim_inf <= b['pat'] <= lim_sup)

    kept = [b for b in beats if b['pat_kept'] and b['pat'] is not None]
    kept_pats = np.array([b['pat'] for b in kept])
    # Teste 1 exige >= 6 batimentos limpos E baixa dispersão do PAT
    if len(kept) < 6 or np.std(kept_pats) >= 0.05:
        return beats, None, win_ecg, win_ppg

    def med(lst, key):
        vals = [b[key] for b in lst if b[key] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    features = {
        'pat_mean': med(kept, 'pat'), 'hr_mean': med(kept, 'hr'),
        'qrs_duration_mean': med([b for b in kept if b['qrs_ok']], 'qrs_duration'),
        'r_amplitude_mean':  med([b for b in kept if b['qrs_ok']], 'r_amplitude'),
        'qrs_amplitude_mean':med([b for b in kept if b['qrs_ok']], 'qrs_amplitude'),
        'qrs_area_mean':     med([b for b in kept if b['qrs_ok']], 'qrs_area'),
        'qt_interval_mean':  med([b for b in kept if b['qt_interval']  is not None], 'qt_interval'),
        'tq_interval_mean':  med([b for b in kept if b['tq_interval']  is not None], 'tq_interval'),
        'qtc_interval_mean': med([b for b in kept if b['qtc_interval'] is not None], 'qtc_interval'),
        'sdi_interaction_mean': med([b for b in kept if b['sdi_interaction'] is not None], 'sdi_interaction'),
        'crest_time_mean':   med(kept, 'crest_time'), 'decay_time_mean': med(kept, 'decay_time'),
        'alpha_mean':        med(kept, 'alpha'), 'alpha_b_mean': med(kept, 'alpha_b'),
        'pulse_area_mean':   med([b for b in kept if b['pulse_area'] is not None], 'pulse_area'),
        'w50_mean':          med([b for b in kept if b['pulse_area'] is not None], 'w50'),
        'ppg_valley_intensity_mean': med(kept, 'ppg_valley_intensity'),
        'ppg_ac_amplitude_mean':     med(kept, 'ppg_ac_amplitude'),
    }
    return beats, features, win_ecg, win_ppg


def extrair_espectro_fft_batimento(win_ecg, beats, fs, n_coefs=24, beat_len=128):
    """
    Feature vector no domínio da frequência por batimento ("frequency whole-based",
    Mousavi et al. 2018, ICCKE — BP a partir só do espectro do ECG): para cada
    batimento aceite (pat_kept — mesmo critério de qualidade já usado no Teste 1),
    extrai o segmento ECG entre dois R consecutivos, reamostra-o para um comprimento
    fixo (`beat_len`, independente da FC) e normaliza a amplitude (z-score do próprio
    segmento) ANTES da FFT. Essa normalização remove a dependência do ganho/contacto
    do elétrodo — a mesma invariância que Treebupachatsakul et al. 2022 obtêm por
    randomização de amplitude, só que aplicada diretamente em vez de via aumento de
    dados (não há "época de treino" nesta extração tabular offline).

    Devolve a média do módulo dos primeiros `n_coefs` coeficientes da FFT entre os
    batimentos aceites da janela (mais robusto a um único batimento ruidoso do que usar
    só um) — não os 512 pontos + PCA do paper original: com modelos de árvores (XGB) um
    espectro truncado e compacto, já concentrado na banda de energia do QRS/T, evita
    colinearidade excessiva sem precisar de um PCA à parte.
    """
    r_idxs = [b['r_idx'] for b in beats]
    espectros = []
    for i in range(len(beats) - 1):
        if not beats[i].get('pat_kept'):
            continue
        r_now, r_next = r_idxs[i], r_idxs[i + 1]
        if r_next <= r_now or r_next > len(win_ecg):
            continue
        segmento = win_ecg[r_now:r_next]
        if len(segmento) < 8:
            continue
        x_orig = np.linspace(0.0, 1.0, len(segmento))
        x_novo = np.linspace(0.0, 1.0, beat_len)
        segmento_resample = np.interp(x_novo, x_orig, segmento)
        segmento_norm = (segmento_resample - np.mean(segmento_resample)) / (np.std(segmento_resample) + 1e-8)
        espectro = np.abs(np.fft.rfft(segmento_norm))
        espectros.append(espectro[:n_coefs])

    if len(espectros) < 3:
        return None
    return np.mean(espectros, axis=0)


def processar_linha_individual(row, lock=None):
    """
    Função atómica que processa apenas UMA linha do CSV original.
    Distribuída em paralelo por múltiplos cores.
    """
    s_id = str(row['subject_folder'])

    # --- DETEÇÃO AUTOMÁTICA DE DATASET ---
    if len(s_id) == 7:
        dataset_label = "MIMIC-III"
        canais_abp, canais_ppg = ['ABP', 'ART'], ['PLETH', 'PPG']
    elif len(s_id) == 9:
        dataset_label = "MIMIC-IV"
        canais_abp, canais_ppg = ['ABP'], ['PLETH']
    else:
        return 0

    seg_name = str(row['segment'])
    try:
        seg_order = int(seg_name.split('_')[-1])
    except:
        seg_order = 0

    base_path = row['absolute_path'].replace('.dat', '').replace('.hea', '')
    if not os.path.exists(base_path + '.hea'):
        return 0

    try:
        # MEMÓRIA: ler primeiro só o HEADER (barato) para descobrir os canais, e depois carregar
        # APENAS os 3 canais necessários (ECG/PPG/ABP) em vez do registo inteiro. Alguns registos
        # têm 6-8 canais e até 75h de duração (>1.5 GB carregados por inteiro em float64); carregar
        # só 3 canais + converter para float32 reduz o pico de RAM por worker para <1/4. Sem isto,
        # vários workers a apanhar registos grandes ao mesmo tempo esgotavam os 30 GB (OOM/crash).
        hdr = wfdb.rdheader(base_path)
        fs = hdr.fs
        sig_names = [s.upper() for s in hdr.sig_name]

        idx_ecg = selecionar_ecg(sig_names)
        idx_ppg = next((i for i, s in enumerate(sig_names) if any(x in s for x in canais_ppg)), None)
        idx_abp = next((i for i, s in enumerate(sig_names) if any(x in s for x in canais_abp)), None)
        if idx_ecg is None or idx_ppg is None or idx_abp is None:
            return 0

        record = wfdb.rdrecord(base_path, channels=[idx_ecg, idx_ppg, idx_abp])
        ecg_raw = record.p_signal[:, 0].astype(np.float32)
        ppg_raw = record.p_signal[:, 1].astype(np.float32)
        abp_raw = record.p_signal[:, 2].astype(np.float32)
        del record  # liberta a matriz float64 devolvida pelo wfdb assim que temos as cópias float32

        if np.isnan(ecg_raw).any() or np.isnan(ppg_raw).any() or np.isnan(abp_raw).any():
            ecg_raw = pd.Series(ecg_raw).interpolate().ffill().bfill().values
            ppg_raw = pd.Series(ppg_raw).interpolate().ffill().bfill().values
            abp_raw = pd.Series(abp_raw).interpolate().ffill().bfill().values

        ecg_raw = remover_picos_isolados(ecg_raw)  # suaviza amostras isoladas antes do highpass

        # Filtragem Biomédica dos Sinais
        ppg_filt = butter_lowpass_filter(ppg_raw, 8, fs)
        ecg_filt = butter_highpass_filter(ecg_raw, 0.5, fs)
        # Correção de inversão: se skew(ECG)<0 o sinal está invertido → repõe R para cima
        if skew(ecg_filt) < 0:
            ecg_filt = -ecg_filt
    except:
        return 0

    resultados_da_linha = []

    # --- Janelas deslizantes com sobreposição ---
    samples_per_win = int(WINDOW_SIZE_SEC * fs)
    step_samples = int(STRIDE_SEC * fs)
    num_windows = int((len(ecg_filt) - samples_per_win) / step_samples) + 1

    for w in range(num_windows):
        start = w * step_samples
        end = start + samples_per_win

        win_ecg, win_ppg, win_abp = ecg_filt[start:end], ppg_filt[start:end], abp_raw[start:end]

        if np.std(win_ppg) < 0.005 or (np.max(win_abp) - np.min(win_abp)) < 15:
            continue

        # Gate ABP (targets fisiológicos) — computado cedo, antes do processamento pesado de batimentos
        sbp, dbp, sbp_std, pp = extrair_targets_abp_robusto(win_abp, fs)
        if sbp is None or not (pp > 25 and sbp_std < 12 and 65 < sbp < 195 and 35 < dbp < 115):
            continue

        res = processar_janela_teste1(win_ecg, win_ppg, fs)
        if res is None or res[1] is None:
            continue

        beats, features, _, _ = res
        espectro = extrair_espectro_fft_batimento(win_ecg, beats, fs)
        if espectro is None:
            continue

        linha = {
            'subject_id': s_id,
            'segment_name': seg_name,
            'segment_order': seg_order,
            'window_idx': w,
            'dataset': dataset_label,
            'gender': row['gender'],
            'age': row['age'],
            'pat_mean': features['pat_mean'],
            'hr_mean': features['hr_mean'],
            'qrs_duration': features['qrs_duration_mean'],
            'r_amplitude': features['r_amplitude_mean'],
            'qrs_amplitude': features['qrs_amplitude_mean'],
            'qrs_area': features['qrs_area_mean'],
            'w50': features['w50_mean'],
            'area_mean': features['pulse_area_mean'],
            'ppg_valley_intensity': features['ppg_valley_intensity_mean'],
            'ppg_ac_amplitude': features['ppg_ac_amplitude_mean'],
            'ppg_crest_time': features['crest_time_mean'],
            'ppg_decay_time': features['decay_time_mean'],
            'alpha_womersley': features['alpha_mean'],
            'alpha_womersley_b': features['alpha_b_mean'],
            'qt_interval': features['qt_interval_mean'],
            'tq_interval': features['tq_interval_mean'],
            'qtc_interval': features['qtc_interval_mean'],
            'sdi_interaction': features['sdi_interaction_mean'],
            'sbp_target': round(sbp, 2),
            'dbp_target': round(dbp, 2)
        }
        linha.update({f'ecg_fft_mag_{i:02d}': round(float(v), 5) for i, v in enumerate(espectro)})
        resultados_da_linha.append(linha)

    # Escrita INCREMENTAL por segmento (resume-safe) -- em vez do padrão antigo
    # de acumular tudo em memória e escrever no fim (pico de RAM de vários GB +
    # tudo perdido se crashar). `lock` evita corrida entre processos.
    if resultados_da_linha and lock is not None:
        df_linha = pd.DataFrame(resultados_da_linha)
        os.makedirs(PATH_FEATURES_OUT, exist_ok=True)
        with lock:
            for g, label in [('M', 'homens'), ('F', 'mulheres')]:
                subset = df_linha[df_linha['gender'] == g]
                if not subset.empty:
                    caminho = f"{PATH_FEATURES_OUT}/features_{label}_fft.csv"
                    subset.to_csv(caminho, mode='a', header=not os.path.exists(caminho), index=False)
    return len(resultados_da_linha)


def _obter_ja_processados():
    ja = set()
    for label in ['homens', 'mulheres']:
        caminho = f"{PATH_FEATURES_OUT}/features_{label}_fft.csv"
        if os.path.exists(caminho):
            existentes = pd.read_csv(caminho, usecols=['subject_id', 'segment_name'])
            ja |= set(zip(existentes['subject_id'].astype(str), existentes['segment_name'].astype(str)))
    return ja


def processar_datasets_hibridos():
    for csv_file in INPUT_CSVS:
        if not os.path.exists(csv_file):
            print(f"Aviso: {csv_file} não encontrado.")
            continue

        df = pd.read_csv(csv_file)
        n0 = len(df)
        ja = _obter_ja_processados()
        if ja:
            chave = pd.Series(list(zip(df['subject_folder'].astype(str), df['segment'].astype(str))), index=df.index)
            df = df[~chave.isin(ja)].reset_index(drop=True)
            print(f"↩️  A retomar: {len(ja)} segmentos já gravados -- {n0} -> {len(df)} por processar.")
        print(f"A processar {len(df)} registos em: {csv_file}")

        num_cores = N_JOBS if N_JOBS is not None else multiprocessing.cpu_count()
        print(f"{num_cores} cores (de {multiprocessing.cpu_count()}).")

        lock = multiprocessing.Manager().Lock()
        contagens = Parallel(n_jobs=num_cores, backend="multiprocessing", verbose=10, max_nbytes=None)(
            delayed(processar_linha_individual)(row, lock) for _, row in df.iterrows()
        )
        print(f"Concluído! Janelas: {sum(contagens)} (features_*_fft.csv)")


if __name__ == "__main__":
    processar_datasets_hibridos()
