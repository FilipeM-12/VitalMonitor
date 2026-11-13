import os
import pandas as pd
import wfdb
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from scipy.integrate import simpson
from scipy.stats import skew
from joblib import Parallel, delayed
import multiprocessing

# --- CONFIGURAÇÃO ---
PATH_DATASET = "/media/filipe/Dados/MIMIC_DATASET_CONSOLIDADO"   # onde está o dataset bruto (leitura)
INPUT_CSVS = [f"{PATH_DATASET}/dataset_para_extração_features.csv"]
PATH_FEATURES_OUT = "/media/filipe/Dados/Features"               # onde gravar as features (escrita)
WINDOW_SIZE_SEC = 15
STRIDE_SEC = 5

# Nº de processos paralelos. Com o processamento em blocos (CHUNK_LIMIAR_MIN) o pico de memória
# por worker fica limitado a ~1-1.5GB mesmo nos segmentos gigantes (546 min: ~950MB). Ajustar ao número de
# núcleos e à memória da máquina.
N_JOBS = 10


# ========== FILTROS ==========
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data)

def butter_highpass_filter(data, cutoff, fs, order=4):
    """Remove flutuações respiratórias da linha de base do ECG (Baseline Wander)"""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='high', analog=False)
    return filtfilt(b, a, data)


# ========== DESPIKE + CANAL ECG ==========
def remover_picos_isolados(sinal, limiar=0.15):
    """Marca a amostra i só se AMBOS os vizinhos imediatos divergem muito dela mas concordam ENTRE si (pensado para falhas de UMA amostra)
    e substitui-a pela média dos vizinhos. Aplicado antes de QUALQUER limpeza (inclui a própria nk.ecg_clean), porque os filtros de
    limpeza do NeuroKit não visam este tipo de falha."""
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


# ========== TARGETS ABP ==========
def extrair_targets_abp_robusto(win_abp, fs):
    sbp_idxs, _ = find_peaks(win_abp, distance=int(fs * 0.4), height=np.percentile(win_abp, 50))
    dbp_idxs, _ = find_peaks(-win_abp, distance=int(fs * 0.4))
    if len(sbp_idxs) < 3 or len(dbp_idxs) < 3:
        return 0, 0, 0, 0
    sbp_vals  = win_abp[sbp_idxs]
    sbp_mean  = np.mean(sbp_vals)
    sbp_std   = np.std(sbp_vals)
    dbp_mean  = np.mean(win_abp[dbp_idxs])
    pp        = sbp_mean - dbp_mean
    return sbp_mean, dbp_mean, sbp_std, pp


# ========== VALLEY: último mínimo local antes do pico sistólico ==========
def encontrar_valley(win_ppg, ppg_pk, nk_idx, ppg_peaks_win, nk_ppg_onsets_win, fs):
    """
    Procura o foot do pulso PPG como o último mínimo local imediatamente
    antes do pico sistólico, evitando capturar o vale do pico diastólico anterior.
    Todos os índices são relativos à janela.
    """
    ppg_vly = nk_ppg_onsets_win[nk_idx] if 0 <= nk_idx < len(nk_ppg_onsets_win) else None

    if ppg_vly is None or ppg_vly >= ppg_pk:
        # Busca desde o pico PPG anterior até ao pico atual
        if nk_idx > 0:
            search_start = ppg_peaks_win[nk_idx - 1]
        else:
            search_start = max(0, ppg_pk - int(0.8 * fs))
        search_start = max(0, search_start)

        if search_start < ppg_pk:
            seg = win_ppg[search_start:ppg_pk]
            # Último mínimo local = foot da subida sistólica
            local_mins, _ = find_peaks(-seg, distance=int(0.1 * fs))
            if len(local_mins) > 0:
                ppg_vly = int(search_start + local_mins[-1])
            else:
                restricted = max(search_start, ppg_pk - int(0.35 * fs))
                ppg_vly = int(restricted + np.argmin(win_ppg[restricted:ppg_pk]))
        else:
            ppg_vly = max(0, ppg_pk - int(0.15 * fs))

    return ppg_vly


# ========== PE CORRIGIDO: primeiro mínimo local após o pico sistólico ==========
def encontrar_pe(win_ppg, ppg_pk, nk_idx, ppg_peaks_win, nk_ppg_notches_win, r_next, fs):
    """
    Procura o fim do pulso PPG como o primeiro mínimo local imediatamente
    após o pico sistólico (notch dicrotico / início da diástole).
    Todos os índices são relativos à janela.
    """
    pe = nk_ppg_notches_win[nk_idx] if 0 <= nk_idx < len(nk_ppg_notches_win) else None

    if pe is None or pe <= ppg_pk:
        search_end = r_next if r_next is not None else min(ppg_pk + int(0.8 * fs), len(win_ppg) - 1)
        search_end = min(search_end, len(win_ppg) - 1)

        if ppg_pk < search_end:
            seg = win_ppg[ppg_pk:search_end]
            # Primeiro mínimo local = notch dicrotico / vale diastólico
            local_mins, _ = find_peaks(-seg, distance=int(0.1 * fs))
            if len(local_mins) > 0:
                pe = int(ppg_pk + local_mins[0])
            else:
                restricted = min(ppg_pk + int(0.35 * fs), search_end)
                pe = int(ppg_pk + np.argmin(win_ppg[ppg_pk:restricted]))
        else:
            pe = min(ppg_pk + int(0.25 * fs), len(win_ppg) - 1)

    return pe


# ========== FUNÇÃO PRINCIPAL (por linha do CSV) ==========

# ========== FEATURES DERIVATIVAS APG (2ª derivada do PPG) ==========
def extrair_apg_features(pulso, fs):
    """Ondas a,b,c,d,e da 2ª derivada (APG) de UM pulso PPG e os rácios b/a, c/a,
    d/a + índice de envelhecimento (b-c-d-e)/a. Método Elgendi/Takazawa: NÃO precisa
    do pé nem do entalhe dicrótico (mais robusto que crest/decay times). None se não
    fiável."""
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

    # c,d,e na região dicrótica (até ~0.45s após b), sinais alternados esperados
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


# ========== PROCESSAMENTO EM BLOCOS (segmentos gigantes, ex. 546 min) ==========
# nk.ecg_clean/ecg_peaks/ecg_delineate(dwt) e nk.ppg_process criam arrays intermédios
# bem maiores que o sinal de entrada (várias escalas de wavelet, buffers internos).
# Num segmento de 500+ min isso já estourou a RAM com 4 workers em simultâneo (medido:
# workers a 2-6GB cada, swap 100% cheio). Acima de CHUNK_LIMIAR_MIN processa-se em
# blocos de CHUNK_MIN com sobreposição de OVERLAP_SEC (para os filtros/deteção não
# perderem contexto na fronteira) -- os R/Q/S/T da zona de overlap são descartados,
# só ficam os do "núcleo" de cada bloco, para não duplicar batimentos.
CHUNK_LIMIAR_MIN = 90    # só ativa blocos acima disto (a maioria dos segmentos fica no caminho normal)
CHUNK_MIN = 60
OVERLAP_SEC = 30


def _limpar_delinear_bloco_ecg(ecg_bloco, fs, nk):
    """ecg_clean + ecg_peaks + ecg_delineate(dwt) num bloco. Devolve
    (ecg_limpo, r_local, q_local, s_local, t_local) -- tudo em índices LOCAIS ao bloco."""
    limpo = nk.ecg_clean(ecg_bloco, sampling_rate=fs, method="neurokit")
    _, r_info = nk.ecg_peaks(limpo, sampling_rate=fs, method="neurokit")
    r_local = r_info['ECG_R_Peaks']
    try:
        _, waves = nk.ecg_delineate(limpo, r_local, sampling_rate=fs, method="dwt")
        def _safe(k):
            raw = waves.get(k, [])
            return [int(x) if (x is not None and not np.isnan(x)) else None for x in raw]
        q_local, s_local, t_local = _safe('ECG_Q_Peaks'), _safe('ECG_S_Peaks'), _safe('ECG_T_Peaks')
    except Exception:
        n = len(r_local)
        q_local = s_local = t_local = [None] * n
    return limpo, r_local, q_local, s_local, t_local


def processar_ecg_em_blocos(ecg_raw, fs, nk):
    """Equivalente a limpar+delinear o segmento inteiro, mas em blocos com RAM limitada.
    Devolve os mesmos 4 objetos que o caminho direto: ecg_cleaned (array completo),
    r_peaks_global, q_global, s_global, t_global (índices GLOBAIS, alinhados 1-a-1
    com r_peaks_global, ordenados)."""
    n = len(ecg_raw)
    chunk_samples = int(CHUNK_MIN * 60 * fs)
    overlap = int(OVERLAP_SEC * fs)

    ecg_cleaned = np.empty(n, dtype=np.float32)
    r_all, q_all, s_all, t_all = [], [], [], []

    pos = 0
    while pos < n:
        nucleo_fim = min(n, pos + chunk_samples)
        ini = max(0, pos - overlap)
        fim = min(n, nucleo_fim + overlap)

        limpo, r_local, q_local, s_local, t_local = _limpar_delinear_bloco_ecg(
            ecg_raw[ini:fim], fs, nk)

        # núcleo deste bloco dentro do array local [ini:fim]
        nuc_ini_local = pos - ini
        nuc_fim_local = nucleo_fim - ini
        ecg_cleaned[pos:nucleo_fim] = limpo[nuc_ini_local:nuc_fim_local]

        # só guarda R (e Q/S/T correspondentes) cujo R cai dentro do núcleo -- evita
        # duplicar o mesmo batimento contado pelo bloco seguinte (que também o vê,
        # na sua zona de overlap à esquerda).
        for i, r in enumerate(r_local):
            if nuc_ini_local <= r < nuc_fim_local:
                r_all.append(int(r + ini))
                q_all.append((q_local[i] + ini) if (i < len(q_local) and q_local[i] is not None) else None)
                s_all.append((s_local[i] + ini) if (i < len(s_local) and s_local[i] is not None) else None)
                t_all.append((t_local[i] + ini) if (i < len(t_local) and t_local[i] is not None) else None)

        pos = nucleo_fim

    if skew(ecg_cleaned) < 0:
        ecg_cleaned = -ecg_cleaned

    ordem = np.argsort(r_all)
    r_peaks_global = np.array(r_all)[ordem]
    q_global = [q_all[i] for i in ordem]
    s_global = [s_all[i] for i in ordem]
    t_global = [t_all[i] for i in ordem]
    return ecg_cleaned, r_peaks_global, q_global, s_global, t_global


def processar_ppg_em_blocos(ppg_raw, fs, nk):
    """Equivalente a nk.ppg_process no segmento inteiro, em blocos. Devolve
    (ppg_cleaned, ppg_quality_global, nk_ppg_peaks_global) -- arrays/lista completos."""
    n = len(ppg_raw)
    chunk_samples = int(CHUNK_MIN * 60 * fs)
    overlap = int(OVERLAP_SEC * fs)

    ppg_cleaned = np.empty(n, dtype=np.float32)
    ppg_quality = np.empty(n, dtype=np.float32)
    peaks_all = []

    pos = 0
    while pos < n:
        nucleo_fim = min(n, pos + chunk_samples)
        ini = max(0, pos - overlap)
        fim = min(n, nucleo_fim + overlap)

        ppg_signals, ppg_info = nk.ppg_process(ppg_raw[ini:fim], sampling_rate=fs, method="elgendi")
        limpo_local = ppg_signals["PPG_Clean"].values
        qual_local = ppg_signals["PPG_Quality"].values
        pk_local = np.array(ppg_info["PPG_Peaks"])

        nuc_ini_local = pos - ini
        nuc_fim_local = nucleo_fim - ini
        ppg_cleaned[pos:nucleo_fim] = limpo_local[nuc_ini_local:nuc_fim_local]
        ppg_quality[pos:nucleo_fim] = qual_local[nuc_ini_local:nuc_fim_local]

        pk_nucleo = pk_local[(pk_local >= nuc_ini_local) & (pk_local < nuc_fim_local)]
        peaks_all.extend((pk_nucleo + ini).tolist())

        pos = nucleo_fim

    peaks_all = sorted(peaks_all)
    return ppg_cleaned, ppg_quality, peaks_all


# ========== FUNÇÃO PRINCIPAL (por linha do CSV) — versão nk2 ==========
# Escolhas desta versão:
#   1. Delineação DWT do ECG feita 1x por SEGMENTO (não por janela) -> ~6.4x mais
#      rápido (elimina a redundância 3x do overlap). RAM da DWT é desprezável.
#   2. Gate de PAT FIXO (0.19-0.50s) substituído pelo CLUSTER ADAPTATIVO por janela
#      (bin dominante ± 40ms + IQR, exigindo std<0.05) -- igual ao baseline. Recupera
#      registos com latência de oxímetro (PAT ~0.6s) que o gate fixo deitava fora.
#   3. crest_time/decay_time (dependiam do pé/entalhe, frágeis) SUBSTITUÍDOS pelas
#      features derivativas APG (b/a, c/a, d/a, índice de envelhecimento) -- método
#      Elgendi/Takazawa, robusto, não precisa de localizar pé nem entalhe dicrótico.
# ppg_delineate NÃO existe em NeuroKit (nem em 0.2.13) -> valley usa o fallback scipy
# de encontrar_valley (só p/ amplitudes, robusto). encontrar_pe mantém-se só p/ o gate
# de validade do pulso, já não gera feature.
def processar_linha_individual(row, lock=None):
    import neurokit2 as nk

    s_id = str(row['subject_folder'])

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
    except Exception:
        seg_order = 0

    base_path = row['absolute_path'].replace('.dat', '').replace('.hea', '')
    if not os.path.exists(base_path + '.hea'):
        return 0

    # ── Leitura do registo (só os 3 canais, float32) ────────────────────────────
    try:
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
        del record
        ecg_raw = remover_picos_isolados(ecg_raw)
    except Exception:
        return 0

    # ── NeuroKit: limpeza + picos R + delineação DWT (1x por segmento, ou em blocos
    #    se o segmento for gigante -- ver processar_ecg_em_blocos/processar_ppg_em_blocos) ──
    try:
        duracao_min = len(ecg_raw) / fs / 60
        if duracao_min > CHUNK_LIMIAR_MIN:
            ecg_cleaned, r_peaks_global, q_global, s_global, t_global = \
                processar_ecg_em_blocos(ecg_raw, fs, nk)
            ppg_cleaned, ppg_quality_global, nk_ppg_peaks_global = \
                processar_ppg_em_blocos(ppg_raw, fs, nk)
        else:
            ecg_cleaned = nk.ecg_clean(ecg_raw, sampling_rate=fs, method="neurokit")
            if skew(ecg_cleaned) < 0:
                ecg_cleaned = -ecg_cleaned

            _, r_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method="neurokit")
            r_peaks_global = r_info['ECG_R_Peaks']

            # Delineação DWT no segmento inteiro (Q/S/T por batimento, índices GLOBAIS).
            try:
                _, waves_ecg_global = nk.ecg_delineate(ecg_cleaned, r_peaks_global,
                                                        sampling_rate=fs, method="dwt")
                def _safe_global(k):
                    raw = waves_ecg_global.get(k, [])
                    return [int(x) if (x is not None and not np.isnan(x)) else None for x in raw]
                q_global = _safe_global('ECG_Q_Peaks')
                s_global = _safe_global('ECG_S_Peaks')
                t_global = _safe_global('ECG_T_Peaks')
            except Exception:
                n = len(r_peaks_global)
                q_global = s_global = t_global = [None] * n

            ppg_signals, ppg_info = nk.ppg_process(ppg_raw, sampling_rate=fs, method="elgendi")
            ppg_cleaned         = ppg_signals["PPG_Clean"].values
            ppg_quality_global  = ppg_signals["PPG_Quality"].values
            nk_ppg_peaks_global = ppg_info["PPG_Peaks"].tolist()
    except Exception:
        return 0

    resultados_da_linha = []
    samples_per_win = int(WINDOW_SIZE_SEC * fs)
    step_samples    = int(STRIDE_SEC * fs)
    num_windows     = int((len(ecg_cleaned) - samples_per_win) / step_samples) + 1

    for w in range(num_windows):
        start = w * step_samples
        end   = start + samples_per_win
        if end > len(ecg_cleaned):
            break

        if np.mean(ppg_quality_global[start:end]) < 0.75:
            continue

        win_ecg = ecg_cleaned[start:end]
        win_ppg = ppg_cleaned[start:end]
        win_abp = abp_raw[start:end]
        win_ppg_pos = win_ppg - np.min(win_ppg) + 0.01

        mask_r        = (r_peaks_global >= start) & (r_peaks_global < end)
        r_seg_indices = np.where(mask_r)[0]
        r_peaks_win   = r_peaks_global[mask_r] - start
        if len(r_peaks_win) < 8:
            continue

        nk_ppg_arr        = np.array(nk_ppg_peaks_global)
        mask_ppg          = (nk_ppg_arr >= start) & (nk_ppg_arr < end)
        ppg_global_in_win = nk_ppg_arr[mask_ppg].tolist()
        ppg_peaks_win     = [p - start for p in ppg_global_in_win]

        # ── Q/S/T deste janela: fatiar os GLOBAIS (delineados 1x) → relativos à janela ──
        # r_seg_indices[i] é a posição global do i-ésimo batimento da janela; alinhado
        # 1-a-1 com r_peaks_win. (Antes vinha de nk.ecg_delineate por janela.)
        def _slice_win(glob_list):
            out = []
            for gbi in r_seg_indices:
                v = glob_list[gbi] if gbi < len(glob_list) else None
                if v is None:
                    out.append(None)
                else:
                    rel = v - start
                    out.append(rel if 0 <= rel < len(win_ecg) else None)
            return out
        q_peaks_win = _slice_win(q_global)
        s_peaks_win = _slice_win(s_global)
        t_peaks_win = _slice_win(t_global)
        # ppg_delineate inexistente em NK 0.2.9 → fallback scipy em encontrar_valley/pe
        nk_ppg_onsets_win, nk_ppg_notches_win = [], []

        ppg_peak_to_nk_idx = {pk: i for i, pk in enumerate(ppg_peaks_win)}

        # Batimentos PPG: recolhidos TODOS (cluster adaptativo aplicado após o loop)
        ppg_beats = []
        # QRS/QT independentes do PAT (filtrados por IQR próprio depois)
        b_qrs_durations, b_r_amplitudes, b_qrs_amplitudes, b_qrs_areas = [], [], [], []
        b_qts, b_tqs, b_qtcs, b_sdis = [], [], [], []

        for i, r_seg_idx in enumerate(r_seg_indices):
            r      = int(r_peaks_win[i])
            r_next = int(r_peaks_win[i + 1]) if i + 1 < len(r_peaks_win) else None
            if r_next is None:
                continue
            hr = (60 * fs) / (r_next - r)
            if not (45 < hr < 140):
                continue

            q_idx = q_peaks_win[i] if i < len(q_peaks_win) else None
            s_idx = s_peaks_win[i] if i < len(s_peaks_win) else None
            t_idx = t_peaks_win[i] if i < len(t_peaks_win) else None

            if q_idx is None or not (0 <= q_idx < r):
                q_s   = max(0, r - int(0.05 * fs))
                q_idx = int(q_s + np.argmin(win_ecg[q_s:r])) if q_s < r else r
            if s_idx is None or not (r < s_idx < len(win_ecg)):
                s_e   = min(r + int(0.07 * fs), len(win_ecg))
                s_idx = int(r + np.argmin(win_ecg[r:s_e])) if r < s_e else r

            # ── QRS features ──
            if q_idx < s_idx < len(win_ecg):
                qrs_duration = (s_idx - q_idx) / fs
                if 0.04 < qrs_duration < 0.16:
                    r_amplitude   = float(win_ecg[r])
                    qrs_amplitude = float(win_ecg[r] - min(win_ecg[q_idx], win_ecg[s_idx]))
                    seg_qrs       = win_ecg[q_idx:s_idx + 1] - win_ecg[q_idx]
                    qrs_area      = float(simpson(y=seg_qrs)) if len(seg_qrs) > 1 else 0.0
                    b_qrs_durations.append(qrs_duration)
                    b_r_amplitudes.append(r_amplitude)
                    b_qrs_amplitudes.append(qrs_amplitude)
                    b_qrs_areas.append(qrs_area)

            # ── QT / TQ / QTc / SDI ──
            if t_idx is not None and q_idx is not None and q_idx < t_idx < len(win_ecg):
                qt_interval = (t_idx - q_idx) / fs
                rr_sec      = (r_next - r) / fs
                qtc         = qt_interval / np.sqrt(rr_sec + 1e-6)
                b_qts.append(qt_interval)
                b_qtcs.append(qtc)
                qn_idx = q_peaks_win[i + 1] if (i + 1) < len(q_peaks_win) else None
                if qn_idx is not None and t_idx < qn_idx < len(win_ecg):
                    tq_interval = (qn_idx - t_idx) / fs
                    b_tqs.append(tq_interval)
                    b_sdis.append(min(qt_interval / (tq_interval + 1e-6), 10.0))

            # ── PPG peak deste batimento ──
            ppg_candidates = [pk for pk in ppg_peaks_win if r < pk < r_next]
            if not ppg_candidates:
                continue
            ppg_pk = ppg_candidates[0]
            nk_idx = ppg_peak_to_nk_idx.get(ppg_pk, -1)
            ppg_vly = encontrar_valley(win_ppg, ppg_pk, nk_idx, ppg_peaks_win, nk_ppg_onsets_win, fs)
            pe      = encontrar_pe(win_ppg, ppg_pk, nk_idx, ppg_peaks_win, nk_ppg_notches_win, r_next, fs)
            if not (ppg_vly < ppg_pk < pe):
                continue

            pat = (ppg_pk - r) / fs
            # Gate SÓ com limite inferior (0.10s), sem limite superior fixo:
            #   - a latência do oxímetro só ADICIONA atraso -> nunca encurta o PAT,
            #     por isso um PAT < 0.10s é sempre artefacto (entalhe dicrótico/QRS a
            #     contaminar o PPG), não latência. Rejeita-se -> mantém a "limpeza" do NK.
            #   - SEM limite superior fixo (só < RR): assim os registos com latência de
            #     oxímetro (PAT ~0.6s, tight cluster) sobrevivem -> recupera os pacientes
            #     que o gate fixo 0.19-0.50 do com_nk deitava fora.
            # O cluster adaptativo (após o loop) fica com o agrupamento dominante entre
            # os PATs assim filtrados.
            if not (0.10 < pat < (r_next - r) / fs):
                continue

            ppg_valley_intensity = float(win_ppg_pos[ppg_vly])
            ppg_ac_amplitude     = float(win_ppg_pos[ppg_pk] - win_ppg_pos[ppg_vly])
            omega                = hr / 60.0
            rho_blood            = 1060
            alpha = ppg_valley_intensity * np.sqrt((rho_blood * omega) / (ppg_ac_amplitude + 1e-6))
            upslope   = win_ppg_pos[ppg_vly:ppg_pk + 1]
            deriv     = np.diff(upslope) * fs
            max_slope = float(np.max(deriv)) if len(deriv) > 0 else 0.0
            alpha_b   = ppg_valley_intensity * np.sqrt((omega * rho_blood) / (1.0 / (max_slope + 1e-6) + 1e-6))

            area_beat = None
            w50_beat  = None
            p_low  = max(0, ppg_pk - int(0.1 * fs))
            p_high = min(len(win_ppg_pos), ppg_pk + int(0.3 * fs))
            pulso  = win_ppg_pos[p_low:p_high]
            if len(pulso) > 1 and (np.max(pulso) - np.min(pulso)) > 0.01:
                norm_p = (pulso - np.min(pulso)) / (np.max(pulso) - np.min(pulso))
                area_beat = float(simpson(y=norm_p))
                idx50 = np.where(norm_p >= 0.5)[0]
                if len(idx50) > 1:
                    w50_beat = (idx50[-1] - idx50[0]) / fs

            # APG (2ª derivada) num pulso completo a partir do pé (~1.1×RR),
            # substitui crest/decay. Robusto: não usa pé/entalhe como fiducial.
            fim_pulso = min(ppg_vly + int(1.1 * (r_next - r)), len(win_ppg))
            apg_feat = extrair_apg_features(win_ppg[ppg_vly:fim_pulso], fs)

            ppg_beats.append({
                'pat': pat, 'hr': hr,
                'valley': ppg_valley_intensity, 'ac': ppg_ac_amplitude,
                'alpha': alpha, 'alpha_b': alpha_b,
                'apg_b_a':   apg_feat['apg_b_a']   if apg_feat else np.nan,
                'apg_c_a':   apg_feat['apg_c_a']   if apg_feat else np.nan,
                'apg_d_a':   apg_feat['apg_d_a']   if apg_feat else np.nan,
                'apg_aging': apg_feat['apg_aging'] if apg_feat else np.nan,
                'area': area_beat, 'w50': w50_beat,
            })

        # ── Cluster adaptativo de PAT por janela + IQR + std<0.05 (igual ao baseline) ──
        if len(ppg_beats) < 6:
            continue
        pats_all = np.array([b['pat'] for b in ppg_beats])
        bins = np.arange(0, pats_all.max() + 0.04, 0.04)
        hist, edges = np.histogram(pats_all, bins=bins)
        i_max = int(np.argmax(hist))
        cl_min, cl_max = edges[i_max] - 0.04, edges[i_max + 1] + 0.04
        kept_mask = (pats_all >= cl_min) & (pats_all <= cl_max)

        if kept_mask.sum() >= 4:
            pk_vals = pats_all[kept_mask]
            q1, q3 = np.percentile(pk_vals, [25, 75])
            iqr = (q3 - q1) if (q3 - q1) != 0 else 0.05
            lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            kept_mask = kept_mask & (pats_all >= lim_inf) & (pats_all <= lim_sup)

        kept_beats = [b for b, k in zip(ppg_beats, kept_mask) if k]
        if len(kept_beats) < 6:
            continue
        kept_pats = np.array([b['pat'] for b in kept_beats])
        if np.std(kept_pats) >= 0.05:
            continue

        # QRS IQR (independente, como em Extração_features.ipynb)
        if len(b_r_amplitudes) >= 4:
            r_arr = np.array(b_r_amplitudes)
            q1r, q3r = np.percentile(r_arr, [25, 75])
            iqr_r = q3r - q1r
            mask_r_amp = (r_arr >= q1r - 2.0 * iqr_r) & (r_arr <= q3r + 2.0 * iqr_r)
            b_qrs_durations  = [v for v, ok in zip(b_qrs_durations,  mask_r_amp) if ok]
            b_r_amplitudes   = [v for v, ok in zip(b_r_amplitudes,   mask_r_amp) if ok]
            b_qrs_amplitudes = [v for v, ok in zip(b_qrs_amplitudes, mask_r_amp) if ok]
            b_qrs_areas      = [v for v, ok in zip(b_qrs_areas,      mask_r_amp) if ok]

        sbp, dbp, sbp_std, pp = extrair_targets_abp_robusto(win_abp, fs)
        if not (pp > 25 and sbp_std < 12 and 65 < sbp < 195 and 35 < dbp < 115):
            continue

        def _m(key):
            vals = [b[key] for b in kept_beats if b[key] is not None]
            return round(float(np.mean(vals)), 4) if vals else np.nan

        resultados_da_linha.append({
            'subject_id':           s_id,
            'segment_name':         seg_name,
            'segment_order':        seg_order,
            'window_idx':           w,
            'dataset':              dataset_label,
            'gender':               row['gender'],
            'age':                  row['age'],
            'pat_mean':             round(float(np.mean(kept_pats)), 4),
            'hr_mean':              round(float(np.mean([b['hr'] for b in kept_beats])), 2),
            'qrs_duration':         round(np.mean(b_qrs_durations),  4) if b_qrs_durations  else np.nan,
            'r_amplitude':          round(np.mean(b_r_amplitudes),   4) if b_r_amplitudes   else np.nan,
            'qrs_amplitude':        round(np.mean(b_qrs_amplitudes), 4) if b_qrs_amplitudes else np.nan,
            'qrs_area':             round(np.mean(b_qrs_areas),      4) if b_qrs_areas      else np.nan,
            'w50':                  _m('w50'),
            'area_mean':            _m('area'),
            'ppg_valley_intensity': _m('valley'),
            'ppg_ac_amplitude':     _m('ac'),
            'apg_b_a':              _m('apg_b_a'),
            'apg_c_a':              _m('apg_c_a'),
            'apg_d_a':              _m('apg_d_a'),
            'apg_aging':            _m('apg_aging'),
            'alpha_womersley':      _m('alpha'),
            'alpha_womersley_b':    _m('alpha_b'),
            'qt_interval':          round(np.mean(b_qts),  4) if b_qts  else np.nan,
            'tq_interval':          round(np.mean(b_tqs),  4) if b_tqs  else np.nan,
            'qtc_interval':         round(np.mean(b_qtcs), 4) if b_qtcs else np.nan,
            'sdi_interaction':      round(np.mean(b_sdis), 4) if b_sdis else np.nan,
            'sbp_target':           round(sbp, 2),
            'dbp_target':           round(dbp, 2),
        })

    # Escreve já este segmento no CSV (em vez de devolver para acumular em
    # memória e só gravar no fim) -- se o processo for morto/crashar a meio das
    # ~9600 tasks, tudo o que já foi processado fica guardado. `lock` evita que
    # dois workers escrevam ao mesmo tempo (corrida entre processos).
    if resultados_da_linha and lock is not None:
        df_linha = pd.DataFrame(resultados_da_linha)
        os.makedirs(PATH_FEATURES_OUT, exist_ok=True)
        with lock:
            for g, label in [('M', 'homens'), ('F', 'mulheres')]:
                subset = df_linha[df_linha['gender'] == g]
                if not subset.empty:
                    caminho = f"{PATH_FEATURES_OUT}/features_{label}_com_nk2.csv"
                    subset.to_csv(caminho, mode='a', header=not os.path.exists(caminho), index=False)
    return len(resultados_da_linha)


# ========== ORQUESTRADOR PARALELO ==========
def _obter_ja_processados():
    """(subject_id, segment_name) já gravados em corridas anteriores -- para retomar
    sem duplicar caso o processo tenha sido parado/crashado a meio."""
    ja = set()
    for label in ['homens', 'mulheres']:
        caminho = f"{PATH_FEATURES_OUT}/features_{label}_com_nk2.csv"
        if os.path.exists(caminho):
            existentes = pd.read_csv(caminho, usecols=['subject_id', 'segment_name'])
            ja |= set(zip(existentes['subject_id'], existentes['segment_name']))
    return ja


def processar_datasets_hibridos():
    for csv_file in INPUT_CSVS:
        if not os.path.exists(csv_file):
            print(f"Aviso: {csv_file} não encontrado.")
            continue
        df = pd.read_csv(csv_file)
        n_original = len(df)

        ja_processados = _obter_ja_processados()
        if ja_processados:
            chave = pd.Series(list(zip(df['subject_folder'], df['segment'])), index=df.index)
            df = df[~chave.isin(ja_processados)].reset_index(drop=True)
            print(f"↩️  A retomar: {len(ja_processados)} segmentos já gravados numa corrida anterior "
                  f"-- {n_original} -> {len(df)} por processar.")

        print(f"A inicializar processamento paralelo para {len(df)} registos em: {csv_file}")
        num_cores = N_JOBS if N_JOBS is not None else max(1, multiprocessing.cpu_count() // 2)
        print(f"Motor paralelo ativo: {num_cores} cores (de {multiprocessing.cpu_count()}).")

        lock = multiprocessing.Manager().Lock()
        contagens = Parallel(n_jobs=num_cores, backend="multiprocessing", verbose=10, max_nbytes=None)(
            delayed(processar_linha_individual)(row, lock) for _, row in df.iterrows()
        )
        total_janelas = sum(contagens)
        print(f"Concluído! Janelas processadas nesta corrida: {total_janelas} "
              f"(ficheiros em {PATH_FEATURES_OUT}/features_*_com_nk2.csv, incluindo corridas anteriores retomadas)")


if __name__ == "__main__":
    processar_datasets_hibridos()
