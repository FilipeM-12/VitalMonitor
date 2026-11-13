"""
09 — Extração PPG-pura + features APG (2.ª derivada do PPG).

Igual ao 01_extrair_features_ppg_puro.py, mais quatro features por batimento
vindas da ACELERAÇÃO do pulso: apg_b_a, apg_c_a, apg_d_a e o índice de
envelhecimento (b-c-d-e)/a. Método Takazawa/Elgendi.

PORQUÊ: são os únicos marcadores novos que se conseguem tirar do MAX30102 sem
hardware adicional, e medem tónus vascular / rigidez arterial — a fisiologia
que muda primeiro numa variação de pressão. São também mais robustas do que o
que o pipeline usa hoje: a função vem de extrair_features_nk2.py, onde
substituiu crest_time/decay_time precisamente por estas "dependerem do pé/
entalhe, frágeis" (comentário do próprio código). O APG não precisa de
localizar nem o pé do pulso nem o entalhe dicrótico.

ÚNICA ADAPTAÇÃO face a extrair_features_nk2.py: lá o pulso é delimitado por `r_next` (o
pico R seguinte); aqui é delimitado pelo PICO PPG seguinte, coerente com o
resto deste pipeline. A função de extração em si não foi tocada.

O que muda face à extração do absoluto está em 01_extrair_features_ppg_puro.py.
"""
import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from scipy.integrate import simpson
from scipy.stats import skew
from joblib import Parallel, delayed

# --- CONFIGURAÇÃO (janela e passo como na extração do absoluto) ---
PATH_DATASET = "/media/filipe/Dados/MIMIC_DATASET_CONSOLIDADO"
INPUT_CSVS = [f"{PATH_DATASET}/dataset_para_extração_features.csv"]
PATH_FEATURES_OUT = "/media/filipe/Dados/Features"
WINDOW_SIZE_SEC = 15
STRIDE_SEC = 5
# Nº de processos paralelos. Aqui só se carregam 2 canais (PPG+ABP), por isso a memória não limita (o maior registo estima ~520 MB e cada worker usa 0,16-0,56 GB): o processo é limitado pelo CPU.
N_JOBS = 16

RHO_SANGUE = 1060

# Portões de qualidade — os limiares que têm equivalente direto na extração do absoluto
# mantêm o mesmo valor, para a comparação não misturar "mudei a segmentação"
# com "mudei o critério de aceitação".
MIN_BATIMENTOS      = 8       # era len(r_peaks) < 8
MIN_BATIMENTOS_KEPT = 6       # era len(kept) < 6
CV_RITMO_MAX        = 0.25    # era cv_rr > 0.25 (irregularidade do ritmo)
DISPERSAO_MAX       = 0.05    # limite de std(crest_time)
HR_MIN, HR_MAX      = 45, 140  # o mesmo intervalo do filtro de PAT
CREST_MIN, CREST_MAX = 0.05, 0.40   # crest time fisiologicamente plausível (s)


def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)
    return filtfilt(b, a, data)


def extrair_targets_abp_robusto(win_abp, fs):
    """É a etiqueta, não pode mudar."""
    sbp_idxs, _ = find_peaks(win_abp, distance=int(fs * 0.4), height=np.percentile(win_abp, 50))
    dbp_idxs, _ = find_peaks(-win_abp, distance=int(fs * 0.4))
    if len(sbp_idxs) < 3 or len(dbp_idxs) < 3:
        return None, None, None, None
    sbp_vals = win_abp[sbp_idxs]
    sbp_mean = float(np.mean(sbp_vals))
    sbp_std = float(np.std(sbp_vals))
    dbp_mean = float(np.mean(win_abp[dbp_idxs]))
    return sbp_mean, dbp_mean, sbp_std, sbp_mean - dbp_mean


def extrair_apg_features(pulso, fs):
    """Ondas a,b,c,d,e da 2ª derivada (APG) de UM pulso PPG e os rácios b/a, c/a,
    d/a + índice de envelhecimento (b-c-d-e)/a. Método Elgendi/Takazawa: NÃO precisa
    do pé nem do entalhe dicrótico (mais robusto que crest/decay times). None se não
    fiável.

    Cópia literal de 05_EXTRACAO_FEATURES/scripts/extrair_features_nk2.py — não
    alterar sem alterar lá também, senão as duas variantes deixam de ser
    comparáveis.
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


def corrigir_orientacao(win_ppg):
    """Repõe a convenção do PLETH do MIMIC (pico sistólico = MÁXIMO).

    No MIMIC é sempre um no-op (o PLETH já vem nesta convenção), mas TEM de existir
    aqui para o código de treino e o de produção serem o mesmo: o MAX30102 da
    bancada devolve intensidade de luz recebida e sai com o pulso invertido
    (skew(diff) negativo). Ver extracao_ppg_classificador.py no servidor.
    """
    return -win_ppg if skew(np.diff(win_ppg)) < 0 else win_ppg


def processar_janela_so_ppg(win_ppg, fs):
    """Segmenta batimentos e extrai features usando APENAS o PPG.

    Devolve um dict de features, ou None se a janela não passar os portões.
    """
    win_ppg = corrigir_orientacao(win_ppg)
    win = win_ppg - np.min(win_ppg) + 0.01     # positividade para alpha (não move índices)
    amplitude_janela = float(np.max(win) - np.min(win))
    if amplitude_janela <= 0:
        return None

    # ── 1. Picos sistólicos, diretamente no PPG ──────────────────────────────
    # Mesmos parâmetros que a extração do absoluto usa para `ppg_peaks_all` (distância
    # mínima de 300 ms afasta o entalhe dicrótico; proeminência de 15% da
    # amplitude da janela evita marcar ondulação de linha de base).
    prom_min = amplitude_janela * 0.15
    picos, _ = find_peaks(win, distance=int(fs * 0.30), prominence=prom_min)
    if len(picos) < MIN_BATIMENTOS:
        return None

    # ── 2. Portões de qualidade (equivalentes de PPG aos do ECG) ─────────────
    # Ritmo: a extração do absoluto rejeita por CV do RR > 0,25 (só quando a amplitude do R também é baixa, < 0,15); aqui é o CV do intervalo pico-a-pico do PPG, que mede a regularidade do ciclo cardíaco sem ECG, e rejeita sempre.
    pp = np.diff(picos) / fs
    if len(pp) < 2:
        return None
    cv_ritmo = float(np.std(pp) / (np.mean(pp) + 1e-9))
    hr_mediana = 60.0 / (float(np.median(pp)) + 1e-9)
    if not (HR_MIN <= hr_mediana <= HR_MAX):
        return None
    if cv_ritmo > CV_RITMO_MAX:
        return None

    # Excursão anómala (artefacto de movimento): a extração do absoluto compara o máximo
    # absoluto do ECG com 4x a amplitude típica do R. Aqui compara-se o desvio
    # face à mediana com 4x a amplitude pulsátil típica.
    amp_pulsatil = float(np.median([win[p] for p in picos])) - float(np.median(win))
    if amp_pulsatil <= 0:
        return None
    if float(np.max(np.abs(win - np.median(win)))) > 4 * amp_pulsatil:
        return None

    # ── 3. Features por batimento ────────────────────────────────────────────
    batimentos = []
    n = len(picos)
    for i in range(n):
        pk = int(picos[i])
        pk_next = int(picos[i + 1]) if i + 1 < n else None
        pk_prev = int(picos[i - 1]) if i > 0 else None

        # HR do intervalo pico-a-pico do PPG (substitui o R-R)
        hr = round((60 * fs) / (pk_next - pk), 2) if pk_next is not None else None

        # Vale/pé: mínimo antes do pico, limitado pelo pico PPG anterior. Esta
        # lógica já é só-PPG na extração do absoluto (o comentário lá diz explicitamente
        # "o pé pertence à estrutura do PRÓPRIO PPG").
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

        upslope = win[vly:pk + 1]
        deriv = np.diff(upslope) * fs
        max_slope = float(np.max(deriv)) if len(deriv) > 0 else 0.0
        b_coef = 1.0 / (max_slope + 1e-6)
        alpha_b = float(vly_val * np.sqrt((omega * RHO_SANGUE) / (b_coef + 1e-6)))

        # Fim do pulso: 1.º mínimo local depois do pico, limitado pelo pico
        # seguinte. Não usa o R.
        limite = pk_next if pk_next is not None else min(pk + int(0.9 * fs), len(win))
        pe_ini2 = pk + int(0.04 * fs)
        pe_fim = min(pk + int(0.50 * fs), len(win), limite)
        if pe_ini2 < pe_fim:
            seg = win[pe_ini2:pe_fim]
            mins, _ = find_peaks(-seg, distance=int(fs * 0.05))
            pe = int(pe_ini2 + (mins[0] if len(mins) > 0 else np.argmin(seg)))
            decay_time = round((pe - pk) / fs, 4)
        else:
            decay_time = None

        # Geometria do pulso — janela e normalização idênticas às da extração do absoluto
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

        # ── APG: 2.ª derivada de um pulso completo a partir do PÉ ──────────
        # No extrair_features_nk2.py o fim do pulso era `ppg_vly + 1.1*(r_next-r)`,
        # ou seja 1,1 x o intervalo R-R. Aqui usa-se 1,1 x o intervalo
        # pico-a-pico do PPG, que é a mesma duração de ciclo medida sem ECG.
        if pk_next is not None:
            fim_pulso = min(vly + int(1.1 * (pk_next - pk)), len(win))
        else:
            fim_pulso = min(vly + int(1.1 * fs * 60.0 / (hr_mediana + 1e-9)), len(win))
        apg = extrair_apg_features(win[vly:fim_pulso], fs)

        batimentos.append({
            'hr': hr, 'crest_time': crest_time, 'decay_time': decay_time,
            'alpha': round(alpha, 4), 'alpha_b': round(alpha_b, 4),
            'pulse_area': pulse_area, 'w50': w50,
            'ppg_valley_intensity': round(vly_val, 4),
            'ppg_ac_amplitude': round(ac, 4),
            'apg_b_a':   apg['apg_b_a']   if apg else None,
            'apg_c_a':   apg['apg_c_a']   if apg else None,
            'apg_d_a':   apg['apg_d_a']   if apg else None,
            'apg_aging': apg['apg_aging'] if apg else None,
        })

    # ── 4. Filtro IQR sobre o crest_time ─────────────────────────────────────
    # Substitui o filtro IQR que a extração do absoluto faz sobre o PAT: remove
    # batimentos cuja morfologia destoa dos restantes da mesma janela.
    crests = np.array([b['crest_time'] for b in batimentos], dtype=float)
    if len(crests) >= 4:
        q1, q3 = np.percentile(crests, [25, 75])
        iqr = (q3 - q1) if (q3 - q1) != 0 else 0.05
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        kept = [b for b in batimentos if lo <= b['crest_time'] <= hi]
    else:
        kept = list(batimentos)

    # Só batimentos com HR fisiológica (a extração do absoluto aplica o mesmo intervalo)
    kept = [b for b in kept if b['hr'] is None or HR_MIN < b['hr'] < HR_MAX]

    if len(kept) < MIN_BATIMENTOS_KEPT:
        return None

    # ── 5. Portão de dispersão ───────────────────────────────────────────────
    # Análogo direto de `std(PAT) >= 0.05`: mesma unidade (s) e
    # mesma ideia — se os fiduciais temporais da janela não são consistentes
    # entre batimentos, a janela não é de confiança.
    if float(np.std([b['crest_time'] for b in kept])) >= DISPERSAO_MAX:
        return None

    def med(key, subset=None):
        # `np.isfinite` e não só `is not None`: extrair_apg_features devolve
        # np.nan (não None) quando não consegue localizar as ondas c/d/e, e um
        # único NaN envenenava a média de toda a janela.
        vals = [b[key] for b in (subset if subset is not None else kept)
                if b.get(key) is not None and np.isfinite(b[key])]
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
        'alpha_womersley_b': med('alpha_b'),
        'apg_b_a': med('apg_b_a'),
        'apg_c_a': med('apg_c_a'),
        'apg_d_a': med('apg_d_a'),
        'apg_aging': med('apg_aging'),
        'n_batimentos': len(kept),
    }


def processar_linha_individual(row):
    s_id = str(row['subject_folder'])
    if len(s_id) == 7:
        dataset_label, canais_abp, canais_ppg = "MIMIC-III", ['ABP', 'ART'], ['PLETH', 'PPG']
    elif len(s_id) == 9:
        dataset_label, canais_abp, canais_ppg = "MIMIC-IV", ['ABP'], ['PLETH']
    else:
        return []

    seg_name = str(row['segment'])
    try:
        seg_order = int(seg_name.split('_')[-1])
    except Exception:
        seg_order = 0

    base_path = row['absolute_path'].replace('.dat', '').replace('.hea', '')
    if not os.path.exists(base_path + '.hea'):
        return []

    try:
        hdr = wfdb.rdheader(base_path)
        fs = hdr.fs
        sig_names = [s.upper() for s in hdr.sig_name]

        # NOTA: não se procura canal de ECG. Um registo sem derivação II mas com
        # PLETH passa a ser utilizável — é precisamente uma das vantagens de um
        # dispositivo sem elétrodos.
        idx_ppg = next((i for i, s in enumerate(sig_names) if any(x in s for x in canais_ppg)), None)
        idx_abp = next((i for i, s in enumerate(sig_names) if any(x in s for x in canais_abp)), None)
        if idx_ppg is None or idx_abp is None:
            return []

        record = wfdb.rdrecord(base_path, channels=[idx_ppg, idx_abp])
        ppg_raw = record.p_signal[:, 0].astype(np.float32)
        abp_raw = record.p_signal[:, 1].astype(np.float32)
        del record

        if np.isnan(ppg_raw).any() or np.isnan(abp_raw).any():
            ppg_raw = pd.Series(ppg_raw).interpolate().ffill().bfill().values
            abp_raw = pd.Series(abp_raw).interpolate().ffill().bfill().values

        ppg_filt = butter_lowpass_filter(ppg_raw, 8, fs)
    except Exception:
        return []

    resultados = []
    samples_per_win = int(WINDOW_SIZE_SEC * fs)
    step_samples = int(STRIDE_SEC * fs)
    num_windows = int((len(ppg_filt) - samples_per_win) / step_samples) + 1

    for w in range(num_windows):
        start = w * step_samples
        end = start + samples_per_win
        win_ppg, win_abp = ppg_filt[start:end], abp_raw[start:end]

        if np.std(win_ppg) < 0.005 or (np.max(win_abp) - np.min(win_abp)) < 15:
            continue

        sbp, dbp, sbp_std, pp = extrair_targets_abp_robusto(win_abp, fs)
        if sbp is None or not (pp > 25 and sbp_std < 12 and 65 < sbp < 195 and 35 < dbp < 115):
            continue

        feats = processar_janela_so_ppg(win_ppg, fs)
        if feats is None:
            continue

        resultados.append({
            'subject_id': s_id, 'segment_name': seg_name, 'segment_order': seg_order,
            'window_idx': w, 'dataset': dataset_label,
            'gender': row['gender'], 'age': row['age'],
            **feats,
            'sbp_target': round(sbp, 2), 'dbp_target': round(dbp, 2),
        })

    return resultados


def main():
    for csv_file in INPUT_CSVS:
        if not os.path.exists(csv_file):
            print(f"Aviso: {csv_file} não encontrado.")
            continue

        df = pd.read_csv(csv_file)
        print(f"A processar {len(df)} registos (só PPG, sem ECG) com {N_JOBS} processos...")

        raw = Parallel(n_jobs=N_JOBS, backend="multiprocessing", verbose=10, max_nbytes=None)(
            delayed(processar_linha_individual)(row) for _, row in df.iterrows()
        )

        linhas = []
        for r in raw:
            if r:
                linhas.extend(r)

        df_feat = pd.DataFrame(linhas)
        if df_feat.empty:
            print("Nenhuma janela válida gerada.")
            return

        os.makedirs(PATH_FEATURES_OUT, exist_ok=True)
        for g, label in [('M', 'homens'), ('F', 'mulheres')]:
            subset = df_feat[df_feat['gender'] == g]
            if not subset.empty:
                out = f"{PATH_FEATURES_OUT}/features_{label}_ppg_apg.csv"
                subset.to_csv(out, index=False)
                print(f"Guardado: {out} ({len(subset)} janelas)")
        print(f"Total de janelas: {len(df_feat)} | sujeitos: {df_feat.subject_id.nunique()}")


if __name__ == "__main__":
    main()
