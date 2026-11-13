"""
09b — Extração PPG-pura + APG + 5 características morfológicas novas.

É o 09_extrair_features_ppg_apg.py SEM ALTERAÇÃO a nenhuma característica, portão ou seleção de batimentos/janelas, mais cinco
características por batimento, que entram no resultado como a média sobre os batimentos aceites, como as restantes:
  ppg_max_slope_norm   máxima inclinação sistólica / amplitude do pulso (1/s). Insensível ao ganho do sensor.
  ppg_max_slope        a mesma, sem normalizar (unidades do sinal por segundo)
  w25, w75             largura do pulso (s) a 25 % e a 75 % da amplitude, calculada como w50 (mesma janela e normalização)
  reflection_index     (pico diastólico - pé) / (pico sistólico - pé), só quando há um pico diastólico distinto depois do entalhe
Colunas extra de diagnóstico: n_ri (batimentos com índice de reflexão), w25_trunc, w75_trunc (fração de batimentos em que a
largura toca o fim da janela do pulso, por isso é um limite inferior).

NÃO sobrescreve nada: grava numa pasta nova (Features_morfologia/) e recusa correr se os ficheiros de saída já existirem.
Processa por blocos de registos e guarda cada bloco, por isso pode ser interrompido e retomado sem refazer o que já está feito.

Uso:  python3 09b_extrair_features_ppg_morfologia.py [--amostra N] [--saida PASTA] [--jobs N]
  --amostra N  só os primeiros N registos de uma amostra fixa (semente 0), para verificar (grava em --saida, que deve ser uma pasta de teste)

O resto (a 2.ª derivada, e o que muda face à extração do absoluto) está em 09_extrair_features_ppg_apg.py e 01_extrair_features_ppg_puro.py.
"""
import os
import time
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from scipy.integrate import simpson
from scipy.stats import skew

# --- CONFIGURAÇÃO (janela e passo como na extração do absoluto) ---
PATH_DATASET = "/media/filipe/Dados/MIMIC_DATASET_CONSOLIDADO"
INPUT_CSVS = [f"{PATH_DATASET}/dataset_para_extração_features.csv"]
PATH_FEATURES_OUT = "/media/filipe/Dados/Features_morfologia"
WINDOW_SIZE_SEC = 15
STRIDE_SEC = 5
# Nº de processos paralelos. Aqui só se carregam 2 canais (PPG+ABP), por isso a memória não limita (o maior registo estima ~520 MB e cada worker usa 0,16-0,56 GB): o processo é limitado pelo CPU.
N_JOBS = 14   # deixa 2 núcleos livres para o resto do sistema

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
        pe = None
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

        # ── APG: 2.ª derivada de um pulso completo a partir do PÉ ──────────
        # O fim do pulso é `vly + 1.1 x` o intervalo pico-a-pico do PPG, a duração
        # de um ciclo medida sem ECG.
        if pk_next is not None:
            fim_pulso = min(vly + int(1.1 * (pk_next - pk)), len(win))
        else:
            fim_pulso = min(vly + int(1.1 * fs * 60.0 / (hr_mediana + 1e-9)), len(win))
        apg = extrair_apg_features(win[vly:fim_pulso], fs)

        batimentos.append({
            'hr': hr, 'crest_time': crest_time, 'decay_time': decay_time,
            'alpha': round(alpha, 4), 'alpha_b': round(alpha_b, 4),
            'pulse_area': pulse_area, 'w50': w50,
            'w25': w25, 'w75': w75, 'w25_trunc': float(w25_t) if w25 is not None else None,
            'w75_trunc': float(w75_t) if w75 is not None else None,
            'ppg_max_slope': round(max_slope, 4), 'ppg_max_slope_norm': round(max_slope / ac, 4),
            'reflection_index': ri,
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
        'w25': med('w25'), 'w75': med('w75'), 'w25_trunc': med('w25_trunc'), 'w75_trunc': med('w75_trunc'),
        'ppg_max_slope': med('ppg_max_slope'), 'ppg_max_slope_norm': med('ppg_max_slope_norm'),
        'reflection_index': med('reflection_index'),
        'n_ri': sum(1 for b in kept if b.get('reflection_index') is not None),
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


def _tarefa(t):
    i, row = t
    return i, processar_linha_individual(row) or []


def main():
    import argparse
    import glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--amostra", type=int, default=None, help="só N registos (amostra fixa), para verificar")
    ap.add_argument("--saida", default=PATH_FEATURES_OUT)
    ap.add_argument("--jobs", type=int, default=N_JOBS)
    ap.add_argument("--bloco", type=int, default=400, help="registos por bloco guardado")
    a = ap.parse_args()
    saida = a.saida
    finais = [f"{saida}/features_{l}_ppg_apg_morf.csv" for l in ("homens", "mulheres")]
    antigas = "/media/filipe/Dados/Features"
    if os.path.abspath(saida) == os.path.abspath(antigas):
        raise SystemExit("recuso gravar na pasta Features/ (dados já existentes)")
    if any(os.path.exists(p) for p in finais):
        raise SystemExit(f"já existe {finais}; não sobrescrevo. Apagar à mão se for para refazer.")
    tmp = f"{saida}/blocos"
    os.makedirs(tmp, exist_ok=True)

    df = pd.read_csv(INPUT_CSVS[0])
    if a.amostra:
        df = df.sample(a.amostra, random_state=0)
    df = df.reset_index(drop=True)
    print(f"{len(df)} registos (só PPG, sem ECG), {a.jobs} processos, blocos de {a.bloco}", flush=True)

    # Os blocos continuam a ser a unidade guardada (e retomável), mas os registos são distribuídos de forma contínua: cada processo
    # vai buscar o registo seguinte assim que acaba o anterior, sem esperar pelo mais lento do bloco. As linhas de cada bloco são
    # escritas pela ordem original dos registos, por isso o ficheiro é idêntico ao do processamento por blocos.
    blocos = {b0: f"{tmp}/bloco_{b0:06d}.csv" for b0 in range(0, len(df), a.bloco)}
    por_fazer = [b0 for b0, p in blocos.items() if not os.path.exists(p)]
    for b0 in blocos:
        if b0 not in por_fazer:
            print(f"bloco {b0} já feito, salto", flush=True)
    tarefas = [(i, df.iloc[i]) for b0 in por_fazer for i in range(b0, min(b0 + a.bloco, len(df)))]
    resultados, em_falta = {}, {b0: min(a.bloco, len(df) - b0) for b0 in por_fazer}
    import multiprocessing as mp
    with mp.Pool(a.jobs, maxtasksperchild=40) as pool:
        for i, linhas in pool.imap_unordered(_tarefa, tarefas, chunksize=1):
            resultados[i] = linhas
            b0 = i - i % a.bloco
            em_falta[b0] -= 1
            if em_falta[b0] == 0:
                regs = [x for k in range(b0, min(b0 + a.bloco, len(df))) for x in resultados.pop(k)]
                pd.DataFrame(regs).to_csv(blocos[b0] + ".parcial", index=False)
                os.replace(blocos[b0] + ".parcial", blocos[b0])        # só aparece completo
                print(f"bloco {b0}-{min(b0 + a.bloco, len(df))}: {len(regs)} janelas  ({time.strftime('%H:%M:%S')})", flush=True)

    df_feat = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{tmp}/bloco_*.csv"))], ignore_index=True)
    if df_feat.empty:
        raise SystemExit("Nenhuma janela válida gerada.")
    for g, label, out in [('M', 'homens', finais[0]), ('F', 'mulheres', finais[1])]:
        subset = df_feat[df_feat['gender'] == g]
        if not subset.empty:
            subset.to_csv(out + ".parcial", index=False)
            os.replace(out + ".parcial", out)
            print(f"Guardado: {out} ({len(subset)} janelas)")
    print(f"Total de janelas: {len(df_feat)} | sujeitos: {df_feat.subject_id.nunique()}")


if __name__ == "__main__":
    main()
