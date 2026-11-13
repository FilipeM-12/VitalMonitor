import glob, os
import numpy as np
import pandas as pd
from suavizacao_e_normalizacao_causal import normalizar_causal_calibracao

def get_ultra_robust_baseline(values):
    """Bloco de 5 janelas mais estavel nas primeiras 10 — copia exacta do notebook."""
    head_vals = values.head(10).tolist()
    n = len(head_vals)
    if n < 5:
        return np.mean(head_vals) if n > 0 else 0
    window_size = 5
    best_mean = np.mean(head_vals[:window_size])
    min_cv = np.std(head_vals[:window_size]) / (best_mean + 1e-6)
    for i in range(n - window_size + 1):
        subset = head_vals[i:i + window_size]
        cm = np.mean(subset)
        cv = np.std(subset) / (cm + 1e-6)
        if cv < min_cv:
            min_cv, best_mean = cv, cm
    return best_mean


def carregar_e_processar_todos_deltas(diretorio, faixa_etaria=None, cardiaco=None, respiratorio=None, window_size=10, padrao_ficheiro="features_*.csv", chaves_janelas_excluidas=None,
                                        causal_global_z=False, n_calib_morf=10, nrows=None,
                                        grupo_ancoras='subject_id'):
    """
    Função Completa e Otimizada com Estratégia A:
    - Normalização local por SESSÃO para features de tempo/ritmo (PAT, HR).
    - Normalização global por PACIENTE para features âncora morfológicas (PPG e ECG).

    `padrao_ficheiro` controla que ficheiros dentro de `diretorio` são carregados
    (glob). Permite carregar variantes diferentes de extração de features (ex:
    "features_*.csv" para a variante sem NeuroKit, ou "fatures_*_mais_recente_nk2.csv"
    para a variante com NeuroKit) sem duplicar esta função.
    """
    # 1. Carregamento de Ficheiros
    ficheiros = glob.glob(os.path.join(diretorio, padrao_ficheiro))
    if not ficheiros:
        print("Nenhum ficheiro encontrado no diretório especificado.")
        return None

    lista_dfs = []
    for f in ficheiros:
        try:
            df_temp = pd.read_csv(f, nrows=nrows)
            if df_temp.empty: continue
            if 'gender' not in df_temp.columns:
                df_temp['gender'] = 'M' if 'homens' in f.lower() else 'F'
            df_temp = df_temp.rename(columns={'subject_folder': 'subject_id', 'segment': 'segment_name'})
            lista_dfs.append(df_temp)
        except Exception as e:
            print(f"Erro ao ler {f}: {e}")

    if not lista_dfs: return None
    df = pd.concat(lista_dfs, ignore_index=True)

    # Exclusão das janelas sob fármaco vasoativo (vasopressor/vasodilatador)
    # Aplicada aos dados BRUTOS, ANTES do colapso/deteção de sessão -- para a # calibração não ficar desatualizada. Chaves geradas por
    # identificar_janelas_vasoativas.py / identificar_janelas_vasoativas_mimic4.py.
    if chaves_janelas_excluidas is not None:
        chave_linha = pd.Series(
            list(zip(df['subject_id'], df['segment_name'], df['segment_order'], df['window_idx'])),
            index=df.index,
        )
        n_antes = len(df)
        df = df[~chave_linha.isin(chaves_janelas_excluidas)].reset_index(drop=True)
        print(f"Excluídas {n_antes - len(df)} janelas sobrepostas a fármacos vasoativos "
              f"({len(chaves_janelas_excluidas)} chaves de exclusão): {n_antes} -> {len(df)} linhas.")

    # 2. Filtros Clínicos e Populacionais
    if faixa_etaria:
        df = df[(df['age'] >= faixa_etaria[0]) & (df['age'] < faixa_etaria[1])]
    if cardiaco is not None:
        df = df[df['is_cardiac'] == (1 if cardiaco else 0)]
    if respiratorio is not None:
        df = df[df['is_respiratory'] == (1 if respiratorio else 0)]

    # ==============================================================================
    # 3. COLAPSO TEMPORAL (Agrupa 3 janelas de 5s numa única representação de 15s)
    # ==============================================================================
    df['record_id'] = df['segment_name'].astype(str).apply(lambda x: x.rsplit('_', 1)[0])
    df['bloco_id'] = df['window_idx'] // 3

    colunas_meta = ['subject_id', 'record_id', 'segment_order', 'bloco_id', 'gender', 'dataset']
    for col_clinica in ['age', 'is_cardiac', 'is_respiratory']:
        if col_clinica in df.columns:
            colunas_meta.append(col_clinica)

    # as colunas novas entram aqui automaticamente (list comprehension)
    colunas_para_media = [
        c for c in df.columns
        if c not in colunas_meta and c not in ['segment_name', 'window_idx']
    ]

    print(f"A colapsar dataset... Linhas originais: {len(df)}")
    df = df.groupby(colunas_meta)[colunas_para_media].mean().reset_index()
    df = df.sort_values(by=['subject_id', 'record_id', 'segment_order', 'bloco_id']).reset_index(drop=True)

    df['seg_diff'] = df.groupby('record_id')['segment_order'].diff()
    df['win_diff'] = df.groupby(['record_id', 'segment_order'])['bloco_id'].diff()

    df['is_new_session'] = (
        (df['seg_diff'] > 1) |
        (df['win_diff'] > 4) |
        df['seg_diff'].isna() |
        df['win_diff'].isna()
    )
    df['session_id'] = (df['record_id'] + "_" +
                        df.groupby('record_id')['is_new_session'].cumsum().astype(str))

    # ==============================================================================
    # INJEÇÃO ANTI-DRIFT: RELÓGIO, BASELINE ADAPTATIVA E TENDÊNCIAS LONGAS
    # ==============================================================================
    print("A gerar métricas e relógio dinâmico contra o drift de calibração...")
    df = df.sort_values(by=['session_id', 'bloco_id']).reset_index(drop=True)

    primeiro_bloco_sessao = df.groupby('session_id')['bloco_id'].transform('min')
    df['tempo_desde_calib_seg'] = (df['bloco_id'] - primeiro_bloco_sessao) * 15
    df['tempo_pat_interaction'] = df['tempo_desde_calib_seg'] * df['pat_mean']

    # ==============================================================================
    # FEATURES DE DRIFT PERSONALIZADO
    # Estimam o ritmo de mudança do estado cardiovascular deste paciente específico
    # a partir dos primeiros minutos da sessão — sem precisar de recalibrar.
    # ==============================================================================

    # ---- 1. Slope do PAT nos primeiros 8 blocos (~2 min) de cada sessão ----
    # Captura "a que velocidade este paciente deriva naturalmente desde a calibração"
    N_CALIB_BLOCOS = 8

    def _slope_pat_inicial(grupo):
        calib = grupo.head(N_CALIB_BLOCOS)
        t = calib['tempo_desde_calib_seg'].values
        pat = calib['pat_mean'].values
        if len(t) < 4 or np.std(t) < 1e-6:
            return np.nan
        return float(np.polyfit(t, pat, 1)[0])   # s/s

    slopes_iniciais = (
        df.groupby('session_id')
        .apply(_slope_pat_inicial, include_groups=False)
        .reset_index(name='pat_slope_inicial')
    )
    df = df.merge(slopes_iniciais, on='session_id', how='left')
    df['pat_slope_inicial'] = df['pat_slope_inicial'].fillna(0.0)

    # Interacção slope × tempo: estimativa do drift de PAT acumulado neste paciente
    df['drift_personalizado'] = df['pat_slope_inicial'] * df['tempo_desde_calib_seg']

    # ---- 2. Velocidade local do QTc (slope nas últimas 5 janelas) ----
    # Captura se o intervalo QT está a alongar (stress, isquemia) ou a encurtar (taquicardia)
    # Usa shift(1) para não incluir a janela actual — sem lookahead bias
    def _rolling_slope_5(series):
        result = np.full(len(series), np.nan)
        vals = series.values
        for i in range(4, len(vals)):
            y = vals[i - 4 : i]    # últimas 4 janelas (excluindo a actual via shift abaixo)
            x = np.arange(4, dtype=float)
            if np.std(y) > 1e-8:
                result[i] = np.polyfit(x, y, 1)[0]
            else:
                result[i] = 0.0
        return pd.Series(result, index=series.index)

    if 'qtc_interval' in df.columns:
        df['qtc_velocity'] = (
            df.groupby('session_id')['qtc_interval']
            .transform(lambda x: _rolling_slope_5(x.shift(1).fillna(method='bfill')))
            .fillna(0.0)
        )
    else:
        df['qtc_velocity'] = 0.0

    # ---- 3. Estabilidade do PAT na janela de calibração ----
    # Pacientes com PAT instável no arranque tendem a ter maior drift a longo prazo
    def _std_calib(grupo):
        return float(grupo['pat_mean'].head(N_CALIB_BLOCOS).std())

    stab_calib = (
        df.groupby('session_id')
        .apply(_std_calib, include_groups=False)
        .reset_index(name='pat_calib_stability')
    )
    df = df.merge(stab_calib, on='session_id', how='left')
    df['pat_calib_stability'] = df['pat_calib_stability'].fillna(df['pat_calib_stability'].median())

    rolling_median = (
        df.groupby('session_id')['pat_mean']
          .apply(lambda x: x.shift(1).rolling(4, min_periods=1).median())
    )
    df['pat_mean_baseline_adaptativo'] = rolling_median.reset_index(level=0, drop=True)
    df['pat_mean_baseline_adaptativo'] = df['pat_mean_baseline_adaptativo'].fillna(df['pat_mean'])
    df['pat_mean_diff_adaptativo'] = df['pat_mean'] - df['pat_mean_baseline_adaptativo']

    df['pat_mean_ewma_longo'] = df.groupby('session_id')['pat_mean'].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )

    # 4. Limpeza de Outliers Fisiológicos Extremos e Calibração Robusta
    df = df[(df['sbp_target'] > 70) & (df['sbp_target'] < 190)]
    df = df[(df['pat_mean'] > 0.15) & (df['pat_mean'] < 0.60)]

    baselines = df.groupby('session_id').agg({
        'sbp_target': get_ultra_robust_baseline,
        'pat_mean': get_ultra_robust_baseline
    }).reset_index().rename(columns={'sbp_target': 'sbp_calib', 'pat_mean': 'pat_calib'})
    df = df.merge(baselines, on='session_id')

    # 5. Engenharia de Features Estáticas e Biológicas
    df['age_pat_interaction'] = df['age'] * df['pat_mean']
    df['age_squared'] = df['age'] ** 2
    df['is_senior_critical'] = ((df['age'] >= 60) & (df['age'] <= 75)).astype(int)

    # 6. Normalização Z-Score CAUSAL por SESSÃO (Métricas de Ritmo/Tempo)
    #    Usa apenas a janela de calibração (primeiras 10 janelas de cada
    #    sessão), coerente com get_ultra_robust_baseline. Sem lookahead.
    cols_norm_sessao = ['pat_mean', 'hr_mean']
    df = normalizar_causal_calibracao(
        df, cols_norm_sessao, n_calib=10, group_col='session_id', sufixo='_z'
    )

    gb_session = df.groupby('session_id')
    df['interaction_critical_pat_z'] = df['is_senior_critical'] * df['pat_mean_z']

    # 7. Engenharia de Features Dinâmicas e Biofísicas
    df['sbp_delta'] = df['sbp_target'] - df['sbp_calib']
    df['pat_diff'] = df['pat_mean'] - df['pat_calib']

    df['pat_velocity'] = gb_session['pat_mean'].diff()
    df['pat_accel'] = gb_session['pat_mean'].diff().diff()
    df['pat_trend'] = gb_session['pat_mean'].diff(periods=3)

    df['gender_bin'] = df['gender'].map({'M': 0, 'F': 1})
    df['pat_lag1'] = gb_session['pat_mean_z'].shift(1)
    df['hr_lag1'] = gb_session['hr_mean_z'].shift(1)
    df['dynamic_ratio'] = df['pat_velocity'] * df['hr_mean_z']

    # --- PROCESSAMENTO DAS IDEIAS DO PAPER ---
    rho_blood = 1060
    df['omega_freq'] = df['hr_mean'] / 60.0

    if 'ppg_valley_intensity' in df.columns and 'ppg_ac_amplitude' in df.columns:
        df['alpha_womersley'] = (
            df['ppg_valley_intensity']
            * np.sqrt((rho_blood * df['omega_freq']) / (df['ppg_ac_amplitude'] + 1e-6))
        )

    if 'qtc_interval' not in df.columns and 'qt_interval' in df.columns:
        df['rr_interval_sec'] = 60.0 / (df['hr_mean'] + 1e-6)
        df['qtc_interval'] = df['qt_interval'] / np.sqrt(df['rr_interval_sec'] + 1e-6)

    if 'sdi_interaction' not in df.columns and 'qt_interval' in df.columns and 'tq_interval' in df.columns:
        df['sdi_interaction'] = df['qt_interval'] / (df['tq_interval'] + 1e-6)

    # ==============================================================================
    # 7.2 Âncoras morfológicas
    # ==============================================================================
    gb_subject = df.groupby('subject_id')

    features_ancora = [
        'alpha_womersley', 'qtc_interval', 'sdi_interaction', 'area_mean', 'w50',
        'qrs_duration',    'r_amplitude',  'qrs_amplitude',   'qrs_area',
        'ppg_crest_time',  'ppg_decay_time',
        'alpha_womersley_b',
    ]

    # NOTA: normalização por PACIENTE (todas as sessões). É uma personalização
    # intencional (assume uma calibração morfológica por paciente) e NÃO cria
    # fuga entre sujeitos — no GroupKFold/holdout cada paciente está inteiro de
    # um só lado. Se quiseres também torná-la estritamente causal em tempo real,
    # troca por normalizar_causal_calibracao(..., group_col='subject_id').
    cols_ancora = [c for c in features_ancora if c in df.columns]
    if causal_global_z:
        # CAUSAL: estatistica das primeiras n_calib_morf janelas DE CADA DOENTE.
        # E a unica versao computavel a cabeceira: um doente novo nao tem futuro.
        df = df.sort_values(['subject_id', 'session_id', 'bloco_id']).reset_index(drop=True)
        df = normalizar_causal_calibracao(df, cols_ancora, n_calib=n_calib_morf,
                                          group_col=grupo_ancoras, sufixo='_global_z')
        df = df.sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
    else:
        for col in cols_ancora:
            df[f'{col}_global_z'] = gb_subject[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-6)
            )
    # ==============================================================================

    # 8. Filtro de Comprimento Mínimo por Bloco
    min_len = window_size * 2
    df = df.groupby('session_id').filter(lambda x: len(x) >= min_len)

    # 9. Tratamento de NaNs das Colunas Temporais e Globais
    df = df.replace([np.inf, -np.inf], np.nan)

    colunas_com_nan_temporal = [
        'pat_velocity', 'pat_accel', 'pat_trend', 'pat_lag1', 'hr_lag1',
        'pat_slope_inicial', 'drift_personalizado', 'qtc_velocity', 'pat_calib_stability',
    ]
    df = df.dropna(subset=colunas_com_nan_temporal)

    # remove os NaN das colunas transformadas (o scikit-learn não os aceita)
    cols_ancora_criadas = [f'{c}_global_z' for c in features_ancora if f'{c}_global_z' in df.columns]
    df = df.dropna(subset=cols_ancora_criadas)

    print(f"Processamento Concluído: {df['subject_id'].nunique()} pacientes e {len(df)} janelas reais.")

    # 10. O cohort válido não se guarda em ficheiro, para esta função não ter efeitos secundários.

    return df
