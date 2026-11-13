"""Dados do absoluto: UMA só derivação, usada por todos os scripts.

É a única derivação de `carregar_e_processar_todos_deltas` e de `injetar_ajuste_variancia_fisiologica` e a única definição das listas de características. Os parâmetros são:

  ancoras = "causal"        as 12 âncoras morfológicas são normalizadas pela janela de calibração da sessão
                            (primeiros N_BLOCOS_CALIBRACAO blocos), a mesma do PAT e da FC e a da tendência.
  ancoras = "doente_inteiro" normalização com o doente todo (olha para o futuro). Só existe para medir o custo da
                            normalização causal (08_custo_normalizacao_causal.py).
  min_blocos                comprimento mínimo da sessão: a constante MIN_BLOCOS_SESSAO (20, como na tendência), independente do `window_size` do carregador.

Devolve a tabela por bloco de 15 s (3 janelas de 5 s colapsadas), com uma linha por (sujeito, sessão, bloco).
"""
import glob
import hashlib
import os

import numpy as np
import pandas as pd

from . import config as C


# ── funções auxiliares da derivação ───────────────────────────────────────────────────
def _calibracao_robusta(valores):
    """Média do bloco de 5 blocos mais estável (menor coeficiente de variação) entre os 10 primeiros."""
    cabeca = valores.head(10).tolist()
    n = len(cabeca)
    if n < 5:
        return np.mean(cabeca) if n > 0 else 0
    melhor, cv_min = np.mean(cabeca[:5]), np.std(cabeca[:5]) / (np.mean(cabeca[:5]) + 1e-6)
    for i in range(n - 5 + 1):
        sub = cabeca[i:i + 5]
        m = np.mean(sub)
        cv = np.std(sub) / (m + 1e-6)
        if cv < cv_min:
            cv_min, melhor = cv, m
    return melhor


def normalizar_por_calibracao(df, cols, grupo, n_calib=C.N_BLOCOS_CALIBRACAO, sufixo="_z", eps=1e-6):
    """z-score com a média e o desvio das primeiras `n_calib` linhas de cada grupo (a janela de calibração).
    `df` tem de vir ordenado no tempo dentro de cada grupo. Cria as colunas `<col><sufixo>`."""
    cols = [c for c in cols if c in df.columns]
    cabeca = df.groupby(grupo, sort=False).head(n_calib)
    mu = cabeca.groupby(grupo)[cols].mean()
    sd = cabeca.groupby(grupo)[cols].std()
    for col in cols:
        sd_col = sd[col]
        reserva = sd_col.median()
        if not np.isfinite(reserva) or reserva < eps:
            reserva = 1.0
        sd_col = sd_col.fillna(reserva).replace(0.0, reserva)
        df[col + sufixo] = (df[col] - df[grupo].map(mu[col])) / (df[grupo].map(sd_col) + eps)
    return df


def _declive_pat_inicial(grupo, n=8):
    calib = grupo.head(n)
    t, pat = calib['tempo_desde_calib_seg'].values, calib['pat_mean'].values
    if len(t) < 4 or np.std(t) < 1e-6:
        return np.nan
    return float(np.polyfit(t, pat, 1)[0])


def _declive_movel_5(serie):
    res = np.full(len(serie), np.nan)
    v = serie.values
    for i in range(4, len(v)):
        y = v[i - 4:i]
        res[i] = np.polyfit(np.arange(4, dtype=float), y, 1)[0] if np.std(y) > 1e-8 else 0.0
    return pd.Series(res, index=serie.index)


def _variancia_fisiologica(df):
    """pat_variance_ratio: desvio acumulado do PAT-diff sobre o desvio dos 5 primeiros blocos da sessão."""
    partes = []
    for _, g in df.groupby('session_id'):
        g = g.copy()
        acum = g['pat_diff'].expanding(min_periods=5).std().fillna(1e-5)
        ini = g['pat_diff'].head(5).std()
        if pd.isna(ini) or ini < 1e-5:
            ini = 1.0
        g['pat_variance_ratio'] = acum / ini
        partes.append(g)
    return pd.concat(partes, axis=0).sort_values(by=['subject_id', 'session_id', 'bloco_id']).reset_index(drop=True)


# ── leitura e colapso ─────────────────────────────────────────────────────────────────────────────────────────────
def _ler_csv(variante):
    ficheiros = sorted(glob.glob(os.path.join(C.DIR_FEATURES, variante.padrao)))
    if not ficheiros:
        raise FileNotFoundError(f"sem ficheiros para {variante.padrao!r} em {C.DIR_FEATURES}")
    partes = []
    for f in ficheiros:
        d = pd.read_csv(f)
        if d.empty:
            continue
        if 'gender' not in d.columns:
            d['gender'] = 'M' if 'homens' in f.lower() else 'F'
        partes.append(d.rename(columns={'subject_folder': 'subject_id', 'segment': 'segment_name'}))
    df = pd.concat(partes, ignore_index=True)
    df['subject_id'] = df['subject_id'].astype(str)
    return df


def _fundir(df, fundir):
    """Junta ao CSV de características uma tabela extra, ANTES de colapsar."""
    n = len(df)
    if fundir == "clinicas":
        clin = pd.read_csv(C.CSV_CLINICAS).rename(columns={"subject_id_str": "subject_id"})
        clin["record_id"] = clin["record_id"].astype(str)
        clin = clin[["subject_id", "record_id", "segment_order", "window_idx", *C.COLUNAS_CLINICAS]]
        df['_chave_registo'] = df['segment_name'].astype(str).str.rsplit("_", n=1).str[0]
        df = df.merge(clin, left_on=["subject_id", "_chave_registo", "segment_order", "window_idx"],
                      right_on=["subject_id", "record_id", "segment_order", "window_idx"], how="left")
        df = df.drop(columns=["record_id", "_chave_registo"])
    elif fundir == "diagnostico":
        diag = pd.read_csv(C.CSV_DIAGNOSTICO).rename(columns={"subject_id_str": "subject_id"})
        df = df.merge(diag[["subject_id", "is_cardiac", "is_respiratory"]], on="subject_id", how="left")
        df["is_cardiac"] = df["is_cardiac"].fillna(0).astype(int)
        df["is_respiratory"] = df["is_respiratory"].fillna(0).astype(int)
    else:
        raise ValueError(fundir)
    assert len(df) == n, f"a junção '{fundir}' alterou o número de linhas: {n} -> {len(df)}"
    return df


def _colapsar_e_sessoes(df):
    df['record_id'] = df['segment_name'].astype(str).apply(lambda x: x.rsplit('_', 1)[0])
    df['bloco_id'] = df['window_idx'] // 3                      # 3 janelas de 5 s -> 1 bloco de 15 s
    meta = ['subject_id', 'record_id', 'segment_order', 'bloco_id', 'gender', 'dataset']
    meta += [c for c in ('age', 'is_cardiac', 'is_respiratory') if c in df.columns]
    medias = [c for c in df.columns if c not in meta and c not in ('segment_name', 'window_idx')]
    df = df.groupby(meta)[medias].mean().reset_index()
    df = df.sort_values(by=['subject_id', 'record_id', 'segment_order', 'bloco_id']).reset_index(drop=True)
    # nova sessão quando há um buraco de mais de um segmento ou de mais de 4 blocos
    df['seg_diff'] = df.groupby('record_id')['segment_order'].diff()
    df['win_diff'] = df.groupby(['record_id', 'segment_order'])['bloco_id'].diff()
    nova = (df['seg_diff'] > 1) | (df['win_diff'] > 4) | df['seg_diff'].isna() | df['win_diff'].isna()
    df['is_new_session'] = nova
    df['session_id'] = df['record_id'] + "_" + df.groupby('record_id')['is_new_session'].cumsum().astype(str)
    return df.sort_values(by=['session_id', 'bloco_id']).reset_index(drop=True)


# ── derivação ─────────────────────────────────────────────────────────────────────────────────────────────────────
def derivar(variante, ancoras="causal", min_blocos=C.MIN_BLOCOS_SESSAO):
    """Da leitura dos CSV à tabela por bloco, com todas as características. Sem cache (ver `carregar`)."""
    if ancoras not in ("causal", "doente_inteiro"):
        raise ValueError(ancoras)
    df = _ler_csv(variante)
    if variante.fundir:
        df = _fundir(df, variante.fundir)
    df = _colapsar_e_sessoes(df)

    # relógio desde a calibração e características anti-deriva (calculadas na janela de calibração da sessão)
    df['tempo_desde_calib_seg'] = (df['bloco_id'] - df.groupby('session_id')['bloco_id'].transform('min')) * 15
    df['tempo_pat_interaction'] = df['tempo_desde_calib_seg'] * df['pat_mean']
    declive = df.groupby('session_id').apply(_declive_pat_inicial, include_groups=False).reset_index(name='pat_slope_inicial')
    df = df.merge(declive, on='session_id', how='left')
    df['pat_slope_inicial'] = df['pat_slope_inicial'].fillna(0.0)
    df['drift_personalizado'] = df['pat_slope_inicial'] * df['tempo_desde_calib_seg']
    if 'qtc_interval' in df.columns:
        df['qtc_velocity'] = (df.groupby('session_id')['qtc_interval']
                              .transform(lambda x: _declive_movel_5(x.shift(1).bfill())).fillna(0.0))
    else:
        df['qtc_velocity'] = 0.0
    estab = df.groupby('session_id').apply(lambda g: float(g['pat_mean'].head(8).std()), include_groups=False)
    df = df.merge(estab.reset_index(name='pat_calib_stability'), on='session_id', how='left')
    df['pat_calib_stability'] = df['pat_calib_stability'].fillna(df['pat_calib_stability'].median())
    mediana_movel = df.groupby('session_id')['pat_mean'].apply(lambda x: x.shift(1).rolling(4, min_periods=1).median())
    df['pat_mean_baseline_adaptativo'] = mediana_movel.reset_index(level=0, drop=True).fillna(df['pat_mean'])
    df['pat_mean_diff_adaptativo'] = df['pat_mean'] - df['pat_mean_baseline_adaptativo']
    df['pat_mean_ewma_longo'] = df.groupby('session_id')['pat_mean'].transform(lambda x: x.ewm(span=12, adjust=False).mean())

    # limites fisiológicos e calibração com manguito (sbp_calib, pat_calib)
    df = df[(df['sbp_target'] > 70) & (df['sbp_target'] < 190)]
    df = df[(df['pat_mean'] > 0.15) & (df['pat_mean'] < 0.60)]
    calib = (df.groupby('session_id').agg({'sbp_target': _calibracao_robusta, 'pat_mean': _calibracao_robusta})
             .reset_index().rename(columns={'sbp_target': 'sbp_calib', 'pat_mean': 'pat_calib'}))
    df = df.merge(calib, on='session_id')

    # demográficas e z-score do PAT e da FC pela janela de calibração da sessão (causal)
    df['age_pat_interaction'] = df['age'] * df['pat_mean']
    df['age_squared'] = df['age'] ** 2
    df['is_senior_critical'] = ((df['age'] >= 60) & (df['age'] <= 75)).astype(int)
    df = normalizar_por_calibracao(df, ['pat_mean', 'hr_mean'], grupo='session_id', sufixo='_z')
    df['interaction_critical_pat_z'] = df['is_senior_critical'] * df['pat_mean_z']

    # dinâmicas
    gs = df.groupby('session_id')
    df['sbp_delta'] = df['sbp_target'] - df['sbp_calib']
    df['pat_diff'] = df['pat_mean'] - df['pat_calib']
    df['pat_velocity'] = gs['pat_mean'].diff()
    df['pat_accel'] = gs['pat_mean'].diff().diff()
    df['pat_trend'] = gs['pat_mean'].diff(periods=3)
    df['gender_bin'] = df['gender'].map({'M': 0, 'F': 1})
    df['pat_lag1'] = gs['pat_mean_z'].shift(1)
    df['hr_lag1'] = gs['hr_mean_z'].shift(1)
    df['dynamic_ratio'] = df['pat_velocity'] * df['hr_mean_z']

    # Womersley, QTc e SDI
    df['omega_freq'] = df['hr_mean'] / 60.0
    if 'ppg_valley_intensity' in df.columns and 'ppg_ac_amplitude' in df.columns:
        df['alpha_womersley'] = df['ppg_valley_intensity'] * np.sqrt((1060 * df['omega_freq']) / (df['ppg_ac_amplitude'] + 1e-6))
    if 'qtc_interval' not in df.columns and 'qt_interval' in df.columns:
        df['rr_interval_sec'] = 60.0 / (df['hr_mean'] + 1e-6)
        df['qtc_interval'] = df['qt_interval'] / np.sqrt(df['rr_interval_sec'] + 1e-6)
    if 'sdi_interaction' not in df.columns and 'qt_interval' in df.columns and 'tq_interval' in df.columns:
        df['sdi_interaction'] = df['qt_interval'] / (df['tq_interval'] + 1e-6)

    # âncoras morfológicas -> <nome>_global_z
    cols = [c for c in (*C.ANCORAS, *variante.ancoras_extra) if c in df.columns]
    if ancoras == "causal":                    # janela de calibração da sessão, a mesma do PAT e da FC (e da tendência)
        df = normalizar_por_calibracao(df, cols, grupo='session_id', sufixo='_global_z')
    else:
        gd = df.groupby('subject_id')
        for c in cols:
            df[f'{c}_global_z'] = gd[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))

    # sessões curtas e linhas incompletas
    df = df.groupby('session_id').filter(lambda x: len(x) >= min_blocos)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['pat_velocity', 'pat_accel', 'pat_trend', 'pat_lag1', 'hr_lag1', 'pat_slope_inicial',
                           'drift_personalizado', 'qtc_velocity', 'pat_calib_stability'])
    df = df.dropna(subset=[f'{c}_global_z' for c in cols])
    return _variancia_fisiologica(df.dropna(subset=[C.ALVO]).copy())


# ── cache e partição ──────────────────────────────────────────────────────────────────────────────────────────────
def _assinatura(variante, ancoras, min_blocos):
    h = hashlib.md5()
    for f in (__file__, C.__file__):
        h.update(open(f, 'rb').read())
    for f in sorted(glob.glob(os.path.join(C.DIR_FEATURES, variante.padrao))):
        h.update(f"{os.path.basename(f)}{os.path.getsize(f)}{int(os.path.getmtime(f))}".encode())
    h.update(f"{variante.fundir}{variante.ancoras_extra}{ancoras}{min_blocos}".encode())
    return h.hexdigest()[:12]


def carregar_tabela(variante, ancoras="causal", min_blocos=C.MIN_BLOCOS_SESSAO):
    """Tabela completa (dev + holdout) já derivada, com cache em disco."""
    os.makedirs(C.DIR_CACHE, exist_ok=True)
    base = f"{C.DIR_CACHE}/{variante.nome}__{ancoras}__min{min_blocos}"
    sig = _assinatura(variante, ancoras, min_blocos)
    if os.path.exists(base + ".sig") and open(base + ".sig").read() == sig:
        return pd.read_pickle(base + ".pkl")
    print(f"[dados] a derivar {variante.nome} (ancoras={ancoras}, min_blocos={min_blocos}) ...", flush=True)
    df = derivar(variante, ancoras, min_blocos)
    df.to_pickle(base + ".pkl")
    open(base + ".sig", "w").write(sig)
    return df


def ids_holdout():
    with open(C.IDS_HOLDOUT) as f:
        return [l.strip() for l in f if l.strip()]


def separar(df, ids_teste=None):
    """Separa por sujeito: os IDs do holdout nunca entram no desenvolvimento. Devolve (dev, teste, info)."""
    ids = set(map(str, ids_teste if ids_teste is not None else ids_holdout()))
    no_teste = df['subject_id'].astype(str).isin(ids)
    dev, teste = df[~no_teste].copy(), df[no_teste].copy()
    info = dict(n_ids_pedidos=len(ids), n_sujeitos_dev=int(dev.subject_id.nunique()), n_sujeitos_teste=int(teste.subject_id.nunique()),
                n_janelas_dev=len(dev), n_janelas_teste=len(teste),
                ids_ausentes=sorted(ids - set(df['subject_id'].astype(str))))
    return dev, teste, info


def carregar(variante, ancoras="causal", min_blocos=C.MIN_BLOCOS_SESSAO, features=None):
    """(dev, teste, info) prontos a treinar. `features` por omissão são as da variante; as obrigatórias não podem ter NaN."""
    if isinstance(variante, str):
        variante = C.VARIANTES[variante]
    df = carregar_tabela(variante, ancoras, min_blocos)
    dev, teste, info = separar(df)
    feats = list(features if features is not None else variante.features)
    obrigatorias = [f for f in feats if f not in variante.opcionais] + [C.ALVO]
    dev = dev.dropna(subset=obrigatorias).reset_index(drop=True)
    teste = teste.dropna(subset=obrigatorias).reset_index(drop=True)
    info.update(n_sujeitos_dev=int(dev.subject_id.nunique()), n_sujeitos_teste=int(teste.subject_id.nunique()),
                n_janelas_dev=len(dev), n_janelas_teste=len(teste))
    return dev, teste, info
