"""
Suavização causal, normalização causal e separação dev/holdout para o pipeline de estimação de PA.

Contém apenas utilitários SEM dependência dos modelos:

  1. Filtro de Kalman causal 1D (substitui a suavização EWMA do pós-processamento).
  2. Normalização z causal por janela de calibração (remove o look-ahead das
     estatísticas de sessão inteira).
  3. Separação dev/holdout por sujeito (conjunto de teste bloqueado).
"""

import numpy as np
import pandas as pd


# =====================================================================
# 1. FILTRO DE KALMAN CAUSAL 1D
# =====================================================================
def kalman_causal_1d(y, q_over_r=0.12, r=1.0, trend=False, robust_k=4.0):
    """
    Filtro de Kalman causal (só passa para a frente — sem lookahead) para
    suavizar uma série de previsões `preds_delta` dentro de uma sessão.

    É um substituto principiado do EWMA: em vez de um ganho fixo, o ganho de
    Kalman adapta-se e (com `robust_k`) rejeita picos/outliers da previsão.

    Parâmetros
    ----------
    q_over_r : razão ruído-de-processo / ruído-de-medição. Maior => segue mais
               depressa (menos suave); menor => mais suave. O default ~0.12
               dá um ganho estacionário próximo do EWMA span=5 (alpha~0.33).
    trend    : se True usa modelo nível+velocidade (segue drift linear sem
               atraso); se False usa modelo de nível local (mais seguro/estável).
    robust_k : recorta a inovação a robust_k desvios-padrão (Huber-like). None
               desativa a robustez. Evita que um outlier na previsão desloque a
               estimativa suavizada.

    Devolve um np.ndarray com a série filtrada (mesmo comprimento de `y`).
    NaNs em `y` são tratados como "sem medição" (o filtro apenas propaga o estado).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = np.full(n, np.nan)
    if n == 0:
        return out

    q = q_over_r * r

    if not trend:
        # ---- Modelo de nível local (random walk + ruído) ----
        # Inicializa no primeiro valor finito
        finito0 = next((v for v in y if np.isfinite(v)), 0.0)
        x = float(finito0)
        P = 1e3
        for t in range(n):
            # Predição
            P = P + q
            z = y[t]
            if np.isfinite(z):
                S = P + r
                innov = z - x
                if robust_k is not None and S > 0:
                    lim = robust_k * np.sqrt(S)
                    innov = float(np.clip(innov, -lim, lim))
                K = P / S
                x = x + K * innov
                P = (1.0 - K) * P
            out[t] = x
        return out

    # ---- Modelo nível + velocidade (segue drift linear) ----
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = q * np.array([[0.25, 0.5], [0.5, 1.0]])  # ruído branco em aceleração
    finito0 = next((v for v in y if np.isfinite(v)), 0.0)
    x = np.array([float(finito0), 0.0])
    P = np.eye(2) * 1e3
    I = np.eye(2)
    for t in range(n):
        # Predição
        x = F @ x
        P = F @ P @ F.T + Q
        z = y[t]
        if np.isfinite(z):
            S = float((H @ P @ H.T)[0, 0] + r)
            innov = float(z - (H @ x)[0])
            if robust_k is not None and S > 0:
                lim = robust_k * np.sqrt(S)
                innov = float(np.clip(innov, -lim, lim))
            K = (P @ H.T) / S            # (2,1)
            x = x + K[:, 0] * innov
            P = (I - K @ H) @ P
        out[t] = x[0]
    return out


def aplicar_kalman_por_sessao(df, col_in, col_out=None, group_col="session_id",
                              q_over_r=0.12, trend=False, robust_k=4.0):
    """
    Aplica `kalman_causal_1d` independentemente a cada sessão, preservando o
    índice do DataFrame. Substitui diretamente o padrão:

        df.groupby('session_id')[col].transform(lambda x: x.ewm(span=5).mean())

    Devolve uma pd.Series alinhada com df.index.
    """
    if col_out is None:
        col_out = col_in
    resultado = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby(group_col, sort=False).groups.items():
        vals = df.loc[idx, col_in].to_numpy(dtype=float)
        filt = kalman_causal_1d(vals, q_over_r=q_over_r, trend=trend, robust_k=robust_k)
        resultado.loc[idx] = filt
    return resultado


def aplicar_suavizacao_por_sessao(df, col_in, metodo="kalman", col_out=None,
                                  group_col="session_id", span=5,
                                  q_over_r=0.12, trend=False, robust_k=4.0):
    """
    Suavização causal por sessão de `col_in`, escolhendo o método:

      - 'kalman' : filtro de Kalman causal (aplicar_kalman_por_sessao), com ganho
                   adaptativo e rejeição de outliers. Default = comportamento atual.
      - 'ewma'   : média móvel exponencial causal, ewm(span=..., adjust=False) —
                   reproduz o pós-processamento antigo (span=5, sem lookahead).
      - 'none'   : sem suavização (devolve a coluna tal como está).

    Permite ablações "com vs sem Kalman" mudando um único argumento, sem tocar no
    resto do pipeline. Devolve uma pd.Series alinhada com df.index.
    """
    if col_out is None:
        col_out = col_in
    metodo = (metodo or "kalman").lower()
    if metodo == "kalman":
        return aplicar_kalman_por_sessao(df, col_in, col_out=col_out, group_col=group_col,
                                         q_over_r=q_over_r, trend=trend, robust_k=robust_k)
    if metodo == "ewma":
        return (df.groupby(group_col, sort=False)[col_in]
                  .transform(lambda x: x.ewm(span=span, adjust=False).mean()))
    if metodo in ("none", "nenhum", "raw"):
        return df[col_in].copy()
    raise ValueError(f"metodo de suavização desconhecido: {metodo!r} "
                     f"(usa 'kalman', 'ewma' ou 'none')")


# =====================================================================
# 2. NORMALIZAÇÃO Z CAUSAL POR JANELA DE CALIBRAÇÃO
# =====================================================================
def normalizar_causal_calibracao(df, cols, n_calib=10, group_col="session_id",
                                 sufixo="_z", eps=1e-6):
    """
    Z-score usando SÓ a média/desvio das primeiras `n_calib` linhas de cada
    grupo (a janela de calibração), em vez das estatísticas da sessão inteira.

    Isto remove o look-ahead: a normalização de qualquer janela passa a depender
    apenas do arranque da sessão — coerente com a baseline `sbp_calib`/`pat_calib`,
    que já usa `get_ultra_robust_baseline` sobre as primeiras janelas.

    IMPORTANTE: `df` tem de estar ordenado cronologicamente dentro de cada grupo
    (ex.: sort por ['session_id', 'bloco_id']) ANTES de chamar esta função, para
    que `head(n_calib)` corresponda mesmo ao início da sessão.

    Cria colunas `{col}{sufixo}`. Devolve o df (modificado in-place e devolvido).
    """
    cols = [c for c in cols if c in df.columns]
    head = df.groupby(group_col, sort=False).head(n_calib)
    mu = head.groupby(group_col)[cols].mean()
    sd = head.groupby(group_col)[cols].std()

    for col in cols:
        # desvio de calibração; onde for NaN (grupo com <2 janelas) ou ~0,
        # cai para a mediana dos desvios entre grupos (fallback estável).
        sd_col = sd[col]
        fallback = sd_col.median()
        if not np.isfinite(fallback) or fallback < eps:
            fallback = 1.0
        sd_col = sd_col.fillna(fallback).replace(0.0, fallback)

        m = df[group_col].map(mu[col])
        s = df[group_col].map(sd_col)
        df[col + sufixo] = (df[col] - m) / (s + eps)
    return df


# =====================================================================
# 3. SEPARAÇÃO DEV / HOLDOUT POR SUJEITO (TESTE BLOQUEADO)
# =====================================================================
def separar_dev_holdout(df, ids_teste, col="subject_id"):
    """
    Divide `df` em (dev, teste) por sujeito, de forma disjunta. Os IDs em
    `ids_teste` NUNCA entram no desenvolvimento (otimização + CV), só na
    avaliação final única.

    Devolve (df_dev, df_test, info) onde info documenta quantos IDs pedidos
    estavam realmente presentes no dataset.
    """
    ids_teste = set(map(str, ids_teste))
    presentes = set(df[col].astype(str).unique())
    usados = ids_teste & presentes
    faltam = ids_teste - presentes

    mask_test = df[col].astype(str).isin(usados)
    df_test = df[mask_test].copy()
    df_dev = df[~mask_test].copy()

    info = {
        "n_ids_pedidos": len(ids_teste),
        "n_ids_no_teste": len(usados),
        "n_ids_ausentes": len(faltam),
        "ids_ausentes": sorted(faltam),
        "n_sujeitos_dev": df_dev[col].nunique(),
        "n_sujeitos_teste": df_test[col].nunique(),
        "n_janelas_dev": len(df_dev),
        "n_janelas_teste": len(df_test),
    }
    return df_dev, df_test, info


# =====================================================================
# AUTO-TESTE (corre com:  python suavizacao_e_normalizacao_causal.py)
# =====================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    falhas = 0

    def checa(cond, msg):
        global falhas
        estado = "OK  " if cond else "FALHA"
        if not cond:
            falhas += 1
        print(f"  [{estado}] {msg}")

    print("== 1. Kalman: causalidade (saída em t não depende de t+1) ==")
    base = np.cumsum(rng.normal(0, 1, 60))
    ruido = base + rng.normal(0, 2, 60)
    f_full = kalman_causal_1d(ruido)
    # altera só a última amostra -> só a última saída pode mudar
    ruido2 = ruido.copy(); ruido2[-1] += 50
    f_mod = kalman_causal_1d(ruido2)
    checa(np.allclose(f_full[:-1], f_mod[:-1]), "mudar y[-1] não altera saídas anteriores")

    print("== 2. Kalman vs EWMA num sinal com outliers (robustez) ==")
    verdade = np.linspace(-5, 8, 80)            # drift lento verdadeiro
    obs = verdade + rng.normal(0, 1.5, 80)
    idx_out = rng.choice(80, 6, replace=False)  # outliers grandes
    obs[idx_out] += rng.choice([-1, 1], 6) * 25
    ewma = pd.Series(obs).ewm(span=5, adjust=False).mean().to_numpy()
    kal_lvl = kalman_causal_1d(obs, trend=False)
    kal_trd = kalman_causal_1d(obs, trend=True)
    mae_ewma = np.mean(np.abs(ewma - verdade))
    mae_lvl = np.mean(np.abs(kal_lvl - verdade))
    mae_trd = np.mean(np.abs(kal_trd - verdade))
    print(f"    MAE EWMA={mae_ewma:.3f}  Kalman-nível={mae_lvl:.3f}  Kalman-trend={mae_trd:.3f}")
    checa(mae_lvl <= mae_ewma, "Kalman nível-local não é pior que EWMA (sinal com outliers)")
    checa(mae_trd <= mae_ewma, "Kalman com trend segue o drift melhor que EWMA")

    print("== 3. Kalman: ganho estacionário do nível-local ~ EWMA span=5 ==")
    passo = np.zeros(200); passo[50:] = 10.0
    resp = kalman_causal_1d(passo, trend=False, robust_k=None)
    # aproxima o alpha efetivo pela resposta ao degrau
    alpha_ef = resp[51] / 10.0
    print(f"    alpha efetivo ~ {alpha_ef:.3f} (EWMA span=5 -> 0.333)")
    checa(0.15 < alpha_ef < 0.5, "ganho na gama esperada")

    print("== 4. NaNs no Kalman (propaga estado sem rebentar) ==")
    comnan = obs.copy(); comnan[10:13] = np.nan
    fn = kalman_causal_1d(comnan, trend=False)
    checa(np.all(np.isfinite(fn)), "saída finita mesmo com NaNs na entrada")

    print("== 5. Normalização causal por janela de calibração ==")
    df = pd.DataFrame({
        "session_id": ["s1"] * 30 + ["s2"] * 30,
        "bloco_id": list(range(30)) + list(range(30)),
        "pat_mean": np.r_[rng.normal(0.30, 0.02, 30), rng.normal(0.45, 0.02, 30)],
    }).sort_values(["session_id", "bloco_id"]).reset_index(drop=True)
    df = normalizar_causal_calibracao(df, ["pat_mean"], n_calib=10)
    # a média/desvio usados têm de vir SÓ das 10 primeiras de cada sessão
    for s in ["s1", "s2"]:
        sub = df[df.session_id == s]
        mu = sub["pat_mean"].head(10).mean()
        sd = sub["pat_mean"].head(10).std()
        esperado = (sub["pat_mean"] - mu) / (sd + 1e-6)
        checa(np.allclose(sub["pat_mean_z"], esperado), f"z causal correto na sessão {s}")
    # confirma que difere da versão com estatística da sessão inteira
    mu_all = df[df.session_id == "s1"]["pat_mean"].mean()
    z_all = (df[df.session_id == "s1"]["pat_mean"] - mu_all) / (df[df.session_id == "s1"]["pat_mean"].std() + 1e-6)
    checa(not np.allclose(df[df.session_id == "s1"]["pat_mean_z"], z_all),
          "z causal difere do z de sessão inteira (removeu look-ahead)")

    print("== 6. Separação dev/holdout disjunta por sujeito ==")
    dfp = pd.DataFrame({"subject_id": [f"p{i//10}" for i in range(60)], "x": range(60)})
    dev, test, info = separar_dev_holdout(dfp, ["p0", "p2", "pX"], col="subject_id")
    inter = set(dev.subject_id) & set(test.subject_id)
    checa(len(inter) == 0, "sem sobreposição de sujeitos entre dev e teste")
    checa(set(test.subject_id) == {"p0", "p2"}, "teste contém exatamente os IDs pedidos presentes")
    checa(info["n_ids_ausentes"] == 1 and info["ids_ausentes"] == ["pX"], "reporta ID ausente (pX)")

    print()
    if falhas == 0:
        print("TODOS OS TESTES PASSARAM")
    else:
        print(f"{falhas} TESTE(S) FALHARAM")
        raise SystemExit(1)
