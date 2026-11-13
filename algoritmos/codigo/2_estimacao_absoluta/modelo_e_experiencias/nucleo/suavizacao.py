"""Suavização causal das previsões, por sessão: só olha para trás (nenhuma amostra usa o futuro).

kalman  filtro de Kalman de nível local, com ganho adaptativo e rejeição de valores extremos da inovação (Huber)
ewma    média móvel exponencial com span=5 (a alternativa ao filtro de Kalman)
none    sem suavização
"""
import numpy as np
import pandas as pd

from . import config as C


def kalman_causal_1d(y, q_over_r=0.12, r=1.0, trend=False, robust_k=4.0):
    """Filtro de Kalman só para a frente. NaN em `y` conta como 'sem medição' (o estado propaga-se).
    q_over_r maior segue mais depressa, menor suaviza mais; robust_k recorta a inovação a k desvios-padrão."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    q = q_over_r * r
    primeiro = next((v for v in y if np.isfinite(v)), 0.0)
    if not trend:
        x, P = float(primeiro), 1e3
        for t in range(n):
            P = P + q
            z = y[t]
            if np.isfinite(z):
                S = P + r
                inov = z - x
                if robust_k is not None and S > 0:
                    lim = robust_k * np.sqrt(S)
                    inov = float(np.clip(inov, -lim, lim))
                K = P / S
                x = x + K * inov
                P = (1.0 - K) * P
            out[t] = x
        return out
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = q * np.array([[0.25, 0.5], [0.5, 1.0]])
    x, P, I = np.array([float(primeiro), 0.0]), np.eye(2) * 1e3, np.eye(2)
    for t in range(n):
        x = F @ x
        P = F @ P @ F.T + Q
        z = y[t]
        if np.isfinite(z):
            S = float((H @ P @ H.T)[0, 0] + r)
            inov = float(z - (H @ x)[0])
            if robust_k is not None and S > 0:
                lim = robust_k * np.sqrt(S)
                inov = float(np.clip(inov, -lim, lim))
            K = (P @ H.T) / S
            x = x + K[:, 0] * inov
            P = (I - K @ H) @ P
        out[t] = x[0]
    return out


def suavizar_por_sessao(df, coluna, metodo=C.SUAVIZACAO, grupo="session_id"):
    """Aplica o método a cada sessão de `coluna` e devolve uma Series alinhada com o índice de `df`."""
    metodo = (metodo or "kalman").lower()
    if metodo == "kalman":
        saida = pd.Series(np.nan, index=df.index, dtype=float)
        for _, idx in df.groupby(grupo, sort=False).groups.items():
            saida.loc[idx] = kalman_causal_1d(df.loc[idx, coluna].to_numpy(dtype=float), **C.KALMAN)
        return saida
    if metodo == "ewma":
        return df.groupby(grupo, sort=False)[coluna].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    if metodo in ("none", "nenhum"):
        return df[coluna].copy()
    raise ValueError(f"método de suavização desconhecido: {metodo!r}")
