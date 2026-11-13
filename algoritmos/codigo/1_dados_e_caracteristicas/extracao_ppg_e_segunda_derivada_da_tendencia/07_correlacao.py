"""
07 — Funções de correlação no espaço de variação (Parte B), sobre o dataset
PPG-PURO.

Usa só dados que um dispositivo sem elétrodos consegue produzir. Responde a estas perguntas:

  B1  Quanto tempo cada sinal se "lembra" de si próprio? Daí sai o fator de
      inflação da etiqueta e o número de EVENTOS INDEPENDENTES — o teto
      estatístico real da tese, que não depende do modelo.
  B3  As features têm sinal DENTRO da sessão (o que interessa clinicamente) ou
      só distinguem pacientes entre si?
  B4  O score antecipa o evento ou apenas o deteta?

Saída: analise_correlacao/
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.abspath(__file__))
OUT = f"{RAIZ}/analise_correlacao"
os.makedirs(OUT, exist_ok=True)

BLOCO_SEG = 15
D, W_MIN = 15, 5
W = int(W_MIN * 60 / BLOCO_SEG)
K_MAX = 240          # 60 min de desfasamento
K_X = 80             # +-20 min para a correlação cruzada

AZUL, VERM, VERDE, LARANJA = "#1f5fa8", "#c0392b", "#0a7d54", "#e08214"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 170,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 9})

log_linhas = []


def log(s=""):
    print(s, flush=True)
    log_linhas.append(s)


def serie_concatenada(d, coluna, k_max=K_MAX):
    """Concatena as sessões separadas por NaN, para o desfasamento nunca
    atravessar a fronteira entre sessões."""
    pedacos, sep = [], np.full(k_max + 1, np.nan)
    for _, g in d.groupby("session_id", sort=False):
        g = g.sort_values("bloco_id")
        slot = g.bloco_id.values.astype(int)
        slot = slot - slot.min()
        if slot.max() > 5e5:
            continue
        a = np.full(slot.max() + 1, np.nan)
        a[slot] = g[coluna].values
        pedacos.append(a)
        pedacos.append(sep)
    return np.concatenate(pedacos) if pedacos else np.array([])


def funcoes_correlacao(x, k_max=K_MAX):
    acf = np.empty(k_max + 1)
    for k in range(k_max + 1):
        a = x[:len(x) - k] if k else x
        b = x[k:]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 50:
            acf[k] = np.nan
            continue
        u, v = a[m], b[m]
        su, sv = u.std(), v.std()
        acf[k] = (np.mean((u - u.mean()) * (v - v.mean())) / (su * sv)
                  if su > 0 and sv > 0 else np.nan)
    return acf


def tempo_correlacao(acf):
    idx = np.where(acf < 1 / np.e)[0]
    t_e = idx[0] * BLOCO_SEG / 60.0 if len(idx) else np.nan
    z = np.where(acf <= 0)[0]
    kz = z[0] if len(z) else len(acf)
    return t_e, (1 + 2 * np.nansum(acf[1:kz]))


def correlacao_cruzada(d, ca, cb, k=K_X):
    a, b = serie_concatenada(d, ca), serie_concatenada(d, cb)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    out = np.full(2 * k + 1, np.nan)
    for i, L in enumerate(range(-k, k + 1)):
        u, v = (a[:n - L], b[L:]) if L >= 0 else (a[-L:], b[:n + L])
        m = np.isfinite(u) & np.isfinite(v)
        if m.sum() < 100 or u[m].std() == 0 or v[m].std() == 0:
            continue
        out[i] = np.corrcoef(u[m], v[m])[0, 1]
    return out


def main():
    t0 = time.time()
    f = f"{RAIZ}/previsoes_ppg_puro_descida.csv.gz"
    if not os.path.exists(f):
        log(f"Falta {f} — correr primeiro 04_avaliar_e_exportar.py")
        return

    hold = pd.read_csv(f).sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
    hold = hold.rename(columns={'p': 'score'})
    hold['residuo'] = hold['y'] - hold['score']

    # Δ e PAS absoluta vêm do dataset (as previsões só têm y/p)
    base = pd.read_csv(f"{RAIZ}/dados/holdout_ppg_puro.csv",
                       usecols=['session_id', 'bloco_id', 'sbp_target'])
    hold = hold.merge(base, on=['session_id', 'bloco_id'], how='left')
    hold['delta'] = hold['sbp_target'] - hold.groupby('session_id')['sbp_target'].shift(W)

    log("=" * 78)
    log("PARTE B (PPG-PURO) — FUNÇÕES DE CORRELAÇÃO NO ESPAÇO DE VARIAÇÃO")
    log("=" * 78)
    log(f"holdout: {len(hold)} blocos | {hold.session_id.nunique()} sessões | "
        f"{int(hold.y.sum())} janelas-evento ({100*hold.y.mean():.2f}%)")
    log("")

    # ── B1 ───────────────────────────────────────────────────────────────────
    sinais = {
        "PAS absoluta (referência)": "sbp_target",
        "Δ PAS em 5 min (alvo real)": "delta",
        "etiqueta do evento (0/1)": "y",
        "score do modelo (PPG-puro)": "score",
        "resíduo (etiqueta − score)": "residuo",
    }
    cores = {"PAS absoluta (referência)": "#999999", "Δ PAS em 5 min (alvo real)": AZUL,
             "etiqueta do evento (0/1)": VERM, "score do modelo (PPG-puro)": VERDE,
             "resíduo (etiqueta − score)": LARANJA}

    res = {}
    for etq, col in sinais.items():
        res[etq] = funcoes_correlacao(serie_concatenada(hold, col))
    lags_min = np.arange(K_MAX + 1) * BLOCO_SEG / 60.0

    tab = []
    for etq, acf in res.items():
        t_e, tau = tempo_correlacao(acf)
        tab.append(dict(sinal=etq, acf_1min=acf[4], acf_5min=acf[20], acf_15min=acf[60],
                        acf_30min=acf[120], acf_60min=acf[K_MAX],
                        tempo_1_sobre_e_min=t_e, fator_inflacao=tau))
    tab = pd.DataFrame(tab)
    tab.to_csv(f"{OUT}/B1_tempos_de_correlacao.csv", index=False)
    log("Autocorrelação no holdout:")
    log(tab.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    log("")

    tau_lab = tab.loc[tab.sinal == "etiqueta do evento (0/1)", "fator_inflacao"].iloc[0]
    n_ev = int(hold.y.sum())
    log(f"Janelas-evento no holdout: {n_ev}")
    log(f"Fator de inflação da etiqueta = {tau_lab:.1f}  ->  "
        f"≈ {n_ev/tau_lab:.0f} EVENTOS INDEPENDENTES")
    log("   (não depende do modelo — é o teto estatístico dos próprios dados)")
    log("")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for etq, acf in res.items():
        ax.plot(lags_min, acf, lw=1.9, color=cores[etq],
                ls="--" if etq.startswith("PAS absoluta") else "-", label=etq)
    ax.axhline(1 / np.e, color="k", ls=":", lw=1, label="1/e")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("desfasamento τ (minutos)")
    ax.set_ylabel("autocorrelação ρ(τ)")
    ax.set_xlim(0, lags_min[-1])
    ax.set_title("B1 (PPG-puro) — Autocorrelação no espaço de variação (holdout)\n"
                 "Ao contrário da PAS absoluta (cinzento), o Δ perde memória depressa",
                 fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/B1_autocorrelacao.png", bbox_inches="tight")
    plt.close(fig)
    log("Guardado: B1_autocorrelacao.png")

    # ── B3 ───────────────────────────────────────────────────────────────────
    dev_cols = pd.read_csv(f"{RAIZ}/dados/holdout_ppg_puro.csv", nrows=0).columns.tolist()
    cand = [c for c in dev_cols if c.endswith('_global_z') or c in ('hr_mean_z',)]
    d_full = pd.read_csv(f"{RAIZ}/dados/holdout_ppg_puro.csv",
                         usecols=['subject_id', 'session_id', 'bloco_id', 'sbp_target'] + cand)
    d_full = d_full.sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
    d_full['delta'] = d_full['sbp_target'] - d_full.groupby('session_id')['sbp_target'].shift(W)

    linhas = []
    for c in cand:
        d = d_full[[c, 'delta', 'subject_id', 'session_id']].dropna()
        if len(d) < 500:
            continue
        r_glob = d[c].corr(d.delta)
        m = d.groupby('subject_id').mean(numeric_only=True)
        r_entre = m[c].corr(m.delta) if len(m) > 3 else np.nan
        dc = d.copy()
        dc[c] = dc[c] - dc.groupby('session_id')[c].transform('mean')
        dc['delta'] = dc.delta - dc.groupby('session_id').delta.transform('mean')
        linhas.append(dict(feature=c, r_global=r_glob, r_entre_sujeitos=r_entre,
                           r_intra_sessao=dc[c].corr(dc.delta)))
    tab_f = pd.DataFrame(linhas).sort_values('r_intra_sessao', key=np.abs, ascending=False)
    tab_f.to_csv(f"{OUT}/B3_correlacao_features_delta.csv", index=False)
    log("\nCorrelação das features com o Δ (ordenado pela correlação DENTRO da sessão):")
    log(tab_f.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    top = tab_f.head(12)
    fig, ax = plt.subplots(figsize=(10, 5))
    yy = np.arange(len(top))
    ax.barh(yy + .22, top.r_entre_sujeitos.abs(), height=.42, color=AZUL, label="entre sujeitos")
    ax.barh(yy - .22, top.r_intra_sessao.abs(), height=.42, color=VERM,
            label="dentro da sessão (o que interessa)")
    ax.set_yticks(yy)
    ax.set_yticklabels(top.feature, fontsize=8)
    ax.set_xlabel("|correlação| com o Δ da PAS")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("B3 (PPG-puro) — no espaço de variação a correlação está\n"
                 "DENTRO da sessão, mesmo sem ECG", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/B3_correlacao_features_delta.png", bbox_inches="tight")
    plt.close(fig)
    log("Guardado: B3_correlacao_features_delta.png")

    # ── B4 ───────────────────────────────────────────────────────────────────
    cc = correlacao_cruzada(hold, "score", "y")
    tx = np.arange(-K_X, K_X + 1) * BLOCO_SEG / 60.0
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(tx, cc, lw=2, color=VERDE)
    ax.axvline(0, color="k", lw=.8)
    ax.axhline(0, color="k", lw=.8)
    i_pico = int(np.nanargmax(cc))
    ax.plot(tx[i_pico], cc[i_pico], "o", color=VERM, ms=9,
            label=f"pico ρ={cc[i_pico]:.3f} em τ={tx[i_pico]:+.2f} min")
    ax.set_xlabel("desfasamento τ (min)   —   τ>0: o score antecede o evento")
    ax.set_ylabel("correlação cruzada score × etiqueta")
    ax.set_title("B4 (PPG-puro) — O score antecipa o evento? (holdout)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/B4_correlacao_cruzada.png", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(dict(lag_min=tx, score_x_evento=cc)).to_csv(
        f"{OUT}/B4_correlacao_cruzada.csv", index=False)
    log(f"\nCorrelação cruzada score×evento: pico ρ={cc[i_pico]:.3f} em τ={tx[i_pico]:+.2f} min")
    log("   (τ=0 significa deteção do evento em curso, não previsão antecipada)")

    (open(f"{OUT}/RESUMO.txt", "w", encoding="utf-8")).write("\n".join(log_linhas))
    log(f"\n[tempo total: {time.time()-t0:.0f} s]")


if __name__ == "__main__":
    main()
