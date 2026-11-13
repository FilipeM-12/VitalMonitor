"""persistencia_dev.py — persistência dos episódios por janela temporal W (2, 3, 5 e 10 min), SÓ NO DESENVOLVIMENTO e com a janela por tempo exato.

Regras de persistência sobre a pressão real do DESENVOLVIMENTO
e com Δ_W = PA(t) - PA(t-W) por tempo exato (grade de 15 s: o bloco a t-W tem de existir):
  evento       Δ_W <= -15 (descida) ou >= +15 (subida); episódio = corrida de blocos de evento, fundida a 1 min
  referência   mediana dos 6 aos 4 min ANTES do início do episódio
  variação líquida   mediana dos 6 aos 10 min depois menos a referência
  mantém       mediana dos 9 aos 11 min depois a pelo menos 10 mmHg da referência, no sentido do evento
Para W = 2 e 3 compara-se «só a W min» com «só a 5 min» (episódios sem sobreposição com a outra definição), com IC 95 % da diferença por bootstrap de DOENTES.
Uso: python3 persistencia_dev.py [pasta_dados]
"""
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
TF = os.path.normpath(os.path.join(AQUI, "..", "tendencia_tempo_fixo"))        # tempo_fixo.py e os best_params_*.json só existem em tendencia_tempo_fixo/
sys.path.insert(0, TF)
import tempo_fixo as tf  # noqa: E402

DADOS = sys.argv[1] if len(sys.argv) > 1 else f"{AQUI}/dados"
BLOCO_SEG, D, GAP = 15, 15, 4
PRE_A, PRE_B, POS_A, POS_B, RET_A, RET_B, LIMIAR = 24, 16, 24, 40, 36, 44, 10.0
JANELAS = (2, 3, 5, 10)
B = 1000

dev, _ = tf.preparar_grade(DADOS, 'dev')
dev['subject_id'] = dev.subject_id.astype(str)
horas_total = dev.sbp_target.notna().sum() * BLOCO_SEG / 3600


def runs(mask, gap=GAP):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    cortes = np.where(np.diff(idx) > gap + 1)[0]
    return [(g[0], g[-1]) for g in np.split(idx, cortes + 1)]


def sobrepoe(a0, a1, b0, b1):
    return a1 >= b0 and a0 <= b1


res, linhas = [], []
rng = np.random.default_rng(0)
for direcao in ('descida', 'subida'):
    sinal = -1.0 if direcao == 'descida' else 1.0
    por_suj = {}                        # subject -> {W: [(liq, mantem)], 'so5_W': [...], 'soW': [...]}
    for sid, g in dev.groupby('session_id', sort=False):
        sbp = g['sbp_target'].values; n = len(g); suj = g.subject_id.iloc[0]
        ev = {w: runs((sinal * (g['sbp_target'] - g['sbp_target'].shift(w * 4)) >= D).fillna(False).values) for w in JANELAS}

        def medir(s0):
            if s0 - PRE_A < 0 or s0 + POS_B >= n:
                return None
            pre = np.nanmedian(sbp[s0 - PRE_A:s0 - PRE_B + 1]); pos = np.nanmedian(sbp[s0 + POS_A:s0 + POS_B + 1])
            ret = np.nanmedian(sbp[s0 + RET_A:s0 + RET_B + 1]) if s0 + RET_B < n else np.nan
            if not (np.isfinite(pre) and np.isfinite(pos)):
                return None
            return pos - pre, bool(np.isfinite(ret) and sinal * (ret - pre) >= LIMIAR)
        d = por_suj.setdefault(suj, {})
        for w in JANELAS:
            for s0, s1 in ev[w]:
                r = medir(s0)
                if r is None:
                    continue
                d.setdefault(w, []).append(r)
                if w in (2, 3) and not any(sobrepoe(s0, s1, a, b) for a, b in ev[5]):
                    d.setdefault(f'so{w}', []).append(r)
        for w in (2, 3):
            for s0, s1 in ev[5]:
                if not any(sobrepoe(s0, s1, a, b) for a, b in ev[w]):
                    r = medir(s0)
                    if r is not None:
                        d.setdefault(f'so5_vs{w}', []).append(r)
    docs = sorted(por_suj)
    todos = lambda chave, idx=None: [r for s in (docs if idx is None else idx) for r in por_suj[s].get(chave, [])]
    for w in JANELAS:
        L = todos(w)
        linhas.append(dict(direcao=direcao, janela_min=w, episodios=len(L), por_hora=len(L) / horas_total, variacao_liquida_mediana=float(np.median([r[0] for r in L])),
                           mantem_pct=100 * float(np.mean([r[1] for r in L]))))
    for w in (2, 3):
        def dif(idx):
            a, b = todos(f'so5_vs{w}', idx), todos(f'so{w}', idx)
            return (np.mean([r[1] for r in a]) - np.mean([r[1] for r in b])) if a and b else np.nan
        pt = dif(None)
        bt = [dif([docs[i] for i in rng.integers(0, len(docs), len(docs))]) for _ in range(B)]
        a, b = todos(f'so5_vs{w}'), todos(f'so{w}')
        res.append(dict(direcao=direcao, comparacao=f'só 5 min contra só {w} min', n_so5=len(a), n_soW=len(b), mantem_so5_pct=100 * np.mean([r[1] for r in a]),
                        mantem_soW_pct=100 * np.mean([r[1] for r in b]), dif_pontos=100 * pt, ic_baixo=100 * float(np.nanpercentile(bt, 2.5)),
                        ic_alto=100 * float(np.nanpercentile(bt, 97.5)),
                        liq_so5=float(np.median([r[0] for r in a])), liq_soW=float(np.median([r[0] for r in b]))))
pd.DataFrame(linhas).to_csv(f"{AQUI}/persistencia_dev_por_janela.csv", index=False)
pd.DataFrame(res).to_csv(f"{AQUI}/persistencia_dev_comparacao.csv", index=False)
pd.set_option('display.width', 200)
print(pd.DataFrame(linhas).round(2).to_string(index=False)); print(pd.DataFrame(res).round(2).to_string(index=False))
