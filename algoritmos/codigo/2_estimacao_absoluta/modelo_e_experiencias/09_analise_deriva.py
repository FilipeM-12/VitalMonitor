#!/usr/bin/env python3
"""Análise da deriva do absoluto (Cap. 6, "O Erro é Deriva e não Ruído" e "Onde Está a Informação Utilizável"), sobre as
previsões causais do holdout da variante de referência (nada se treina). Receitas:
  autocorrelação   resíduo = preds_final - sbp_target; correlação de Pearson do resíduo com o resíduo `lag` blocos depois,
                   dentro de cada sessão e agregada. 60 min = 240 blocos de 15 s.
  N efetivo        N / (1 + 2 * soma das autocorrelações de 1 a 240 blocos).
  PAT              correlação PAT-PAS entre sujeitos (médias por sujeito) e dentro da sessão (PAT e PAS menos a média da sessão).
  variação         variação real = sbp_target - sbp_calib; prevista = preds_delta. Desvios padrão, razão, r de Pearson e r².
  direção          janelas com |variação real| > 5 mmHg; acerto = sinal de preds_delta igual ao da variação real.
  variância        quatro definições possíveis de 'variância explicada'."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nucleo import config as C

LAG_60MIN = 240


def acf_residuo(d, ate):
    """Autocorrelação do resíduo com o desfasamento em TEMPO REAL: para cada desfasamento de `lag` blocos de 15 s, formam-se os pares
    (resíduo no bloco b, resíduo no bloco b + lag) dentro de cada sessão, só quando os dois blocos existem, e calcula-se uma só correlação de
    Pearson com os pares de todas as sessões. Antes o desfasamento era em linhas da tabela, que cobrem mais do que lag x 15 s quando faltam blocos."""
    por_sessao = [(g.bloco_id.values, g.res.values) for _, g in d.sort_values('bloco_id').groupby('session_id')]
    rho = []
    for lag in range(1, ate + 1):
        xs, ys = [], []
        for b, r in por_sessao:
            alvo = b + lag
            pos = np.searchsorted(b, alvo)
            ok = pos < len(b)
            ok[ok] = b[pos[ok]] == alvo[ok]
            if ok.any():
                xs.append(r[ok]); ys.append(r[pos[ok]])
        rho.append(float(np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1]) if xs else np.nan)
    return np.array(rho)


d = pd.read_csv(f"{C.DIR_RESULTADOS}/experiencias/XGB_baseline/holdout_predicoes.csv")
d = d.sort_values(['subject_id', 'record_id', 'segment_order', 'bloco_id']).reset_index(drop=True)
d['res'] = d.preds_final - d.sbp_target
d['dreal'] = d.sbp_target - d.sbp_calib
out = dict(n_janelas=int(len(d)), n_sujeitos=int(d.subject_id.nunique()), n_sessoes=int(d.session_id.nunique()))
rho = acf_residuo(d, LAG_60MIN)
out['acf_60min'] = float(rho[-1])
out['n_efetivo'] = float(len(d) / (1 + 2 * rho.sum()))

# figura: autocorrelação do resíduo em função do desfasamento (texto em inglês, como as outras figuras)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
minutos = np.arange(1, LAG_60MIN + 1) * 15 / 60.0
fig, ax = plt.subplots(figsize=(5.8, 3.1))
ax.axhline(0, color="#888888", linewidth=0.8)
ax.plot(minutos, rho, color="#2b6cb0", linewidth=1.8)
ax.plot([minutos[-1]], [rho[-1]], "o", color="#c05621", markersize=6, zorder=3)
ax.annotate(f"{rho[-1]:.2f} at 60 min", xy=(minutos[-1], rho[-1]), xytext=(-8, 12),
            textcoords="offset points", ha="right", fontsize=8, color="#c05621")          # sem linha de ligação: o texto e o ponto têm a mesma cor
ax.set_xlabel("Lag (minutes)")
ax.set_ylabel("Autocorrelation of the residual")
ax.set_xlim(0, 62)
ax.set_ylim(-0.05, 1.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
os.makedirs(f"{C.DIR_RESULTADOS}/figuras", exist_ok=True)
fig.savefig(f"{C.DIR_RESULTADOS}/figuras/acf_residuo.png", dpi=200)
s = d.groupby('subject_id').agg(pat=('pat_mean', 'mean'), sbp=('sbp_target', 'mean'))
out['pat_entre_sujeitos'] = float(np.corrcoef(s.pat, s.sbp)[0, 1])
pc = d.pat_mean - d.groupby('session_id').pat_mean.transform('mean')
sc = d.sbp_target - d.groupby('session_id').sbp_target.transform('mean')
out['pat_dentro_da_sessao'] = float(np.corrcoef(pc, sc)[0, 1])
out['sd_variacao_real'] = float(d.dreal.std())
out['sd_variacao_prevista'] = float(d.preds_delta.std())
out['razao_desvios'] = out['sd_variacao_prevista'] / out['sd_variacao_real']
out['r_variacao'] = float(np.corrcoef(d.preds_delta, d.dreal)[0, 1])
out['r2_variacao'] = out['r_variacao'] ** 2
g = d[d.dreal.abs() > 5]
out['direcao_acerto'] = float((np.sign(g.preds_delta) == np.sign(g.dreal)).mean())
out['direcao_n_janelas'] = int(len(g))
grp = d.groupby('session_id')
pc2, tc2 = d.preds_final - grp.preds_final.transform('mean'), d.sbp_target - grp.sbp_target.transform('mean')
out['variancia_explicada_testes'] = {
    'r2_nivel_global': float(np.corrcoef(d.preds_final, d.sbp_target)[0, 1] ** 2),
    'r2_nivel_global_so_calibracao': float(np.corrcoef(d.sbp_calib, d.sbp_target)[0, 1] ** 2),       # a âncora, sem modelo nenhum
    'r2_dentro_da_sessao': float(np.corrcoef(pc2, tc2)[0, 1] ** 2),
    '1_menos_SSE_SST_dentro_da_sessao': float(1 - ((pc2 - tc2) ** 2).sum() / (tc2 ** 2).sum()),
    '1_menos_MSE_modelo_MSE_ancora': float(1 - (d.res ** 2).mean() / ((d.sbp_calib - d.sbp_target) ** 2).mean())}
os.makedirs(f"{C.DIR_RESULTADOS}/tabelas", exist_ok=True)
json.dump(out, open(f"{C.DIR_RESULTADOS}/tabelas/deriva_absoluto.json", "w"), indent=1)
print(json.dumps(out, indent=1))
