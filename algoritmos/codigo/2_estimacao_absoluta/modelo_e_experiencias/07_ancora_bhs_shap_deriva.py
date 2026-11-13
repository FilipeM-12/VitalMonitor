#!/usr/bin/env python3
"""Âncora de calibração, graus BHS, Shapley e figura da deriva (Capítulo 6, "Síntese: um patamar de erro reprodutível").

Lê as previsões do holdout de 01_correr_experiencia.py. A âncora é "repetir o último valor da braçadeira" (sbp_calib):
  tab:ancora_horizonte  erro do modelo e da âncora por faixa de tempo desde a calibração (variante de referência)
  tab:ancora_variantes  MAE de cada variante e idade mediana da sua calibração
  BHS                   percentagem de erros <= 5, 10, 15 mmHg e grau, por janela e por sujeito
  fig:deriva            as duas primeiras tabelas em gráfico
  fig:shap              contributo de cada característica no modelo de referência (valores de Shapley)
Os textos das figuras estão em inglês."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nucleo import config as C
from nucleo.avaliacao import bhs

VARIANTES = {'baseline': 'Baseline (own extraction)', 'fft': 'Spectral (FFT)', 'nk2': 'NeuroKit2'}    # a variante de filtragem agressiva (nk) saiu da tese
FAIXAS = [0, 5, 15, 30, 60, 120, 1e5]
ROTULOS = ['<5', '5-15', '15-30', '30-60', '60-120', '>120']
OUT = f"{C.DIR_RESULTADOS}/tabelas"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{C.DIR_RESULTADOS}/figuras", exist_ok=True)


def carregar_previsoes(v):
    d = pd.read_csv(f"{C.DIR_RESULTADOS}/experiencias/XGB_{v}/holdout_predicoes.csv")
    d['err_modelo'] = (d.preds_final - d.sbp_target).abs()
    d['err_ancora'] = (d.sbp_calib - d.sbp_target).abs()
    d['idade_calib_min'] = d.tempo_desde_calib_seg / 60.0
    return d


dados = {v: carregar_previsoes(v) for v in VARIANTES}
base = dados['baseline']

# ── tab:ancora_horizonte ─────────────────────────────────────────────────────────────────────────────────────────
g = (base.groupby(pd.cut(base.idade_calib_min, FAIXAS, labels=ROTULOS), observed=True)
     .agg(n=('err_modelo', 'size'), ancora=('err_ancora', 'mean'), modelo=('err_modelo', 'mean')))
g['ganho'] = g['ancora'] - g['modelo']
g = g.reset_index().rename(columns={'idade_calib_min': 'faixa_min'})
g.to_csv(f"{OUT}/ancora_horizonte.csv", index=False)
print(g.to_string(index=False))

# ── tab:ancora_variantes ─────────────────────────────────────────────────────────────────────────────────────────
t = pd.DataFrame([dict(variante=v, nome=VARIANTES[v], sujeitos=int(d.subject_id.nunique()), janelas=len(d),
                       idade_calib_mediana_min=round(float(d.idade_calib_min.median()), 1),
                       mae_ancora=round(float(d.err_ancora.mean()), 2), mae_modelo=round(float(d.err_modelo.mean()), 2),
                       ganho_do_modelo=round(float(d.err_ancora.mean() - d.err_modelo.mean()), 2))
                  for v, d in dados.items()]).sort_values('mae_modelo')
xs, ys = t.idade_calib_mediana_min.values, t.mae_modelo.values
r = float(np.corrcoef(xs, ys)[0, 1])
t.to_csv(f"{OUT}/ancora_variantes.csv", index=False)
print(t.to_string(index=False), f"\nr(MAE, idade da calibração) = {r:.4f}")

# ── efeito da janela de calibração ──────────────────────────────────────────────────────────────────────────────
# Os primeiros N_BLOCOS_CALIBRACAO blocos de cada sessão são normalizados com estatísticas que incluem blocos seguintes da
# própria janela. Um sistema real só daria estimativas depois dela, por isso mede-se o MAE também só a partir dela.
inicio = C.N_BLOCOS_CALIBRACAO * 15
janela = pd.DataFrame([dict(variante=v, mae_todos=round(float(d.err_modelo.mean()), 4), n_todos=len(d),
                            mae_apos_calibracao=round(float(d[d.tempo_desde_calib_seg >= inicio].err_modelo.mean()), 4),
                            n_apos_calibracao=int((d.tempo_desde_calib_seg >= inicio).sum()),
                            ancora_apos_calibracao=round(float(d[d.tempo_desde_calib_seg >= inicio].err_ancora.mean()), 4),
                            fracao_na_janela=round(float((d.tempo_desde_calib_seg < inicio).mean()), 4))
                       for v, d in dados.items()])
janela.to_csv(f"{OUT}/efeito_janela_calibracao.csv", index=False)
print(janela.to_string(index=False))

# ── BHS ──────────────────────────────────────────────────────────────────────────────────────────────────────────
b = bhs(base)
json.dump(dict(bhs=b, r_mae_idade_calibracao=r, n_horizontes_em_que_o_modelo_vence=int((g.ganho > 0).sum()), n_horizontes=int(len(g))),
          open(f"{OUT}/ancora_bhs.json", "w"), indent=1)
print(json.dumps(b, indent=1))

# ── fig:deriva ───────────────────────────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.9))
x = np.arange(len(g))
ax1.plot(x, g['ancora'], 'o-', lw=2.2, ms=8, color='#555555', label='Anchor: repeat the last cuff value')
ax1.plot(x, g['modelo'], 's--', lw=2.2, ms=7, color='#c0392b', label='XGBoost model')
for xi, (a, m) in enumerate(zip(g['ancora'], g['modelo'])):
    ax1.annotate(f'{a - m:+.2f}', (xi, max(a, m) + 0.55), ha='center', fontsize=8, color='#c0392b')      # ganho: âncora - modelo
ax1.set_xticks(x)
ax1.set_xticklabels(g['faixa_min'] if 'faixa_min' in g else ROTULOS)
ga = json.load(open(f"{OUT}/ganho_contra_ancora.json"))["todos"]          # 12_ganho_contra_ancora.py (bootstrap por sujeito); correr antes deste
ax1.set_title(f"(a) Model against the anchor: gain {ga['ganho']:+.2f} mmHg\n(95% CI {ga['ic95'][0]:+.2f} to {ga['ic95'][1]:+.2f}), labels: anchor - model", fontsize=10.5)
ax1.set_xlabel('Time since calibration (min)')
ax1.set_ylabel('Mean absolute error of systolic BP (mmHg)')
ax1.legend(fontsize=8.5, loc='upper left')
ax1.grid(alpha=0.3, ls=':')
ax1.set_ylim(0, 20)
cores = {'baseline': '#c0392b', 'fft': '#e67e22', 'nk2': '#2980b9'}
desloca = {'baseline': (9, 6), 'fft': (9, -16), 'nk2': (9, -4)}
for v, nome in VARIANTES.items():
    d = dados[v]
    ax2.scatter(d.idade_calib_min.median(), d.err_modelo.mean(), s=170, color=cores[v], zorder=3, edgecolor='white', lw=1.5)
    ax2.annotate(f"{nome}\n{d.subject_id.nunique()} subj., {len(d) // 1000}k windows",
                 (d.idade_calib_min.median(), d.err_modelo.mean()), textcoords='offset points', xytext=desloca[v], fontsize=8)
k = np.polyfit(xs, ys, 1)
xx = np.linspace(xs.min() * 0.8, xs.max() * 1.35, 50)
ax2.plot(xx, np.polyval(k, xx), '--', color='#888888', lw=1.4, zorder=1, label='linear fit')
ax2.set_xlabel('Median calibration age (min)')
ax2.set_ylabel('Reported MAE (mmHg)')
ax2.set_title('(b) The reported MAE follows the calibration age,\nnot the method', fontsize=11)
ax2.legend(fontsize=8.5, loc='lower right')
ax2.grid(alpha=0.3, ls=':')
ax2.set_xlim(0, max(xs) * 1.5)
fig.tight_layout()
fig.savefig(f"{C.DIR_RESULTADOS}/figuras/figura_deriva_calibracao.png", dpi=200)
fig.savefig(f"{C.DIR_RESULTADOS}/figuras/figura_deriva_calibracao.pdf")
plt.close(fig)

# ── fig:shap ─────────────────────────────────────────────────────────────────────────────────────────────────────
import shap                                                                    # noqa: E402
from sklearn.preprocessing import StandardScaler                               # noqa: E402
from nucleo.busca import melhores_parametros                                    # noqa: E402
from nucleo.dados import carregar                                               # noqa: E402
from nucleo.modelos import AgeStratifiedXGBEnsemble, fixar_sementes             # noqa: E402

N_AMOSTRA = 3000
dev, teste, _ = carregar("baseline")
F = list(C.FEATURES_V3)
fixar_sementes()
sc = StandardScaler()
Xd, Xt = sc.fit_transform(dev[F]), sc.transform(teste[F])
modelo = AgeStratifiedXGBEnsemble(dict(melhores_parametros("XGB", "baseline")))
modelo.fit(Xd, dev[C.ALVO].values, dev['age'].values)
idx = np.sort(np.random.default_rng(C.SEMENTE).choice(len(teste), size=min(N_AMOSTRA, len(teste)), replace=False))
Xa, idades = Xt[idx], teste['age'].values[idx]
valores = np.zeros_like(Xa)
for mascara, booster in ((idades < 60, modelo.modelo_jovem), (idades >= 60, modelo.modelo_senior)):
    if mascara.any() and booster is not None:
        valores[mascara] = shap.TreeExplainer(booster).shap_values(Xa[mascara])
plt.figure(figsize=(7.4, 5.4))
shap.summary_plot(valores, Xa, feature_names=F, show=False, max_display=15, plot_size=None)
plt.title("Shapley contribution per feature, baseline model (causal)", fontsize=10)
plt.gcf().axes[0].tick_params(labelsize=8)
plt.xlabel("SHAP value (impact on model output)", fontsize=9)
plt.tight_layout()
plt.savefig(f"{C.DIR_RESULTADOS}/figuras/shap_absoluto.png", dpi=200, bbox_inches="tight")
plt.close()
imp = pd.DataFrame({"feature": F, "shap_abs_medio": np.abs(valores).mean(axis=0)}).sort_values("shap_abs_medio", ascending=False)
imp['posicao'] = range(1, len(imp) + 1)
imp.to_csv(f"{OUT}/shap_importancia.csv", index=False)
print(imp.head(12).to_string(index=False))
