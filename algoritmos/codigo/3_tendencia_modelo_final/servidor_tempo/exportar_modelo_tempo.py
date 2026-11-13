"""exportar_modelo_tempo.py — exporta o modelo final da tendência (XGBoost quantílico, 43 características, janela de 5 min por tempo exato) para o worker.

Treina os dois boosters em TODO o desenvolvimento (as linhas com alvo; as
características em falta ficam NaN, que o XGBoost trata) com os hiperparâmetros de 01_optuna_tempo.py (best_params_quant_<direção>_tudo.json) e guarda, no mesmo formato do
joblib instalado no servidor, os limiares do ponto de operação a 10 min/h da UNIÃO dos dois alarmes, escolhidos no desenvolvimento (previsões fora da amostra, critério maximin, ciclo
de serviço <= 30 %, sem portão e sem limite de duração: ponto_tempo_reafinado_cargas.csv, variante «tudo», alvo 10). Escreve modelo_tendencia_ppg_quant_tempo.joblib NESTA pasta,
não no servidor.   Uso: python3 exportar_modelo_tempo.py [pasta_dados]
"""
import datetime
import json
import os
import sys

import joblib
import pandas as pd
from xgboost import XGBRegressor

AQUI = os.path.dirname(os.path.abspath(__file__))
TF = os.path.normpath(os.path.join(AQUI, "..", "tendencia_tempo_fixo"))        # tempo_fixo.py e os best_params_*.json só existem em tendencia_tempo_fixo/
sys.path.insert(0, TF)
import tempo_fixo as tf  # noqa: E402

DADOS = sys.argv[1] if len(sys.argv) > 1 else f"{TF}/dados"
dev, V = tf.preparar_grade(DADOS, 'dev')
FEATS = V['tudo']
dev = dev[dev.delta.notna()].reset_index(drop=True)
pt = pd.read_csv(f"{TF}/ponto_tempo_reafinado_cargas.csv")
e = pt[(pt.variante == 'tudo') & (pt.alvo == 10)].iloc[0]

bundle = {"meta": {
    "variante": "ppg_quant_tempo, 43 caracteristicas, ancoras CAUSAIS por SESSAO, janela de 5 min por tempo exato",
    "nota": ("PPG puro + regressao quantilica sobre o delta a 5 min (PA(t) - PA(t-5 min), por tempo exato); score = queda prevista em mmHg. As variacoes a 2/5/10 min "
             "e a volatilidade vao buscar o bloco ao instante exato e ficam em falta (NaN, tratado pelo XGBoost) se nao existe. Ponto de operacao a 10 min/h da uniao, "
             "escolhido no desenvolvimento, sem portao e sem limite de duracao."),
    "treinado_em": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "n_blocos_dev": int(len(dev)),
    "teste_sens_descida": float(e.sens_d), "teste_sens_subida": float(e.sens_s), "teste_uniao_FA_min_h": float(e.min_falso_h),
    "teste_uniao_duty": float(e.ciclo_servico), "teste_vpp_uniao": float(e.vpp_uniao),
    "substitui": "modelo_tendencia_ppg_quant.joblib (19 caracteristicas, janela por blocos)",
}}
for direcao in ('descida', 'subida'):
    bp = json.load(open(f"{TF}/best_params_quant_{direcao}_tudo.json"))['best_params']
    q = 0.10 if direcao == 'descida' else 0.90
    m = XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=q, n_jobs=14, tree_method='hist').fit(dev[FEATS].values, dev['delta'].values)
    bundle[direcao] = {"booster": m.get_booster().save_raw(), "feats": FEATS, "sinal": -1.0 if direcao == 'descida' else 1.0,
                       "limiar": float(e.tau_d if direcao == 'descida' else e.tau_s), "limite_min": None,
                       "portao_blocos": 4, "portao_limite": 1e9}                      # sem portão: o limite enorme deixa-o sempre aberto
    print(f"  {direcao}: limiar {bundle[direcao]['limiar']} mmHg, {len(FEATS)} caracteristicas, {len(dev)} blocos", flush=True)
out = f"{AQUI}/modelo_tendencia_ppg_quant_tempo.joblib"
joblib.dump(bundle, out)
print(f"Guardado: {out}")
