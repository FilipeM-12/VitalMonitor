#!/usr/bin/env python3
"""verificar_inferencia_tempo.py — as previsões do modelo de tendência INSTALADO no servidor coincidem com as do treino?

Entrada: as 43 características da tabela de teste do treino (tendencia_tempo_fixo, linhas com alvo) e as previsões do modelo do treino (resultados_deteccao/previsoes.npz, teste_<direção> = sinal x previsão).
Servidor: server/modelo_tendencia_ppg_quant.joblib carregado como no worker (Booster + sinal), no ambiente do servidor (o xgboost e o numpy de lá), com NaN onde faltam valores, e o arredondamento do worker (3 casas).
Compara-se a previsão arredondada do servidor com a do treino arredondada igual (e a diferença máxima em mmHg, antes do arredondamento). Escreve resultado_inferencia_tempo.json.
  venv/bin/python verificar_inferencia_tempo.py"""
import json, os, subprocess, sys, tempfile
import numpy as np
AQUI = os.path.dirname(os.path.abspath(__file__)); A = "/home/filipe/Desktop/Tese/algoritmo-ml"; TF = f"{A}/Tentativa_22_10/tendencia_tempo_fixo"
SERVIDOR_PY = "/home/filipe/Desktop/Tese/server/venv/bin/python3"; JOBLIB = "/home/filipe/Desktop/Tese/server/modelo_tendencia_ppg_quant.joblib"
if len(sys.argv) > 1 and sys.argv[1] == "--servidor":
    import joblib, xgboost as xgb
    d = np.load(sys.argv[2]); b = joblib.load(JOBLIB); out = {}
    for dire in ("descida", "subida"):
        m = xgb.Booster(); m.load_model(bytearray(b[dire]["booster"]))
        out[dire] = np.round(b[dire]["sinal"] * m.predict(xgb.DMatrix(d[f"X_{dire}"])), 3)
    np.savez(sys.argv[3], versao=np.array(xgb.__version__), **out); sys.exit(0)
sys.path.insert(0, TF)
import tempo_fixo as tf
import joblib
b = joblib.load(JOBLIB)
hold, V = tf.preparar_grade(f"{TF}/dados", "holdout"); ok = hold.delta.notna().values
z = np.load(f"{A}/Tentativa_22_10/resultados_deteccao/previsoes.npz")
tmp = tempfile.mkdtemp(); ent, sai = f"{tmp}/ent.npz", f"{tmp}/sai.npz"
np.savez(ent, **{f"X_{d}": hold.loc[ok, b[d]["feats"]].values.astype("float64") for d in ("descida", "subida")})
subprocess.run([SERVIDOR_PY, os.path.abspath(__file__), "--servidor", ent, sai], check=True)
s = np.load(sai); R = {"xgboost_servidor": str(s["versao"]), "blocos": int(ok.sum())}
import xgboost; R["xgboost_treino"] = xgboost.__version__
for d in ("descida", "subida"):
    ref = np.round(z[f"teste_{d}"], 3); srv = s[d]
    R[d] = dict(iguais_arredondado=float(np.mean(ref == srv)), dif_max_mmHg=float(np.max(np.abs(z[f"teste_{d}"] - srv))), n_diferentes=int(np.sum(ref != srv)))
    print(d, R[d], flush=True)
json.dump(R, open(f"{AQUI}/resultado_inferencia_tempo.json", "w"), indent=1)
