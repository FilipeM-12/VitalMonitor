#!/usr/bin/env python3
"""Planalto da afinação na tendência (sobre o orçamento de 20 tentativas): para cada estudo do Optuna, o melhor AUC de
validação cruzada até à tentativa 5, 10, 15 e 20, e quanto melhora entre a 5.ª e a 20.ª. Uso: python planalto_arquiteturas.py [pasta com os .db]
(por omissão a pasta dos estudos novos, optuna_arquiteturas). Escreve planalto_arquiteturas[_<etiqueta>].json."""
import glob, json, os, sqlite3, sys
A = "/home/filipe/Desktop/Tese/algoritmo-ml"
pasta = sys.argv[1] if len(sys.argv) > 1 else f"{A}/2026-09-13_modelo_temporal_causal/optuna_arquiteturas"
etiqueta = sys.argv[2] if len(sys.argv) > 2 else ""
res = {}
for f in sorted(glob.glob(f"{pasta}/*.db")):
    c = sqlite3.connect(f)
    v = [r[0] for r in c.execute("select value from trial_values v join trials t on t.trial_id=v.trial_id where t.state='COMPLETE' order by t.number") if r[0] is not None]
    if len(v) < 20:
        continue
    nome = os.path.basename(f).replace("optuna_", "").replace(".db", "").replace("_arquiteturas", "")
    b = [max(v[:k]) for k in (5, 10, 15, 20)]
    res[nome] = dict(melhor_ate=dict(zip((5, 10, 15, 20), b)), ganho_5_a_20=b[3] - b[0], melhor_na_tentativa=v.index(max(v)) + 1)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planalto_arquiteturas" + (f"_{etiqueta}" if etiqueta else "") + ".json")
json.dump(res, open(out, "w"), indent=1)
print(out, len(res), "estudos")
for k, v in res.items():
    print(f"{k:34s} ganho 5->20 = {v['ganho_5_a_20']:.3f}")
