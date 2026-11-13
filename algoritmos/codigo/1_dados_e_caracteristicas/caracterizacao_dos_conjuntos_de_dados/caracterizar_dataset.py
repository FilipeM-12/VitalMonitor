#!/usr/bin/env python3
"""Caracterização dos conjuntos de dados do MIMIC usados nas duas formulações (para a
secção inicial do Cap. 6). Lê exatamente os dados com que os modelos foram treinados e avaliados:
  · absoluto: a tabela por bloco do núcleo causal (tudo_3_dobras/absoluto_causal/cache/baseline__causal__min20.pkl),
    partição pelo ficheiro de sujeitos de teste do núcleo (C.IDS_HOLDOUT);
  · tendência: os datasets causais só-PPG com que o modelo adotado foi treinado e avaliado
    ({dev,holdout}_ppg.csv, a tabela única).
Cada linha é um bloco de 15 s. Por formulação e partição: sujeitos (MIMIC-III / MIMIC-IV), registos, sessões, blocos,
horas de sinal, idade e sexo (por sujeito), PA sistólica de referência (por bloco) e, na tendência, a fração de blocos
com evento (variação de pelo menos 15 mmHg em 5 min, posicional por sessão, como no treino).
Escreve caracterizacao_dataset.json e .md."""
import json, os, sys
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
A = "/home/filipe/Desktop/Tese/algoritmo-ml"
NUCLEO = f"{A}/RESULTADOS_RELATORIO/tudo_3_dobras/absoluto_causal"
sys.path.insert(0, NUCLEO)
from nucleo import config as C  # noqa: E402

BLOCO_S = 15


def resumo(df, eventos=False):
    suj = df.groupby("subject_id").agg(dataset=("dataset", "first"), age=("age", "first"), gender=("gender", "first"))
    r = dict(sujeitos=int(len(suj)),
             sujeitos_mimic3=int((suj.dataset == "MIMIC-III").sum()), sujeitos_mimic4=int((suj.dataset == "MIMIC-IV").sum()),
             registos=int(df.groupby(["subject_id", "record_id"]).ngroups), sessoes=int(df.session_id.nunique()),
             blocos=int(len(df)), horas=round(len(df) * BLOCO_S / 3600, 1),
             idade_media=round(float(suj.age.mean()), 1), idade_dp=round(float(suj.age.std()), 1),
             idade_min=int(suj.age.min()), idade_max=int(suj.age.max()),
             pct_homens=round(100 * float((suj.gender == "M").mean()), 1),
             pas_media=round(float(df.sbp_target.mean()), 1), pas_dp=round(float(df.sbp_target.std()), 1),
             pas_p5=round(float(df.sbp_target.quantile(0.05)), 1), pas_p95=round(float(df.sbp_target.quantile(0.95)), 1))
    if eventos:
        d = df.sort_values(["session_id", "bloco_id"])
        delta = d.sbp_target - d.groupby("session_id").sbp_target.shift(20)
        r["pct_blocos_descida"] = round(100 * float((delta <= -15).sum() / delta.notna().sum()), 2)
        r["pct_blocos_subida"] = round(100 * float((delta >= 15).sum() / delta.notna().sum()), 2)
    return r


out = {}
ids = {l.strip() for l in open(C.IDS_HOLDOUT) if l.strip()}
CACHE = f"{NUCLEO}/cache/baseline__causal__min20.pkl"
ANTES = json.load(open(f"{AQUI}/caracterizacao_dataset_antes_tabela_unica.json"))     # a cache do absoluto foi para o disco
if os.path.exists(CACHE):
    ab = pd.read_pickle(CACHE)
    out["absoluto"] = {"desenvolvimento": resumo(ab[~ab.subject_id.isin(ids)]), "teste": resumo(ab[ab.subject_id.isin(ids)]),
                       "total": resumo(ab)}
else:
    ab = None                                    # a parte do absoluto não mudou (mesma cache, mesma lista de teste): copia-se
    out["absoluto"] = ANTES["absoluto"]
cols = ["subject_id", "record_id", "session_id", "bloco_id", "dataset", "age", "gender", "sbp_target"]
D = f"{A}/2026-09-09_normalizacao_causal_tendencia/dados"
dev = pd.read_csv(f"{D}/dev_ppg.csv", usecols=cols); hold = pd.read_csv(f"{D}/holdout_ppg.csv", usecols=cols)
out["tendencia"] = {"desenvolvimento": resumo(dev, True), "teste": resumo(hold, True),
                    "total": resumo(pd.concat([dev, hold]), True)}
# sujeitos comuns às duas formulações
out["sujeitos_em_ambas"] = (int(len(set(ab.subject_id) & set(pd.concat([dev, hold]).subject_id))) if ab is not None
                            else ANTES["sujeitos_em_ambas"])   # os 350 doentes da tendência são os mesmos de antes (conferido)
json.dump(out, open(f"{AQUI}/caracterizacao_dataset.json", "w"), indent=1, ensure_ascii=False)
with open(f"{AQUI}/caracterizacao_dataset.md", "w", encoding="utf-8") as f:
    for form in ("absoluto", "tendencia"):
        t = pd.DataFrame(out[form])
        f.write(f"## {form}\n\n| | " + " | ".join(t.columns) + " |\n|---|" + "---|" * len(t.columns) + "\n")
        for k, row in t.iterrows():
            f.write(f"| {k} | " + " | ".join(str(v) for v in row) + " |\n")
        f.write("\n")
    f.write(f"Sujeitos presentes nas duas formulações: {out['sujeitos_em_ambas']}\n")
print(open(f"{AQUI}/caracterizacao_dataset.md").read())
