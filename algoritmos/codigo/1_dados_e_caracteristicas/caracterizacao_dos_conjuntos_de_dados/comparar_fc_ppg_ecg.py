#!/usr/bin/env python3
"""A FC que a extração só-PPG dá (picos do PPG, proeminência mínima de 15 %
da amplitude da janela) contra a FC que a extração com ECG dá (intervalos RR), nas mesmas janelas do MIMIC (mesma
subject_id, segment_name, window_idx). Os dois ficheiros são os das extrações que alimentam as duas formulações
(features_*_baseline.csv e features_*_ppg_apg.csv). Escreve comparar_fc_ppg_ecg.json."""
import json
import os

import numpy as np
import pandas as pd

F = "/media/filipe/Dados/Features"
IDS = {l.strip() for l in open("/home/filipe/Desktop/Tese/algoritmo-ml/RESULTADOS_RELATORIO/codigo_dependencias/particao_sujeitos/pacientes_holdout.txt") if l.strip()}
K = ["subject_id", "segment_name", "window_idx"]
partes = []
for g in ("homens", "mulheres"):
    e = pd.read_csv(f"{F}/features_{g}_baseline.csv", usecols=K + ["hr_mean"]).rename(columns={"hr_mean": "hr_ecg"})
    p = pd.read_csv(f"{F}/features_{g}_ppg_apg.csv", usecols=K + ["hr_mean"]).rename(columns={"hr_mean": "hr_ppg"})
    partes.append(e.merge(p, on=K))
d = pd.concat(partes).dropna()
d["erro"] = d.hr_ppg - d.hr_ecg
d["teste"] = d.subject_id.astype(str).isin(IDS)
sai = {}
for nome, x in (("todas", d), ("teste", d[d.teste]), ("desenvolvimento", d[~d.teste])):
    e = x.erro.abs()
    sai[nome] = dict(janelas=int(len(x)), sujeitos=int(x.subject_id.nunique()), vies_bpm=round(float(x.erro.mean()), 2),
                     erro_abs_mediano_bpm=round(float(e.median()), 2), erro_abs_medio_bpm=round(float(e.mean()), 2),
                     p95_erro_abs_bpm=round(float(e.quantile(0.95)), 1), dentro_5bpm_pct=round(100 * float((e <= 5).mean()), 1),
                     dentro_10bpm_pct=round(100 * float((e <= 10).mean()), 1),
                     r=round(float(np.corrcoef(x.hr_ppg, x.hr_ecg)[0, 1]), 4))
json.dump(sai, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparar_fc_ppg_ecg.json"), "w"), indent=1)
print(json.dumps(sai, indent=1))
