#!/usr/bin/env python3
"""Extração própria contra NeuroKit/NeuroKit2 SÓ nas janelas que ambas as variantes têm (Cap. 6, "Cadeias de extração").

O MAE da variante NeuroKit é muito menor do que o da extração própria, mas retém poucos sujeitos. Emparelhar as duas na
mesma janela física neutraliza a atrição. Nas janelas emparelhadas o NeuroKit continua melhor, e quem o explica é a idade da
calibração: a âncora (repetir a última calibração) não usa característica nenhuma e mostra quase toda a diferença. A prova
direta é dar às duas cadeias a MESMA calibração e recompor a estimativa (PA = calibração + desvio previsto).

Um par só conta se os alvos das duas variantes diferirem menos de TOL_ALVO mmHg (0,5 é o valor citado na tese; a saída inclui
também a coincidência exata, para se ver que a conclusão não depende da tolerância)."""
import os
import sys

import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nucleo import config as C

CHAVE = ["subject_id", "record_id", "segment_order", "bloco_id"]
COLS = CHAVE + ["sbp_target", "preds_final", "preds_delta", "sbp_calib", "tempo_desde_calib_seg"]
NOMES = {"baseline": "Extração própria", "nk": "NeuroKit (filtragem agressiva)", "nk2": "NeuroKit2 parcial"}
TOLERANCIAS = [0.5, 0.0]


def carrega(v):
    return pd.read_csv(f"{C.DIR_RESULTADOS}/experiencias/XGB_{v}/holdout_predicoes.csv")[COLS]


def metricas(df, sufixo):
    erro = (df["sbp_target_a"] - df["preds_final" + sufixo]).abs()
    return dict(modelo=erro.mean(), ancora=(df["sbp_target_a"] - df["sbp_calib" + sufixo]).abs().mean(),
                idade=df["tempo_desde_calib_seg" + sufixo].median() / 60, erro=erro)


def calibracao_igualada(df):
    """A mesma calibração para as duas cadeias: preds_final = sbp_calib + preds_delta."""
    saida = {}
    for dono, calib in (("propria", df["sbp_calib_a"]), ("variante", df["sbp_calib_b"])):
        for lado, suf in (("propria", "_a"), ("variante", "_b")):
            saida[f"calib_{dono}_mae_{lado}"] = round((df["sbp_target_a"] - (calib + df["preds_delta" + suf])).abs().mean(), 2)
    return saida


base = carrega("baseline")
linhas = []
for v in ("nk", "nk2"):
    par = base.merge(carrega(v), on=CHAVE, suffixes=("_a", "_b"))
    desvio = (par["sbp_target_a"] - par["sbp_target_b"]).abs()
    for tol in TOLERANCIAS:
        comum = par[desvio < tol] if tol else par[desvio == 0]
        if comum.empty:
            continue
        a, b = metricas(comum, "_a"), metricas(comum, "_b")
        linhas.append({"variante": NOMES[v], "tolerancia_alvo_mmhg": tol, "janelas": len(comum),
                       "sujeitos": comum["subject_id"].nunique(),
                       "mae_propria": round(a["modelo"], 2), "mae_variante": round(b["modelo"], 2),
                       "ancora_propria": round(a["ancora"], 2), "ancora_variante": round(b["ancora"], 2),
                       "idade_calib_propria_min": round(a["idade"], 1), "idade_calib_variante_min": round(b["idade"], 1),
                       "pct_janelas_variante_melhor": round(100 * (b["erro"] < a["erro"]).mean(), 1),
                       "wilcoxon_p": wilcoxon(a["erro"], b["erro"])[1], **calibracao_igualada(comum)})
tab = pd.DataFrame(linhas)
os.makedirs(f"{C.DIR_RESULTADOS}/tabelas", exist_ok=True)
tab.to_csv(f"{C.DIR_RESULTADOS}/tabelas/janelas_emparelhadas.csv", index=False)
with pd.option_context("display.width", 220, "display.max_columns", None):
    print(tab.to_string(index=False))
