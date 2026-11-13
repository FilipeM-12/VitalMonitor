#!/usr/bin/env python3
"""Tempo de cálculo por janela da extração NeuroKit da variante de FILTRAGEM AGRESSIVA (`features_*_com_nk.csv`), contra a cadeia própria e contra a
sequência da variante parcial (NeuroKit2) já medida em custo_extracao_propria_vs_neurokit.py.

A variante agressiva vem do notebook Extração_features.ipynb, secção "3. Com NeuroKit": no segmento inteiro corre `ecg_clean`, `ecg_peaks` e
`ppg_process`; por janela corre `ecg_delineate` (wavelet) e `ppg_delineate` (método da derivada, dentro de um try que ignora o erro). Em produção não há
segmento, só uma janela de 15 s a cada 5 s, por isso o custo por janela é o das cinco chamadas sobre a janela. Mesmos sinais, mesma janela e mesmas
repetições da medição da variante parcial, para as duas serem comparáveis. Escreve custo_extracao_agressiva.csv."""
import os
import sys
import time

import pandas as pd
from scipy.stats import skew

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
import custo_extracao_propria_vs_neurokit as base  # noqa: E402  (reutiliza a janela, a função de produção e as constantes)

nk, FS = base.nk, base.FS


def custo_agressiva(ecg, ppg):
    t = time.perf_counter()
    ok = True
    try:
        limpo = nk.ecg_clean(ecg, sampling_rate=FS, method="neurokit")
        if skew(limpo) < 0:                                   # corrige a inversão do sinal, como no notebook
            limpo = -limpo
        _, info = nk.ecg_peaks(limpo, sampling_rate=FS, method="neurokit")
        picos = info["ECG_R_Peaks"]
        sinais, ppg_info = nk.ppg_process(-ppg, sampling_rate=FS, method="elgendi")
        if len(picos) >= 8:
            nk.ecg_delineate(limpo, picos, sampling_rate=FS, method="dwt")
        else:
            ok = False
        try:
            nk.ppg_delineate(sinais["PPG_Clean"].values, ppg_info["PPG_Peaks"], sampling_rate=FS, method="derivative")
            ppg_delineate_existe = True
        except Exception:                                      # o notebook ignora o erro, como aqui
            ppg_delineate_existe = False
    except Exception:
        ok, ppg_delineate_existe = False, False
    return time.perf_counter() - t, ok, ppg_delineate_existe


def main():
    linhas = []
    for janela, snap, ecg, ppg in base.janelas_do_sinal():
        for _ in range(base.REPETICOES):
            tp, ok_p = base.custo_propria(snap)
            tn, ok_n = base.custo_neurokit(ecg, ppg)
            ta, ok_a, existe = custo_agressiva(ecg, ppg)
            linhas.append(dict(janela=janela, propria_s=tp, parcial_s=tn, agressiva_s=ta, agressiva_extraiu=ok_a, ppg_delineate_existe=existe))
    d = pd.DataFrame(linhas)
    d.to_csv(f"{RAIZ}/custo_extracao_agressiva.csv", index=False)
    print(f"versão do NeuroKit2: {nk.__version__} | ppg_delineate corre sem erro: {d.ppg_delineate_existe.mean():.0%} das janelas")
    for nome, col in (("cadeia própria", "propria_s"), ("NeuroKit2 parcial", "parcial_s"), ("NeuroKit agressiva", "agressiva_s")):
        print(f"{nome:20s} mediana {d[col].median() * 1000:7.2f} ms   min {d[col].min() * 1000:7.2f}   max {d[col].max() * 1000:7.2f}")
    print(f"razão agressiva / própria: {d.agressiva_s.median() / d.propria_s.median():.1f}x | razão agressiva / parcial: {d.agressiva_s.median() / d.parcial_s.median():.2f}")


if __name__ == "__main__":
    main()
