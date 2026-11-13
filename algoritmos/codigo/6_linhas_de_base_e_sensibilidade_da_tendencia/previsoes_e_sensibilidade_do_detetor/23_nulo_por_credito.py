#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modelo contra alarme aleatório de igual estrutura, COM portão, ao ponto de
operação, na grelha gap x crédito de 0,5 / 1 / 2 / 3 min. Reutiliza o 17."""
import importlib.util, os
import numpy as np, pandas as pd
RAIZ = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("n17", f"{RAIZ}/17_numeros_capitulo5.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
cfg = {'descida': (10.21, 1e9), 'subida': (10.34, 1e9)}
V = [0.5, 1, 2, 3]
out = []
for dr, (tau, lim) in cfg.items():
    d = pd.read_csv(f"{RAIZ}/previsoes_causal_{dr}.csv.gz")
    g = m.sessoes(d, com_portao=True, lim=lim)
    for gap in V:
        for cred in V:
            m.GAP_MERGE = int(round(gap * 60 / m.BLOCO_SEG))
            m.PRE_CREDITO = int(round(cred * 60 / m.BLOCO_SEG))
            m.runs.__defaults__ = (m.GAP_MERGE,)
            mod = m.medir(g, tau)['sens']
            al = m.aleatorio_igual_estrutura(g, tau, n_rep=300)
            out.append((dr, gap, cred, mod, al.mean(), np.percentile(al, 2.5), np.percentile(al, 97.5), mod - al.mean()))
            print(f"{dr:7s} gap {gap} cred {cred} | modelo {mod*100:5.1f} | aleatório {al.mean()*100:5.1f} | ganho {100*(mod-al.mean()):+.1f}", flush=True)
pd.DataFrame(out, columns=['direcao','gap_min','credito_min','sens_modelo','sens_aleatorio','ic_inf','ic_sup','ganho']).to_csv(f"{RAIZ}/nulo_por_credito.csv", index=False)
