#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensibilidade das duas convenções do nível de episódio: o intervalo de fusão
(gap) e o crédito de antecipação. Valores 0,5 / 1 / 2 / 3 min para cada um, na
grelha completa, ao ponto de operação adotado (limiar e portão fixos), sobre as
previsões causais guardadas. Não treina nada. Reutiliza medir() do 17."""
import importlib.util, os
import pandas as pd

RAIZ = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("n17", f"{RAIZ}/17_numeros_capitulo5.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

VALORES = [0.5, 1, 2, 3]
cfg = {'descida': (10.21, 1e9), 'subida': (10.34, 1e9)}
linhas = []
for direcao, (tau, lim) in cfg.items():
    d = pd.read_csv(f"{RAIZ}/previsoes_causal_{direcao}.csv.gz")
    for gap in VALORES:
        for cred in VALORES:
            m.GAP_MERGE = int(round(gap * 60 / m.BLOCO_SEG))
            m.PRE_CREDITO = int(round(cred * 60 / m.BLOCO_SEG))
            # runs() tem gap como argumento por defeito, fixado na definição
            m.runs.__defaults__ = (m.GAP_MERGE,)
            g = m.sessoes(d, com_portao=True, lim=lim)
            r = m.medir(g, tau)
            linhas.append(dict(direcao=direcao, gap_min=gap, credito_min=cred,
                               n_episodios=r['n_ev'], sens=round(r['sens'], 4),
                               vpp=round(r['vpp'], 4), alarmes_h=round(r['alarmes_h'], 2),
                               fa_min_h=round(r['fa_min_h'], 2)))
df = pd.DataFrame(linhas)
df.to_csv(f"{RAIZ}/sensibilidade_gap_credito.csv", index=False)
pd.set_option('display.width', 200)
for dr in cfg:
    x = df[df.direcao == dr]
    print(f"\n{dr.upper()}")
    for c in ['n_episodios', 'sens', 'vpp', 'alarmes_h', 'fa_min_h']:
        print(f"-- {c} (linhas: gap, colunas: crédito)")
        print(x.pivot(index='gap_min', columns='credito_min', values=c).to_string())
