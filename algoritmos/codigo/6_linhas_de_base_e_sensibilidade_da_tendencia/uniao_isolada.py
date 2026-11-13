#!/usr/bin/env python3
"""Ciclo de serviço da união quando cada direção é afinada ISOLADAMENTE.

Mostra que fixar cada detetor em X min/h de forma isolada dá uma união com ciclo de serviço acima do limite (34,3 % a 10 min/h, 30,3 % a 8 min/h), e que os dois alarmes somados (cerca de 21/h) dão cerca de 9/h na união (Cap. 6, `sec:vpp` e discussão). Usa as funções do `04_otimizacao_conjunta_causal.py`:

  1. por direção, no DESENVOLVIMENTO, entre as configurações (limiar, limite de duração) da grelha do 04
     com ciclo de serviço <= 30 % e falso alarme da própria direção <= X min/h, escolhe-se a de maior
     sensibilidade;
  2. mede-se a UNIÃO das duas escolhas no HOLDOUT.

Uso: python uniao_isolada.py <pasta com o 04 e os resultados>   (lê; só escreve um .json ao lado deste ficheiro)
     por omissão a pasta desta árvore (3 dobras); passar a pasta original para validar a receita.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = sys.argv[1] if len(sys.argv) > 1 else f"{AQUI}/causal09"
ETIQ = "3dobras" if BASE.startswith(AQUI) else "5dobras_original"

spec = importlib.util.spec_from_file_location("conj", f"{BASE}/04_otimizacao_conjunta_causal.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)            # só define funções e constantes (main protegido)
c.RAIZ = BASE                          # cache_scores_*.npz e ponto_operacao_quantilica.json

cfg = json.load(open(f"{BASE}/ponto_operacao_quantilica.json"))
dev, hold, feats = c.preparar()
for d in ('descida', 'subida'):
    o, h = c.scores(dev, hold, feats, d)
    dev[f'p_{d}'], hold[f'p_{d}'] = o, h
cols = ['p_descida', 'p_subida', 'y_descida', 'y_subida']
cat = {'dev': c.concatenar(dev, cols), 'hold': c.concatenar(hold, cols)}
TAUS = {d: np.round(np.linspace(*np.percentile(dev[f'p_{d}'], [2, 98]), 18), 2) for d in ('descida', 'subida')}


def sozinha(dados, M, direcao):
    """sensibilidade, falso alarme (min/h) e ciclo de serviço de UMA direção, por (limiar, limite)."""
    n = len(dados['p_descida'])
    n_real = int(np.isfinite(dados['p_descida']).sum())      # Blocos reais, sem os separadores entre sessões
    y = dados[f'y_{direcao}'].astype(bool)
    ev = c.runs_np(y)
    cs = c.mascara_eventos(ev, n)
    horas = n_real * c.BLOCO_SEG / 3600.0
    out = {}
    for k, (al, m) in M.items():
        if k[0] != direcao:
            continue
        fa_n, fa_b = c.falsos(al, cs)
        out[k] = dict(sens=c.detecta(al, ev) / max(len(ev), 1), FA_min_h=fa_b * c.BLOCO_SEG / 60 / horas,
                      FA_h=fa_n / horas, duty=m.sum() / n_real)
    return pd.DataFrame(out).T


M = {s: c.mapa(cat[s], cfg, TAUS) for s in ('dev', 'hold')}
res = {}


def uniao_de(kd, ks):
    dados = cat['hold']
    n = len(dados['p_descida'])
    ev_d, ev_s = c.runs_np(dados['y_descida'].astype(bool)), c.runs_np(dados['y_subida'].astype(bool))
    ev_any = c.runs_np(dados['y_descida'].astype(bool) | dados['y_subida'].astype(bool))
    def norm(k):   # o DataFrame transforma o limite None em NaN
        return (k[0], k[1], None if pd.isna(k[2]) else k[2])
    kd, ks = norm(kd), norm(ks)
    (al_d, m_d), (al_s, m_s) = M['hold'][kd], M['hold'][ks]
    u = c.avaliar_par(al_d, m_d, al_s, m_s, ev_d, ev_s, c.mascara_eventos(ev_any, n), int(np.isfinite(dados['p_descida']).sum()),
                      c.detecta(al_d, ev_d), c.detecta(al_s, ev_s))
    def linha(df, k):
        for idx, r in df.iterrows():
            if idx[0] == k[0] and abs(float(idx[1]) - float(k[1])) < 1e-9 and pd.isna(idx[2]) == pd.isna(k[2]) \
                    and (pd.isna(k[2]) or idx[2] == k[2]):
                return r
    u = dict(u)
    u['soma_FA_h'] = float(linha(SOZ['hold']['descida'], kd).FA_h) + float(linha(SOZ['hold']['subida'], ks).FA_h)
    return u


SOZ = {s: {d: sozinha(cat[s], M[s], d) for d in ('descida', 'subida')} for s in ('dev', 'hold')}
for X in (8, 10):
    for sel in ('dev', 'hold'):
        for fa in ('FA_min_h', 'FA_h'):
            for caps in ('todos', 'sem_cap'):
                escolha = {}
                for d in ('descida', 'subida'):
                    t = SOZ[sel][d].copy()
                    t['key'] = list(t.index)
                    t = t[(t[fa].astype(float) <= X) & (t.duty.astype(float) <= c.DUTY_MAX)]
                    if caps == 'sem_cap':
                        t = t[[pd.isna(k[2]) for k in t.key]]
                    if len(t) == 0:
                        break
                    escolha[d] = t.loc[t.sens.astype(float).idxmax(), 'key']
                if len(escolha) < 2:
                    print(f"[{ETIQ}] X={X}|sel={sel}|{fa}|caps={caps}: sem configuração admissível", flush=True)
                    continue
                u = uniao_de(escolha['descida'], escolha['subida'])
                nome = f"X={X}|sel={sel}|{fa}|caps={caps}"
                res[nome] = dict(escolha_descida=[float(escolha['descida'][1]), escolha['descida'][2]],
                                 escolha_subida=[float(escolha['subida'][1]), escolha['subida'][2]],
                                 uniao_duty=round(float(u['u_duty']), 4), uniao_FA_h=round(float(u['u_FA_h']), 2), uniao_FA_min_h=round(float(u['u_FA_min_h']), 2), soma_FA_h=round(float(u['soma_FA_h']), 2),
                                 sens_d=round(float(u['sens_d']), 4), sens_s=round(float(u['sens_s']), 4))
                print(f"[{ETIQ}] {nome}: união duty {100*u['u_duty']:.1f}% FA_h união {u['u_FA_h']:.1f} (soma das duas direções {u['soma_FA_h']:.1f})", flush=True)

json.dump(res, open(f"{AQUI}/uniao_isolada_{ETIQ}.json", "w"), indent=2, ensure_ascii=False)
