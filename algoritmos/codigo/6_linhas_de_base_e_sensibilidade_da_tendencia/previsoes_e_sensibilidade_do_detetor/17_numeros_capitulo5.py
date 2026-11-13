#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recalcula, na variante CAUSAL, os números do Capítulo 5 que faltavam.

Sete afirmações, todas obtidas das previsões causais já guardadas
(previsoes_causal_{direcao}.csv.gz, do 09), sem treinar nada:

  1. tab:triviais  — sensibilidade do modelo contra alarme aleatório de igual
                     estrutura e contra alarme sempre ligado, sem portão
  2. tab:vpp       — valor preditivo positivo a seis níveis de sensibilidade
  3. VPP no ponto de operação adotado (a frase "sete em cada oito")
  4. concentração da carga ("10% das sessões produzem X% do falso alarme")
  5. episódios falhados: quantos e que fração
  6. duração mediana dos falhados contra os detetados
  7. pico do score dentro dos episódios falhados

CONVENÇÕES, iguais às do 08_ponto_operacao_causal.py:
  · alarme e episódio são CORRIDAS de blocos, com fusão de intervalos até 1 min
  · um alarme conta como verdadeiro se sobrepuser um episódio, com crédito de
    2 min de antecipação (detetar antes conta como detetar)
  · sensibilidade é por episódio, não por bloco

O ponto de operação é o que está em produção: limiar 9,13
(descida) e 9,26 (subida), portão de 4 blocos com amplitude <= 3,66 / <= 3,85.
Para a tab:triviais o portão é desligado, como diz a legenda da tabela.
"""

import json
import os

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.abspath(__file__))
BLOCO_SEG = 15
GAP_MERGE = int(1 * 60 / BLOCO_SEG)
PRE_CREDITO = int(2 * 60 / BLOCO_SEG)
SENS_ALVO = [0.80, 0.60, 0.50, 0.30, 0.10, 0.04]
N_BOOT = 2000
RNG = np.random.default_rng(42)


def runs(mask, gap=GAP_MERGE):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    cortes = np.where(np.diff(idx) > gap + 1)[0]
    return [(g[0], g[-1]) for g in np.split(idx, cortes + 1)]


def sessoes(df, com_portao, n=4, lim=None):
    out = []
    for _, g in df.groupby('session_id'):
        p = g.p.values
        y = g.y.values.astype(bool)
        if com_portao:
            s = pd.Series(p).shift(1).rolling(n, min_periods=n)
            amp = (s.max() - s.min()).values
            ok = ~np.isfinite(amp) | (amp <= lim)
        else:
            ok = np.ones(len(p), bool)
        out.append((y, p, ok))
    return out


def medir(grupos, tau):
    """det, n_ev, n_alarmes, n_alarmes_verdadeiros, blocos_alarme, total, fa_min."""
    det = n_ev = n_al = n_true = blocos = tot = 0
    fa_min = 0.0
    for y, p, ok in grupos:
        tot += len(y)
        ev = runs(y)
        n_ev += len(ev)
        al = runs((p >= tau) & ok)
        n_al += len(al)
        for (a0, a1) in al:
            blocos += (a1 - a0 + 1)
            if any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (e0, e1) in ev):
                n_true += 1
            else:
                fa_min += (a1 - a0 + 1) * BLOCO_SEG / 60.0
        for (e0, e1) in ev:
            if any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (a0, a1) in al):
                det += 1
    horas = tot * BLOCO_SEG / 3600.0
    return dict(det=det, n_ev=n_ev, n_al=n_al, n_true=n_true,
                sens=det / max(n_ev, 1), vpp=n_true / max(n_al, 1),
                duty=blocos / tot, alarmes_h=n_al / horas,
                fa_min_h=fa_min / horas, horas=horas)


def tau_para_sens(grupos, alvo, lo, hi):
    """Bissecção no limiar até atingir a sensibilidade pedida."""
    for _ in range(40):
        m = (lo + hi) / 2
        if medir(grupos, m)['sens'] >= alvo:
            lo = m
        else:
            hi = m
    return lo


def aleatorio_igual_estrutura(grupos, tau, n_rep=N_BOOT):
    """Mesmo número e duração de alarmes, posições sorteadas dentro da sessão."""
    sens = []
    for _ in range(n_rep):
        det = n_ev = 0
        for y, p, ok in grupos:
            ev = runs(y)
            n_ev += len(ev)
            al = runs((p >= tau) & ok)
            novos = []
            for (a0, a1) in al:
                dur = a1 - a0 + 1
                if dur >= len(y):
                    novos.append((0, len(y) - 1)); continue
                ini = RNG.integers(0, len(y) - dur + 1)
                novos.append((ini, ini + dur - 1))
            for (e0, e1) in ev:
                if any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (a0, a1) in novos):
                    det += 1
        sens.append(det / max(n_ev, 1))
    return np.array(sens)


def main():
    cfg = {'descida': (10.21, 1e9), 'subida': (10.34, 1e9)}
    saida = {}

    for direcao, (tau_op, lim) in cfg.items():
        d = pd.read_csv(f"{RAIZ}/previsoes_causal_{direcao}.csv.gz")
        g_sem = sessoes(d, com_portao=False)
        g_com = sessoes(d, com_portao=True, lim=lim)
        lo, hi = d.p.min(), d.p.max()
        print(f"\n{'='*74}\n{direcao.upper()}  (limiar de operação {tau_op} mmHg)\n{'='*74}", flush=True)

        # ── 3. ponto de operação adotado, COM portão ─────────────────────────
        op = medir(g_com, tau_op)
        print(f"  ponto adotado: sens {op['sens']:.4f} | VPP {op['vpp']*100:.1f}\\% "
              f"({op['n_true']}/{op['n_al']}) | duty {op['duty']*100:.1f}\\% | "
              f"{op['alarmes_h']:.1f} alarmes/h | {op['fa_min_h']:.2f} min/h")
        um_em = 1 / max(op['vpp'], 1e-9)
        print(f"      -> 1 em cada {um_em:.1f} alarmes é verdadeiro")

        # ── 1. tab:triviais, SEM portão, ao duty do ponto adotado ────────────
        tau_t = tau_op
        mt = medir(g_sem, tau_t)
        alea = aleatorio_igual_estrutura(g_sem, tau_t)
        print(f"\n  tab:triviais (sem portão, limiar {tau_t})")
        print(f"    modelo    : sens {mt['sens']:.3f} | duty {mt['duty']:.3f} | "
              f"{mt['alarmes_h']:.1f} alarmes/h")
        print(f"    aleatório : sens {alea.mean():.3f} "
              f"[{np.percentile(alea,2.5):.3f}--{np.percentile(alea,97.5):.3f}] "
              f"(mesma estrutura)")
        print("    sempre ligado: sens 1,000 | duty 1,000")
        print(f"    ganho sobre o nulo: {mt['sens']-alea.mean():+.3f}")

        # ── 2. tab:vpp ───────────────────────────────────────────────────────
        print("\n  tab:vpp (sem portão)")
        vpp = []
        for s in SENS_ALVO:
            t = tau_para_sens(g_sem, s, lo, hi)
            m = medir(g_sem, t)
            vpp.append((s, m['vpp'], m['n_true'], m['n_al']))
            print(f"    sens {s:.2f} -> VPP {m['vpp']*100:5.1f}\\% "
                  f"({m['n_true']}/{m['n_al']})  [limiar {t:.2f}]")

        # ── 4. concentração da carga por sessão ──────────────────────────────
        carga = []
        for _, gs in d.groupby('session_id'):
            p, y = gs.p.values, gs.y.values.astype(bool)
            ev = runs(y)
            fa = 0.0
            for (a0, a1) in runs(p >= tau_op):
                if not any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (e0, e1) in ev):
                    fa += (a1 - a0 + 1) * BLOCO_SEG / 60.0
            carga.append(fa)
        carga = np.sort(np.array(carga))[::-1]
        top10 = carga[:max(1, len(carga)//10)].sum() / max(carga.sum(), 1e-9)
        zero = (carga == 0).mean()
        print(f"\n  carga: 10\\% das sessões concentram {top10*100:.0f}\\% do falso alarme; "
              f"{zero*100:.0f}\\% das sessões não produzem nenhum")

        # ── 5/6/7. episódios falhados ────────────────────────────────────────
        falh_dur, det_dur, falh_pico, det_pico = [], [], [], []
        n_falh = n_tot = 0
        for y, p, ok in g_com:
            ev = runs(y)
            al = runs((p >= tau_op) & ok)
            for (e0, e1) in ev:
                n_tot += 1
                dur = (e1 - e0 + 1) * BLOCO_SEG / 60.0
                pico = float(p[e0:e1+1].max())
                if any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (a0, a1) in al):
                    det_dur.append(dur); det_pico.append(pico)
                else:
                    n_falh += 1
                    falh_dur.append(dur); falh_pico.append(pico)
        print(f"\n  episódios falhados: {n_falh} de {n_tot} ({100*n_falh/max(n_tot,1):.1f}\\%)")
        print(f"    duração mediana: falhados {np.median(falh_dur):.2f} min | "
              f"detetados {np.median(det_dur):.2f} min")
        print(f"    pico do score  : falhados {np.median(falh_pico):.1f} mmHg | "
              f"detetados {np.median(det_pico):.1f} mmHg")

        saida[direcao] = dict(
            op_sens=round(op['sens'], 4), op_vpp=round(op['vpp'], 4),
            op_true=op['n_true'], op_al=op['n_al'],
            op_duty=round(op['duty'], 4), op_alarmes_h=round(op['alarmes_h'], 1),
            triv_sens=round(mt['sens'], 3), triv_duty=round(mt['duty'], 3),
            triv_alarmes_h=round(mt['alarmes_h'], 1),
            aleat_sens=round(float(alea.mean()), 3),
            aleat_lo=round(float(np.percentile(alea, 2.5)), 3),
            aleat_hi=round(float(np.percentile(alea, 97.5)), 3),
            ganho_nulo=round(mt['sens'] - float(alea.mean()), 3),
            vpp=[(s, round(v, 4), t, a) for s, v, t, a in vpp],
            carga_top10=round(float(top10), 3), carga_zero=round(float(zero), 3),
            falhados=n_falh, episodios=n_tot,
            dur_falh=round(float(np.median(falh_dur)), 2),
            dur_det=round(float(np.median(det_dur)), 2),
            pico_falh=round(float(np.median(falh_pico)), 1),
            pico_det=round(float(np.median(det_pico)), 1))

    json.dump(saida, open(f"{RAIZ}/numeros_capitulo5_causal.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\nGuardado: {RAIZ}/numeros_capitulo5_causal.json")


if __name__ == "__main__":
    main()
