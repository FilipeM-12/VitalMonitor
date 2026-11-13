#!/usr/bin/env python3
"""Igual ao ic_curtos_vs_5.py mas sobre a tabela de teste atual (causal09/dados/holdout_ppg.csv), porque holdout_ppg_apg.csv já não existe. Generaliza ic_so3_vs_so5.py a 2 e 3 min. Para cada horizonte curto C (2 ou 3 min), compara os episódios que só a
definição de 5 min apanha (sem sobreposição com os de C) com os que só a de C apanha (sem sobreposição com os de 5).
Referência justa: mediana dos 2 min antes do início da janela W de cada episódio. Mantém = mediana dos 9 aos 11 min
depois a ≥ 10 mmHg da referência, no sentido do evento; variação líquida = mediana dos 6 aos 10 min depois menos a
referência, com o sinal da direção. IC 95 % por reamostragem de sujeitos (2000 réplicas, semente 0)."""
import importlib.util, json, os
import numpy as np, pandas as pd
AQUI = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("traj", "/home/filipe/Desktop/Tese/algoritmo-ml/2026-09-10_vias_causais/10_trajetorias_definicao.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
WB = {2: 8, 3: 12, 5: 20}; B = 2000
REF = int(os.environ.get("REF_BLOCOS", 8))   # blocos da referência (8 = 2 min); sensibilidade com 4 e 16


def episodios(df, sinal, curto):
    gb = df.groupby("session_id")["sbp_target"]
    y = {w: (sinal * (df.sbp_target - gb.shift(WB[w])) >= T.D).fillna(False).values for w in (curto, 5)}
    reg = []
    for sid, idx in df.groupby("session_id", sort=False).indices.items():
        sbp = df.sbp_target.values[idx]; n = len(sbp); suj = df.subject_id.values[idx[0]]
        ep = {w: T.runs(y[w][idx]) for w in y}
        for w, outro in ((5, curto), (curto, 5)):
            for s0, s1 in ep[w]:
                if any(T.overlaps(s0, s1, c, e) for c, e in ep[outro]):
                    continue
                a = s0 - WB[w] - REF
                if a < 0 or s0 + T.RET_B >= n:
                    continue
                pre = np.nanmedian(sbp[a:s0 - WB[w] + 1]); pos = np.nanmedian(sbp[s0 + T.POS_A:s0 + T.POS_B + 1])
                ret = np.nanmedian(sbp[s0 + T.RET_A:s0 + T.RET_B + 1])
                if not np.isfinite(pre + pos + ret):
                    continue
                reg.append(dict(suj=suj, w=w, liq=sinal * (pos - pre), mantem=sinal * (ret - pre) >= T.RET_LIMIAR))
    return pd.DataFrame(reg)


def dif(r, curto):
    a, b = r[r.w == 5], r[r.w == curto]
    return 100 * (a.mantem.mean() - b.mantem.mean()), a.liq.median() - b.liq.median()


df = pd.read_csv("/home/filipe/Desktop/Tese/algoritmo-ml/RESULTADOS_RELATORIO/tudo_3_dobras/causal09/dados/holdout_ppg.csv", usecols=['subject_id', 'session_id', 'bloco_id', 'sbp_target']).sort_values(['session_id', 'bloco_id']).reset_index(drop=True); out = {}
for curto in (2, 3):
    for d, sinal in (("descida", -1.0), ("subida", 1.0)):
        rng = np.random.default_rng(0)
        r = episodios(df, sinal, curto); sujs = r.suj.unique(); grupos = {s: g for s, g in r.groupby("suj")}
        bb = np.array([dif(pd.concat([grupos[s] for s in rng.choice(sujs, len(sujs))], ignore_index=True), curto)
                       for _ in range(B)])
        e = dif(r, curto); a, b = r[r.w == 5], r[r.w == curto]
        out[f"{curto}_vs_5_{d}"] = dict(
            n_so5=len(a), n_so_curto=len(b), sujeitos=len(sujs),
            mantem_so5=round(100 * a.mantem.mean(), 1), mantem_so_curto=round(100 * b.mantem.mean(), 1),
            dif_mantem_pp=round(e[0], 1), ic_mantem=[round(x, 1) for x in np.percentile(bb[:, 0], [2.5, 97.5])],
            liq_so5=round(a.liq.median(), 1), liq_so_curto=round(b.liq.median(), 1),
            dif_liq=round(e[1], 1), ic_liq=[round(x, 1) for x in np.percentile(bb[:, 1], [2.5, 97.5])])
        print(f"{curto}_vs_5_{d}", out[f"{curto}_vs_5_{d}"], flush=True)
json.dump(out, open("/home/filipe/Desktop/Tese/algoritmo-ml/Tentativa_9_10/resultado_ic_curtos_vs_5_tabela_atual" + ("" if REF == 8 else f"_ref{REF}") + ".json", "w"), indent=1)
