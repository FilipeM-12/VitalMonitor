"""bootstrap_ponto.py — intervalos de confiança (bootstrap de DOENTES) da sensibilidade, do VPP e da carga nos pontos de operação já escolhidos.

Lê os limiares de ponto_tempo_reafinado.csv (escolhidos no desenvolvimento, uma só leitura no teste), pontua o teste com os modelos reafinados
(01_optuna_tempo.py) treinados em todo o desenvolvimento, e conta por doente: episódios de cada direção e quantos foram detetados, alarmes da
união e quantos tocam um episódio, minutos em falso alarme, horas. Depois sorteia os doentes com reposição (1000 reamostragens, as mesmas para
as duas variantes) e recalcula os totais: IC a 95 % de cada medida e da diferença entre as variantes (todas - base).
Uso: python3 bootstrap_ponto.py [pasta_dados] [pasta_com_04_e_params] [n_jobs]
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

AQUI = os.path.dirname(os.path.abspath(__file__))
DADOS = sys.argv[1] if len(sys.argv) > 1 else "dados"
BASE = sys.argv[2] if len(sys.argv) > 2 else AQUI
NJOBS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
N_BOOT = 1000
sys.path.insert(0, AQUI)
import tempo_fixo as tf  # noqa: E402

spec = importlib.util.spec_from_file_location("c", f"{BASE}/04_otimizacao_conjunta_causal.py")
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
VAR_A, VAR_B = os.environ.get("VAR_A", "base"), os.environ.get("VAR_B", "tudo")    # a diferença é B - A
SUF = os.environ.get("SUFIXO", "")
pontos = pd.read_csv(f"{AQUI}/ponto_tempo_reafinado{SUF}.csv").set_index('variante')
hold, V = tf.preparar_grade(DADOS, 'holdout')
dev, _ = tf.preparar_grade(DADOS, 'dev')
dev = dev[dev.delta.notna()].reset_index(drop=True)
hold['subject_id'] = hold.subject_id.astype(str)


def mascara(al, n):
    """Blocos com alarme ligado, a partir dos intervalos já fundidos (como o c.mapa), que é o que dá o ciclo de serviço do 04."""
    m = np.zeros(n, bool)
    for a0, a1 in al:
        m[a0:a1 + 1] = True
    return m


def pontuar(nome, direcao):
    bp = json.load(open(f"{AQUI}/best_params_quant_{direcao}_{nome}.json"))['best_params']
    q, sinal = (0.10, -1.0) if direcao == 'descida' else (0.90, 1.0)
    m = XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=q, n_jobs=NJOBS, tree_method='hist')
    m.fit(dev[V[nome]].values, dev['delta'].values)
    p = np.full(len(hold), np.nan); ok = hold.delta.notna().values
    p[ok] = sinal * m.predict(hold.loc[ok, V[nome]].values)
    return p


def por_doente(nome, p_d, p_s):
    tau_d, tau_s = float(pontos.loc[nome, 'tau_d']), float(pontos.loc[nome, 'tau_s'])
    linhas = {}
    for suj, g in hold.groupby('subject_id', sort=False):
        idx = g.index.values
        d = pd.DataFrame({'session_id': g.session_id.values, 'bloco_id': g.bloco_id.values, 'p_descida': p_d[idx], 'p_subida': p_s[idx],
                          'y_descida': g.y_descida.fillna(0).values, 'y_subida': g.y_subida.fillna(0).values})
        cat = c.concatenar(d, ["p_descida", "p_subida", "y_descida", "y_subida"])
        n = len(cat['p_descida'])
        md, ms = (cat['p_descida'] >= tau_d), (cat['p_subida'] >= tau_s)
        al_d, al_s, al_u = c.runs_np(md), c.runs_np(ms), c.runs_np(md | ms)
        ev_d, ev_s = c.runs_np(cat['y_descida'].astype(bool)), c.runs_np(cat['y_subida'].astype(bool))
        cs_any = c.mascara_eventos(c.runs_np(cat['y_descida'].astype(bool) | cat['y_subida'].astype(bool)), n)
        cs_d, cs_s = c.mascara_eventos(ev_d, n), c.mascara_eventos(ev_s, n)
        fa_n, fa_b = c.falsos(al_u, cs_any)
        fd, _ = c.falsos(al_d, cs_d); fs, _ = c.falsos(al_s, cs_s)
        linhas[suj] = dict(ev_d=len(ev_d), det_d=c.detecta(al_d, ev_d), ev_s=len(ev_s), det_s=c.detecta(al_s, ev_s),
                           al_u=len(al_u), fa_u=fa_n, fa_min=fa_b * c.BLOCO_SEG / 60.0, horas=int(np.isfinite(cat['p_descida']).sum()) * c.BLOCO_SEG / 3600,
                           ciclo_blocos=int(mascara(al_d, n) .__or__(mascara(al_s, n)).sum()), blocos=int(np.isfinite(cat['p_descida']).sum()),
                           al_d=len(al_d), fa_d=fd, al_s=len(al_s), fa_s=fs)
    return pd.DataFrame(linhas).T


def medidas(t):
    s = t.sum()
    return dict(sens_descida=s.det_d / s.ev_d, sens_subida=s.det_s / s.ev_s, min_falso_h=s.fa_min / s.horas,
                ciclo_servico=s.ciclo_blocos / s.blocos, alarmes_h=s.al_u / s.horas, vpp_uniao=1 - s.fa_u / s.al_u,
                vpp_descida=1 - s.fa_d / s.al_d, vpp_subida=1 - s.fa_s / s.al_s, verdadeiros_h=(s.al_u - s.fa_u) / s.horas)


T = {}
for nome in (VAR_A, VAR_B):
    p_d, p_s = pontuar(nome, 'descida'), pontuar(nome, 'subida')
    T[nome] = por_doente(nome, p_d, p_s).astype(float)
    print(nome, {k: round(v, 3) for k, v in medidas(T[nome]).items()}, flush=True)
sujs = list(T[VAR_A].index)
rng = np.random.default_rng(0)
amostras = [rng.choice(len(sujs), len(sujs)) for _ in range(N_BOOT)]
B = {n: pd.DataFrame([medidas(T[n].iloc[i]) for i in amostras]) for n in T}
saida = []
for k in B[VAR_A].columns:
    ponto = {n: medidas(T[n])[k] for n in T}
    dif = B[VAR_B][k] - B[VAR_A][k]
    saida.append(dict(medida=k, base=ponto[VAR_A], base_ic_baixo=B[VAR_A][k].quantile(.025), base_ic_alto=B[VAR_A][k].quantile(.975),
                      tudo=ponto[VAR_B], tudo_ic_baixo=B[VAR_B][k].quantile(.025), tudo_ic_alto=B[VAR_B][k].quantile(.975),
                      dif=ponto[VAR_B] - ponto[VAR_A], dif_ic_baixo=dif.quantile(.025), dif_ic_alto=dif.quantile(.975)))
r = pd.DataFrame(saida); r.to_csv(f"{AQUI}/bootstrap_ponto{SUF}.csv", index=False)
print(r.round(3).to_string(index=False)); print("FIM")
