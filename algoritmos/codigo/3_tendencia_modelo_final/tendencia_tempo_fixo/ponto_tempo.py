"""ponto_tempo.py — sensibilidade e VPP a 10 min/h de falso alarme (união), janela de 5 min por TEMPO, com e sem as características novas.

Mesmo protocolo de tentativas_de_melhoria/ponto_sem_portao.py (sem portão, sem limite de duração, escolha no desenvolvimento pelo critério
maximin com a união <= 10 min/h e ciclo de serviço <= 30 %, uma só leitura no teste), mas com as séries na grade de 15 s: os blocos em falta ficam
como NaN no score e 0 nos rótulos, por isso os intervalos entre alarmes e o crédito de aviso (2 min) correm em tempo real. Os hiperparâmetros são os do
modelo final em todas as variantes. Só se pontuam os blocos com alvo definido.
Variantes: base (19), apg, tudo (43). REAFINADO=1 usa os hiperparâmetros reafinados por tempo de cada variante (01_optuna_tempo.py). Uso: python3 ponto_tempo.py [pasta_dados] [pasta_com_04_e_params] [n_jobs]
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

AQUI = os.path.dirname(os.path.abspath(__file__))
DADOS = sys.argv[1] if len(sys.argv) > 1 else "dados"
BASE = sys.argv[2] if len(sys.argv) > 2 else AQUI
NJOBS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
sys.path.insert(0, AQUI)
import tempo_fixo as tf  # noqa: E402

spec = importlib.util.spec_from_file_location("c", f"{BASE}/04_otimizacao_conjunta_causal.py")
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
c.CAPS_MIN = [None]
VARIANTES = os.environ.get("VARIANTES", "base,apg,tudo").split(",")


def treinar(dev, hold, feats, direcao, nome):
    # REAFINADO=1: hiperparâmetros de 01_optuna_tempo.py para esta variante (best_params_quant_<direção>_<variante>.json, na pasta do script)
    ficheiro = f"{AQUI}/best_params_quant_{direcao}_{nome}.json" if os.environ.get("REAFINADO") else f"{BASE}/best_params_quant_{direcao}.json"
    bp = json.load(open(ficheiro))['best_params']
    q, sinal = (0.10, -1.0) if direcao == 'descida' else (0.90, 1.0)
    d, h = dev[dev.delta.notna()], hold[hold.delta.notna()]
    X, yr, yb = d[feats].values, d['delta'].values, d[f'y_{direcao}'].values
    mk = lambda: XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=q, n_jobs=NJOBS, tree_method='hist')
    oof = np.zeros(len(X))
    for tr, va in GroupKFold(n_splits=3).split(X, yb, d['subject_id'].values):
        oof[va] = sinal * mk().fit(X[tr], yr[tr]).predict(X[va])
    ph = sinal * mk().fit(X, yr).predict(h[feats].values)
    p_dev = np.full(len(dev), np.nan); p_dev[dev.delta.notna().values] = oof
    p_h = np.full(len(hold), np.nan); p_h[hold.delta.notna().values] = ph
    return p_dev, p_h


def vpp(al, cs):
    if al.size == 0:
        return float('nan'), 0
    return float(((cs[al[:, 1] + 1] - cs[al[:, 0]]) > 0).mean()), len(al)


ALVOS = [int(x) for x in os.environ.get('ALVOS', '10').split(',')]      # níveis de carga da união (min/h); o limiar escolhe-se no desenvolvimento em cada um


def ponto(nome, dev, hold, feats):
    cfg = json.load(open(f"{BASE}/ponto_operacao_quantilica.json"))
    for d in cfg:
        cfg[d]["portao_limite"] = 1e9
    cols = ["p_descida", "p_subida", "y_descida", "y_subida"]
    cat = {"dev": c.concatenar(dev, cols), "hold": c.concatenar(hold, cols)}
    TAUS = {d: np.round(np.linspace(*np.nanpercentile(dev[f"p_{d}"], [2, 98]), 18), 2) for d in ("descida", "subida")}
    t_dev, _, _ = c.varrer(cat["dev"], cfg, TAUS)
    t_hold, nd, ns = c.varrer(cat["hold"], cfg, TAUS)
    M = c.mapa(cat["hold"], cfg, TAUS)
    horas = int(np.isfinite(cat['hold']['p_descida']).sum()) * c.BLOCO_SEG / 3600
    ev = {k: c.runs_np(cat["hold"][f"y_{k}"].astype(bool)) for k in ("descida", "subida")}
    ev_any = c.runs_np(cat["hold"]["y_descida"].astype(bool) | cat["hold"]["y_subida"].astype(bool))
    saida = []
    for alvo in ALVOS:
        sel = t_dev[(t_dev.u_FA_min_h <= alvo) & (t_dev.u_duty <= 0.30)]
        e = sel.loc[np.minimum(sel.sens_d, sel.sens_s).idxmax()]
        h = t_hold[(t_hold.tau_d == e.tau_d) & (t_hold.tau_s == e.tau_s)].iloc[0]
        md, ms = M[("descida", round(float(e.tau_d), 2), None)], M[("subida", round(float(e.tau_s), 2), None)]
        n = len(md[1]); al_u = c.runs_np(md[1] | ms[1])
        fa_n, _ = c.falsos(al_u, c.mascara_eventos(ev_any, n))
        v_d, n_d = vpp(md[0], c.mascara_eventos(ev["descida"], n))
        v_s, n_s = vpp(ms[0], c.mascara_eventos(ev["subida"], n))
        saida.append(dict(alvo=alvo, variante=nome, n_caract=len(feats), tau_d=float(e.tau_d), tau_s=float(e.tau_s),
                          sens_d=round(float(h.sens_d), 3), sens_s=round(float(h.sens_s), 3), eventos_d=nd, eventos_s=ns,
                          min_falso_h=round(float(h.u_FA_min_h), 1), ciclo_servico=round(float(h.u_duty), 3),
                          alarmes_h=round(len(al_u) / horas, 1), falsos_alarmes_h=round(fa_n / horas, 1),
                          vpp_uniao=round(1 - fa_n / max(len(al_u), 1), 3), vpp_descida=round(v_d, 3), vpp_subida=round(v_s, 3),
                          alarmes_h_descida=round(n_d / horas, 1), alarmes_h_subida=round(n_s / horas, 1)))
    return saida


dev, V = tf.preparar_grade(DADOS, 'dev')
hold, _ = tf.preparar_grade(DADOS, 'holdout')
res = []
for nome in VARIANTES:
    for d in ("descida", "subida"):
        dev[f"p_{d}"], hold[f"p_{d}"] = treinar(dev, hold, V[nome], d, nome)
        print(f"  {nome} {d}: AUC dev {roc_auc_score(dev.loc[dev.delta.notna(), 'y_' + d], dev.loc[dev.delta.notna(), 'p_' + d]):.4f} "
              f"teste {roc_auc_score(hold.loc[hold.delta.notna(), 'y_' + d], hold.loc[hold.delta.notna(), 'p_' + d]):.4f}", flush=True)
    for d in ("descida", "subida"):
        dev[f"y_{d}"] = dev[f"y_{d}"].fillna(0); hold[f"y_{d}"] = hold[f"y_{d}"].fillna(0)
    for r_ in ponto(nome, dev, hold, V[nome]):
        res.append(r_); print(r_, flush=True)
    for d in ("descida", "subida"):                       # repõe os NaN dos rótulos para a variante seguinte
        dev.loc[~dev.delta.notna(), f"y_{d}"] = np.nan; hold.loc[~hold.delta.notna(), f"y_{d}"] = np.nan
pd.DataFrame(res).to_csv(f"{AQUI}/ponto_tempo{'_reafinado' if os.environ.get('REAFINADO') else ''}{os.environ.get('SUFIXO', '')}.csv", index=False)
print("FIM")
