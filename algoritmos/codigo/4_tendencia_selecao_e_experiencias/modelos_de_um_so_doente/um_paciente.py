"""um_paciente.py — um modelo treinado só com um doente aprende mais do que o modelo geral nesse doente?

Para cada doente escolhido (do DESENVOLVIMENTO, nunca do teste):
  geral        XGBoost quantílico das 43 características treinado em TODOS os outros doentes (hiperparâmetros de 01_optuna_tempo.py)
  local        o mesmo, treinado só nas sessões de treino desse doente
  geral+local  os outros doentes mais as sessões de treino do doente, com peso PESO
O doente é dividido em 3 grupos de sessões CONSECUTIVAS (sessões ordenadas pelo identificador, que segue a ordem dos registos): cada grupo é, à vez,
o teste e os outros dois o treino. Dividir ao acaso por bloco daria resultados falsamente ótimos (blocos vizinhos são quase iguais). As previsões
fora do treino juntam-se e medem-se: AUC e sensibilidade por episódio a 4-10 min/h de falso alarme (fronteira, por direção, como em sensibilidade_tempo.py),
com IC 95 % da diferença face ao modelo geral por bootstrap de SESSÕES. O doente estável serve para a carga de falso alarme do modelo geral e
para ver se os scores o separam do doente com muitos alarmes.
Uso: python3 um_paciente.py ID [ID ...]      (dados em ./dados, hiperparâmetros best_params_quant_<direção>_tudo.json)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor

AQUI = os.path.dirname(os.path.abspath(__file__))
TF = os.path.normpath(os.path.join(AQUI, "..", "..", "3_tendencia_modelo_final", "tendencia_tempo_fixo"))        # tempo_fixo.py e os best_params_*.json só existem em tendencia_tempo_fixo/
sys.path.insert(0, TF)
import tempo_fixo as tf  # noqa: E402

DADOS = os.environ.get("DADOS", f"{AQUI}/dados")
NJOBS = int(os.environ.get("NJOBS", "14"))
PESO, K, B = 10.0, 3, 1000
BLOCO_SEG, GAP_MERGE, PRE_CREDITO, DUTY_MAX = 15, 4, 8, 0.30
ALVOS = np.array([4, 6, 8, 10])
TAU = {'descida': 0.10, 'subida': 0.90}
SINAL = {'descida': -1.0, 'subida': 1.0}
LIM_PONTO = {'descida': 11.34, 'subida': 10.92}       # limiares do ponto de operação das 43 (ponto_tempo_reafinado.csv)

dev, V = tf.preparar_grade(DADOS, 'dev')
dev['subject_id'] = dev.subject_id.astype(str)
FEATS = V['tudo']
tem = dev.delta.notna().values
print(f"desenvolvimento: {int(tem.sum())} blocos com alvo, {dev.subject_id.nunique()} doentes", flush=True)


def runs(mask, gap=GAP_MERGE):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    cortes = np.where(np.diff(idx) > gap + 1)[0]
    return [(g[0], g[-1]) for g in np.split(idx, cortes + 1)]


def contagens(y, alarme, eventos):
    """det, n_ev, min de falso alarme, blocos em alarme, blocos, n.º de alarmes, n.º de alarmes falsos (sem episódio)."""
    al = runs(alarme)
    toca = lambda a0, a1: any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (e0, e1) in eventos)
    det = sum(any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (a0, a1) in al) for (e0, e1) in eventos)
    falsos = [(a0, a1) for (a0, a1) in al if not toca(a0, a1)]
    fa = sum((a1 - a0 + 1) * BLOCO_SEG / 60.0 for (a0, a1) in falsos)
    return det, len(eventos), fa, sum(a1 - a0 + 1 for (a0, a1) in al), len(y), len(al), len(falsos)


def fronteira(soma):
    """soma (T, 7) -> (len(ALVOS), 4): sensibilidade, VPP, alarmes/h e falsos alarmes/h interpolados ao longo da fronteira, a ALVOS min/h de falso alarme
    (ciclo de serviço <= 30 %). O VPP é a fração dos ALARMES (e não dos blocos) que tocam um episódio."""
    vazio = np.full((len(ALVOS), 4), np.nan)
    horas = soma[:, 4] * BLOCO_SEG / 3600.0
    sens = soma[:, 0] / np.maximum(soma[:, 1], 1)
    fa = soma[:, 2] / horas
    vpp = 1 - soma[:, 6] / np.maximum(soma[:, 5], 1)
    al_h, fal_h = soma[:, 5] / horas, soma[:, 6] / horas
    ok = (soma[:, 3] / soma[:, 4]) <= DUTY_MAX
    if ok.sum() < 2:
        return vazio
    s, f, v, ah, fh = sens[ok], fa[ok], vpp[ok], al_h[ok], fal_h[ok]
    o = np.argsort(f, kind='stable'); s, f, v, ah, fh = s[o], f[o], v[o], ah[o], fh[o]
    melhor = np.maximum.accumulate(s)
    m = np.concatenate([[True], s[1:] > melhor[:-1]]); s, f, v, ah, fh = s[m], f[m], v[m], ah[m], fh[m]
    if len(s) < 2:
        return vazio
    return np.column_stack([np.interp(ALVOS, f, x, left=np.nan, right=x[-1]) for x in (s, v, ah, fh)])


def modelo(bp, direcao, X, y, w=None):
    return XGBRegressor(**bp, objective='reg:quantileerror', quantile_alpha=TAU[direcao], n_jobs=NJOBS, tree_method='hist').fit(X, y, sample_weight=w)


def sessoes_do_doente(pid, direcao, scores):
    """Lista de (session_id, y bool, score por bloco na grade) do doente; score NaN onde não há previsão."""
    out = []
    for sid, g in dev[(dev.subject_id == pid)].groupby('session_id', sort=True):
        out.append((sid, g[f'y_{direcao}'].fillna(0).values.astype(bool), scores[g.index.values]))
    return out


def avaliar(sess, nome_col):
    """Soma por sessão das contagens a 120 limiares; devolve array (n_sessoes, T, 5) e os limiares."""
    todos = np.concatenate([s[2][~np.isnan(s[2])] for s in sess])
    lim = np.percentile(todos, np.linspace(55, 99.95, 120))
    por = np.zeros((len(sess), len(lim), 7))
    for i, (sid, y, sc) in enumerate(sess):
        ev = runs(y)
        sc = np.nan_to_num(sc, nan=-np.inf)
        for t, tau in enumerate(lim):
            por[i, t] = contagens(y, sc >= tau, ev)
    return por


resultados = []
for pid in sys.argv[1:]:
    papel = dev[dev.subject_id == pid]
    print(f"\n=== {pid}: {papel.session_id.nunique()} sessões, {int(papel.delta.notna().sum())} blocos com alvo ===", flush=True)
    for direcao in ('descida', 'subida'):
        bp = json.load(open(f"{TF}/best_params_quant_{direcao}_tudo.json"))['best_params']
        eh = (dev.subject_id == pid).values
        treino_g = tem & ~eh
        X, yr = dev[FEATS].values, dev['delta'].values
        g = modelo(bp, direcao, X[treino_g], yr[treino_g])
        sc_g = np.full(len(dev), np.nan)
        idx_p = np.where(eh & tem)[0]
        sc_g[idx_p] = SINAL[direcao] * g.predict(X[idx_p])
        # grupos de sessões consecutivas
        sids = sorted(dev.loc[eh, 'session_id'].unique())
        grupos = np.array_split(np.arange(len(sids)), K)
        sc_l, sc_gl = np.full(len(dev), np.nan), np.full(len(dev), np.nan)
        rich = dev.loc[eh, f'y_{direcao}'].fillna(0).sum() >= 100
        for k, gi in enumerate(grupos):
            teste_s = {sids[i] for i in gi}
            em_teste = eh & dev.session_id.isin(teste_s).values
            em_treino = eh & ~em_teste & tem
            idx_te = np.where(em_teste & tem)[0]
            if em_treino.sum() < 200 or len(idx_te) == 0:
                continue
            if rich:
                loc = modelo(bp, direcao, X[em_treino], yr[em_treino])
                sc_l[idx_te] = SINAL[direcao] * loc.predict(X[idx_te])
                m = np.concatenate([treino_g[treino_g], np.ones(int(em_treino.sum()), bool)])
                Xm = np.vstack([X[treino_g], X[em_treino]]); ym = np.concatenate([yr[treino_g], yr[em_treino]])
                w = np.concatenate([np.ones(int(treino_g.sum())), np.full(int(em_treino.sum()), PESO)])
                gl = modelo(bp, direcao, Xm, ym, w)
                sc_gl[idx_te] = SINAL[direcao] * gl.predict(X[idx_te])
            print(f"  {pid} {direcao} grupo {k + 1}/{K} feito", flush=True)
        # só os blocos de teste de algum grupo têm score local; o geral compara-se nos MESMOS blocos
        base = np.isfinite(sc_l) if rich else np.isfinite(sc_g)
        sc_g_cmp = np.where(base, sc_g, np.nan)
        modelos = {'geral': sc_g_cmp} if not rich else {'geral': sc_g_cmp, 'local': sc_l, 'geral+local': sc_gl}
        y_all = dev[f'y_{direcao}'].values
        rng = np.random.default_rng(0)
        agg = {}
        for nome, sc in modelos.items():
            ok = np.isfinite(sc) & np.isfinite(y_all)
            auc = roc_auc_score(y_all[ok], sc[ok]) if len(np.unique(y_all[ok])) == 2 else np.nan
            por = avaliar(sessoes_do_doente(pid, direcao, sc), nome)
            agg[nome] = (auc, por)
            print(f"  {pid} {direcao} {nome}: AUC {auc:.4f}", flush=True)
        # bootstrap das sessões, com os mesmos sorteios para todos os modelos
        n = len(agg['geral'][1])
        rng = np.random.default_rng(0)
        pesos = [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]
        fr = {nome: (fronteira(por.sum(0)), np.stack([fronteira(np.tensordot(w, por, axes=(0, 0))) for w in pesos])) for nome, (auc, por) in agg.items()}
        NOMES = ('sens', 'vpp', 'alarmes_h', 'falsos_h')
        for nome in modelos:
            row = dict(doente=pid, direcao=direcao, modelo=nome, n_sessoes=n, blocos_teste=int(np.isfinite(modelos[nome]).sum()),
                       blocos_evento=int(np.nansum(y_all[np.isfinite(modelos[nome])])), auc=agg[nome][0])
            for j, mn in enumerate(NOMES):
                for a_, v in zip(ALVOS, fr[nome][0][:, j]):
                    row[f'{mn}_{a_}minh'] = v
            if nome != 'geral':
                for j, mn in enumerate(NOMES):
                    dp = np.nanmean(fr[nome][0][:, j] - fr['geral'][0][:, j])
                    db = np.nanmean(fr[nome][1][:, :, j] - fr['geral'][1][:, :, j], axis=1)
                    row.update({f'dif_{mn}_media': float(dp), f'dif_{mn}_ic_baixo': float(np.nanpercentile(db, 2.5)), f'dif_{mn}_ic_alto': float(np.nanpercentile(db, 97.5))})
                    i10 = list(ALVOS).index(10)
                    d10 = fr[nome][1][:, i10, j] - fr['geral'][1][:, i10, j]
                    row.update({f'dif10_{mn}': float(fr[nome][0][i10, j] - fr['geral'][0][i10, j]), f'dif10_{mn}_ic_baixo': float(np.nanpercentile(d10, 2.5)), f'dif10_{mn}_ic_alto': float(np.nanpercentile(d10, 97.5))})
            resultados.append(row)
        # carga de falso alarme do modelo geral no limiar do ponto de operação
        tau = LIM_PONTO[direcao]
        sess = sessoes_do_doente(pid, direcao, sc_g)
        soma = np.array([contagens(y, np.nan_to_num(sc, nan=-np.inf) >= tau, runs(y)) for _, y, sc in sess]).sum(0)
        resultados.append(dict(doente=pid, direcao=direcao, modelo='geral@ponto_de_operacao', n_sessoes=len(sess), min_falso_h=soma[2] / (soma[4] * BLOCO_SEG / 3600),
                               ciclo=soma[3] / soma[4], sens=soma[0] / max(soma[1], 1)))
        pd.DataFrame(resultados).to_csv(f"{AQUI}/resultados_um_paciente.csv", index=False)
print("FIM")
