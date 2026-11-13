"""cascata_tempo.py — a cascata de janelas (2 -> 3 -> 5 min) com o modelo atual (43 características, janela por tempo exato).

Modelos: o XGBoost quantílico das 43 características para W = 2, 3 e 5 min (alvo Δ_W = PA(t) − PA(t−W) por tempo exato na grade de 15 s, evento |Δ_W| >= 15 mmHg).
Hiperparâmetros: os de janela_desenvolvimento/resultados_janela_dev.csv para W = 2 e 3, e os do modelo adotado (tendencia_tempo_fixo/best_params_quant_<direção>_tudo.json) para W = 5.
Políticas (as do 09): só_W5 (o modelo adotado sozinho) e cascata (W2 até o W3 poder mandar, depois W3 até o W5; R1 = esperar 4 blocos desde que o modelo seguinte tem saída; R2 = só trocar quando nenhum
dos dois está em alarme). Verdade: a definição de 5 minutos (a da tese). Alarmes = score >= limiar, sem portão (como no ponto de operação adotado).
Limiares: grelha de 30 quantis (55 a 99,95 %) das previsões fora da dobra no desenvolvimento, a mesma posição para os três modelos. Medida: sensibilidade por episódio a custo igualado
(4, 6, 8 e 10 min/h de falso alarme), cascata menos só_W5, com intervalo de confiança a 95 % por reamostragem dos doentes (200). Desenvolvimento = previsões fora da dobra (3 dobras por doente);
teste = modelos treinados em todo o desenvolvimento, com os limiares do desenvolvimento.
Uso: python3 cascata_tempo.py [n_threads]     Escreve cascata_tempo.json e cache em previsoes_cascata.npz.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

AQUI = os.path.dirname(os.path.abspath(__file__))
A = "/home/filipe/Desktop/Tese/algoritmo-ml"
TF = f"{A}/Tentativa_22_10/tendencia_tempo_fixo"
sys.path.insert(0, TF)
import tempo_fixo as tf  # noqa: E402

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 14
D, BLOCO_SEG, GAP_MERGE, PRE_CREDITO, DUTY_MAX, PORTAO = 15, 15, 4, 8, 0.30, 4
JAN = (2, 3, 5); ALVOS = np.array([4, 6, 8, 10]); N_TAUS = 30
TAU = {'descida': 0.10, 'subida': 0.90}; SINAL = {'descida': -1.0, 'subida': 1.0}

dev, V = tf.preparar_grade(f"{TF}/dados", 'dev'); hold, _ = tf.preparar_grade(f"{TF}/dados", 'holdout')
FEATS = V['tudo']
for d_ in (dev, hold): d_['subject_id'] = d_.subject_id.astype(str)
RJ = pd.read_csv(f"{A}/Tentativa_22_10/janela_desenvolvimento/resultados_janela_dev.csv")


def hp(W, direcao):
    if W == 5:
        return json.load(open(f"{TF}/best_params_quant_{direcao}_tudo.json"))["best_params"]
    return eval(RJ[(RJ.W == W) & (RJ.direcao == direcao)].best_params.iloc[0])


def alvo(frame, W, direcao):
    n = int(W * 60 / BLOCO_SEG)
    dl = frame['sbp_target'] - frame.groupby('session_id')['sbp_target'].shift(n)
    return np.where(dl.isna(), np.nan, ((dl <= -D) if direcao == 'descida' else (dl >= D)).astype(float)), dl.values


# ── previsões (cache) ──────────────────────────────────────────────────────────────────────────
cache = f"{AQUI}/previsoes_cascata.npz"
if os.path.exists(cache):
    z = np.load(cache); PR = {k: z[k] for k in z.files}
else:
    PR = {}
    for direcao in ('descida', 'subida'):
        for W in JAN:
            t0 = time.time()
            yd, dd = alvo(dev, W, direcao); yh, dh = alvo(hold, W, direcao)
            td, th = ~np.isnan(dd), ~np.isnan(dh)
            X, yr, suj = dev.loc[td, FEATS].values, dd[td], dev.loc[td, 'subject_id'].values
            oof = np.zeros(td.sum())
            for tr, va in GroupKFold(n_splits=3).split(X, yr, suj):
                oof[va] = SINAL[direcao] * XGBRegressor(**hp(W, direcao), objective='reg:quantileerror', quantile_alpha=TAU[direcao], n_jobs=NT, tree_method='hist').fit(X[tr], yr[tr]).predict(X[va])
            m = XGBRegressor(**hp(W, direcao), objective='reg:quantileerror', quantile_alpha=TAU[direcao], n_jobs=NT, tree_method='hist').fit(X, yr)
            pd_, ph = np.full(len(dev), np.nan), np.full(len(hold), np.nan)
            pd_[td] = oof; ph[th] = SINAL[direcao] * m.predict(hold.loc[th, FEATS].values)
            PR[f'dev_{direcao}_{W}'], PR[f'hold_{direcao}_{W}'] = pd_, ph
            print(f'{direcao} W={W}: {time.time() - t0:.0f} s', flush=True)
    np.savez_compressed(cache, **PR)


def runs(mask, gap=GAP_MERGE):
    idx = np.where(mask)[0]
    if len(idx) == 0: return []
    return [(g[0], g[-1]) for g in np.split(idx, np.where(np.diff(idx) > gap + 1)[0] + 1)]


def contagens(y, alarme, eventos):
    al = runs(alarme)
    toca = lambda a0, a1: any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (e0, e1) in eventos)
    det = sum(any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (a0, a1) in al) for (e0, e1) in eventos)
    falsos = [(a0, a1) for (a0, a1) in al if not toca(a0, a1)]
    return np.array([det, len(eventos), sum((a1 - a0 + 1) * BLOCO_SEG / 60.0 for a0, a1 in falsos), sum(a1 - a0 + 1 for a0, a1 in al), len(y), len(al), len(falsos)])


def cascata(oks, als):
    n = len(oks[0]); pos = np.arange(n); saida = als[0].copy()
    for prox in (1, 2):
        idx = np.where(oks[prox])[0]
        if len(idx) == 0: continue
        cond = (pos >= idx[0] + PORTAO) & oks[prox] & ~(saida | als[prox])
        onde = np.where(cond)[0]
        if len(onde) == 0: continue
        saida = np.where(pos >= onde[0], als[prox], saida)
    return saida


def fronteira(soma):
    horas = soma[:, 4] * BLOCO_SEG / 3600.0
    sens, fa = soma[:, 0] / np.maximum(soma[:, 1], 1), soma[:, 2] / horas
    ok = (soma[:, 3] / soma[:, 4]) <= DUTY_MAX
    if ok.sum() < 2: return np.full(len(ALVOS), np.nan)
    s, f = sens[ok], fa[ok]
    o = np.argsort(f, kind='stable'); s, f = s[o], f[o]
    m = np.concatenate([[True], s[1:] > np.maximum.accumulate(s)[:-1]]); s, f = s[m], f[m]
    return np.interp(ALVOS, f, s, left=np.nan, right=s[-1]) if len(s) >= 2 else np.full(len(ALVOS), np.nan)


def avaliar(frame, pref, direcao, grelhas):
    """Contagens por doente, política (so_W5, cascata) e limiar: (n_doentes, 2, T, 7)."""
    P = {W: PR[f'{pref}_{direcao}_{W}'] for W in JAN}
    y5, _ = alvo(frame, 5, direcao)
    docs = sorted(frame.subject_id.unique()); pos = {s: i for i, s in enumerate(docs)}
    grupos = {}
    for sid, g in frame.groupby('session_id', sort=False):
        ix = g.index.values
        grupos.setdefault(pos[g.subject_id.iloc[0]], []).append(([P[W][ix] for W in JAN], np.nan_to_num(y5[ix], nan=0).astype(bool)))

    def um(i):
        acc = np.zeros((2, N_TAUS, 7))
        for ps, y in grupos[i]:
            oks = [~np.isnan(p) for p in ps]; ev = runs(y)
            for t in range(N_TAUS):
                als = [oks[k] & (np.nan_to_num(ps[k], nan=-np.inf) >= grelhas[W][t]) for k, W in enumerate(JAN)]
                acc[0, t] += contagens(y, als[2], ev); acc[1, t] += contagens(y, cascata(oks, als), ev)
        return i, acc
    out = np.zeros((len(docs), 2, N_TAUS, 7))
    for i, acc in Parallel(n_jobs=NT)(delayed(um)(i) for i in sorted(grupos)): out[i] = acc
    return out


rng = np.random.default_rng(0); R = {}
dev = dev.reset_index(drop=True); hold = hold.reset_index(drop=True)
for direcao in ('descida', 'subida'):
    grelhas = {W: np.quantile(PR[f'dev_{direcao}_{W}'][~np.isnan(PR[f'dev_{direcao}_{W}'])], np.linspace(0.55, 0.9995, N_TAUS)) for W in JAN}
    for nome, frame, pref in (('desenvolvimento', dev, 'dev'), ('teste', hold, 'hold')):
        C = avaliar(frame, pref, direcao, grelhas)
        pto = [fronteira(C.sum(0)[k]) for k in (0, 1)]
        dif = []
        for _ in range(200):
            b = C[rng.integers(0, len(C), len(C))].sum(0)
            dif.append(fronteira(b[1]) - fronteira(b[0]))
        dif = np.array(dif)
        R[f'{direcao}_{nome}'] = dict(alvos_min_h=ALVOS.tolist(), so_W5=pto[0].tolist(), cascata=pto[1].tolist(), diferenca=(pto[1] - pto[0]).tolist(),
                                      ic95=[np.nanpercentile(dif, 2.5, axis=0).tolist(), np.nanpercentile(dif, 97.5, axis=0).tolist()],
                                      diferenca_media=float(np.nanmean(pto[1] - pto[0])))
        print(direcao, nome, 'cascata − só_W5:', np.round(pto[1] - pto[0], 4), 'média', round(float(np.nanmean(pto[1] - pto[0])), 4), flush=True)
json.dump(R, open(f"{AQUI}/cascata_tempo.json", 'w'), indent=1)
print('FIM')
