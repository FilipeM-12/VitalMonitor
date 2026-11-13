"""
Comparação do SISTEMA COMPLETO (score + portão de estabilidade + limiar), XGBoost
contra híbridos, na regressão quantílica. Por direção, para os três modelos.

Porquê: a sensibilidade a custo igualado de sensibilidade_custo_igualado_quantilica.py
foi medida SEM portão, porque o portão afina-se com previsões fora de partição
(OOF) no desenvolvimento, que não existiam para os híbridos. Aqui produzem-se.

PROCEDIMENTO (por direção, modelo e semente; o mesmo para os três modelos)
  1. Previsões OOF no desenvolvimento, validação cruzada de 3 dobras por sujeito
     (a tese usa 5 para o XGBoost sozinho, mas 3 é o esquema das comparações com
     redes), com os hiperparâmetros já afinados em cache_hibrido_quantilico/.
  2. Grelha de 80 limiares entre os percentis 1 e 99 dessas previsões.
  3. Portão afinado no DESENVOLVIMENTO: n em (4, 8, 12, 20) blocos e limite de
     amplitude entre os percentis (50, 75, 90, 95, 99) de |diff| do score, e
     escolhe-se o par com maior ganho médio, a custo igualado, sobre a fronteira
     sem portão.
  4. Limiar escolhido no desenvolvimento, com o portão ativo: a maior
     sensibilidade com ciclo de serviço <= 30% e falso alarme <= 10 min/h.
  5. Medição única no holdout (44 sujeitos): fronteira com portão, sensibilidade
     interpolada a 4 a 10 min/h, e o ponto de operação escolhido em 4.
  6. Ganho emparelhado do híbrido sobre o XGBoost da MESMA semente, IC95 por 1000
     reamostragens de sujeitos do holdout, com portão e limiares FIXOS (não se
     voltam a escolher em cada reamostragem).

DIFERENÇAS FACE AO MODELO EM PRODUÇÃO, ditas no texto
  - cada direção é avaliada sozinha, e não a carga da UNIÃO das duas
    (04_otimizacao_conjunta_causal.py)
  - as linhas são as das sequências de 20 blocos, as mesmas dos três modelos
  - OOF a 3 dobras em vez de 5

A avaliação reproduz a semântica de `avaliar` (02_avaliar.py), mas vetorizada
sobre a série concatenada com separadores entre sessões (como em
04_otimizacao_conjunta_causal.py), porque o ciclo por sessão levaria dezenas de
horas. `validar_avaliacao` compara as duas num subconjunto antes de correr.

Retoma sozinho (cache por direção, modelo, semente e dobra em cache_sistema_completo/).
Sai com código 75 quando a memória passa o limite, para o lançador reiniciar.
Saída: sistema_completo_quantilica.{csv,json}
"""
import gc
import json
import os
import sys
import time

ORIG = "/home/filipe/Desktop/Tese/algoritmo-ml/2026-09-13_modelo_temporal_causal"
SAIDA = os.path.dirname(os.path.abspath(__file__))
os.chdir(ORIG)
sys.path.insert(0, ORIG)

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from tensorflow.keras.models import Model

import comparacao_arquiteturas_classificador as base
import hibrido_quantilico_busca_conjunta as hq

BLOCO_SEG = 15
GAP_MERGE = int(1 * 60 / BLOCO_SEG)
PRE_CREDITO = int(2 * 60 / BLOCO_SEG)
SEP = max(GAP_MERGE, PRE_CREDITO) + 2
DUTY_MAX, FA_ALVO = 0.30, 10.0
ALVOS_FA = np.array([4, 5, 6, 7, 8, 9, 10])
N_LIMIARES, N_GATE, B = 80, (4, 8, 12, 20), 1000
SEMENTES, MODELOS = (0, 1, 2), ('XGB', 'BiLSTM', 'BiTCN')
LIMITE_RSS_GB = 15
CACHE = f"{SAIDA}/cache_sistema_completo"


# ───────────────────────── avaliação vetorizada ─────────────────────────
def runs_np(mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.empty((0, 2), int)
    cortes = np.flatnonzero(np.diff(idx) > GAP_MERGE + 1)
    return np.column_stack([np.r_[idx[0], idx[cortes + 1]], np.r_[idx[cortes], idx[-1]]])


class Estrutura:
    """Série concatenada de sessões contíguas, com SEP blocos falsos entre elas."""

    def __init__(self, sess, y, subj_idx=None):
        n = len(y)
        muda = np.r_[True, sess[1:] != sess[:-1]]
        self.n = n
        self.inicio_sessao = np.flatnonzero(muda)
        cod = np.cumsum(muda) - 1
        assert len(np.unique(sess)) == len(self.inicio_sessao), "sessões não contíguas"
        self.k = np.arange(n) - self.inicio_sessao[cod]            # posição dentro da sessão
        self.pos = np.arange(n) + SEP * cod
        self.M = n + SEP * len(self.inicio_sessao)
        y_conc = np.zeros(self.M, bool)
        y_conc[self.pos] = y.astype(bool)
        self.ev = runs_np(y_conc)
        m = np.zeros(self.M + 1, int)
        ini = np.maximum(0, self.ev[:, 0] - PRE_CREDITO)
        np.add.at(m, ini, 1)
        np.add.at(m, self.ev[:, 1] + 1, -1)
        self.cs_ev = np.concatenate([[0], np.cumsum(np.cumsum(m)[:-1] > 0)])
        self.ev_ini = ini
        self.subj_conc = None
        if subj_idx is not None:
            self.subj_conc = np.zeros(self.M, int)
            self.subj_conc[self.pos] = subj_idx
            self.n_subj = int(subj_idx.max()) + 1
            self.ev_subj = self.subj_conc[self.ev[:, 0]]
            self.nev_subj = np.bincount(self.ev_subj, minlength=self.n_subj)
            self.blocos_subj = np.bincount(subj_idx, minlength=self.n_subj)
        self.horas = n * BLOCO_SEG / 3600.0

    def avaliar(self, alarme, por_sujeito=False):
        """alarme: bool (n,). Reproduz `avaliar` de 02_avaliar.py."""
        A = np.zeros(self.M, bool)
        A[self.pos] = alarme
        al = runs_np(A)
        if al.size == 0:
            r = dict(sens=0.0, FA_h=0.0, FA_min_h=0.0, duty=0.0)
            if por_sujeito:
                z = np.zeros(self.n_subj)
                r.update(det_s=z, fa_blocos_s=z, alarme_blocos_s=z)
            return r
        d = np.bincount(al[:, 0], minlength=self.M + 1) - np.bincount(al[:, 1] + 1, minlength=self.M + 1)
        cm = np.concatenate([[0], np.cumsum(np.cumsum(d)[:-1] > 0)])
        det_ev = (cm[self.ev[:, 1] + 1] - cm[self.ev_ini]) > 0
        toca = (self.cs_ev[al[:, 1] + 1] - self.cs_ev[al[:, 0]]) > 0
        comp = al[:, 1] - al[:, 0] + 1
        fa = ~toca
        r = dict(sens=float(det_ev.sum() / max(len(self.ev), 1)),
                 FA_h=float(fa.sum() / self.horas),
                 FA_min_h=float(comp[fa].sum() * BLOCO_SEG / 60.0 / self.horas),
                 duty=float(comp.sum() / self.n))
        if por_sujeito:
            r['det_s'] = np.bincount(self.ev_subj[det_ev], minlength=self.n_subj).astype(float)
            sa = self.subj_conc[al[:, 0]]
            r['fa_blocos_s'] = np.bincount(sa[fa], weights=comp[fa], minlength=self.n_subj)
            r['alarme_blocos_s'] = np.bincount(sa, weights=comp, minlength=self.n_subj).astype(float)
        return r


def amplitudes(p, st, n):
    """Amplitude causal do score nos n blocos ANTERIORES; NaN nos primeiros n da sessão."""
    amp = np.full(len(p), np.nan)
    if len(p) > n:
        w = sliding_window_view(p, n)                      # w[j] = p[j:j+n]
        a = w.max(1) - w.min(1)
        amp[n:] = a[:len(p) - n]
    amp[st.k < n] = np.nan
    return amp


def portao(amp, lim):
    return ~np.isfinite(amp) | (amp <= lim)


def fronteira_interp(sens, fa, duty):
    ok = duty <= DUTY_MAX
    if ok.sum() < 2:
        return np.full(len(ALVOS_FA), np.nan)
    s, f = sens[ok], fa[ok]
    o = np.argsort(f, kind='stable')
    s, f = s[o], f[o]
    mx = np.maximum.accumulate(s)
    mantem = np.r_[True, s[1:] > mx[:-1]]
    s, f = s[mantem], f[mantem]
    if len(s) < 2:
        return np.full(len(ALVOS_FA), np.nan)
    return np.interp(ALVOS_FA, f, s, left=np.nan, right=s[-1])


def varrer(p, st, taus, ok=None, por_sujeito=False):
    ok = np.ones(len(p), bool) if ok is None else ok
    return [dict(limiar=float(t), **st.avaliar((p >= t) & ok, por_sujeito)) for t in taus]


def frontier_de(linhas):
    sens = np.array([r['sens'] for r in linhas]); fa = np.array([r['FA_min_h'] for r in linhas])
    duty = np.array([r['duty'] for r in linhas])
    return fronteira_interp(sens, fa, duty)


def afinar_no_dev(p_dev, st_dev):
    """Portão e limiar escolhidos no desenvolvimento, como em 02_avaliar.py."""
    lo, hi = np.percentile(p_dev, [1, 99])
    taus = np.round(np.linspace(lo, hi, N_LIMIARES), 3)
    ref = frontier_de(varrer(p_dev, st_dev, taus))
    amps = np.round(np.percentile(np.abs(np.diff(p_dev)), [50, 75, 90, 95, 99]), 2)
    melhor, melhor_g, tabela = None, -9, []
    for n in N_GATE:
        a = amplitudes(p_dev, st_dev, n)
        for lim in amps:
            g = frontier_de(varrer(p_dev, st_dev, taus, portao(a, lim))) - ref
            g = g[np.isfinite(g)]
            if len(g) >= 4:
                tabela.append((n, float(lim), float(np.mean(g))))
                if np.mean(g) > melhor_g:
                    melhor_g, melhor = float(np.mean(g)), (int(n), float(lim))
    n, lim = melhor
    ok = portao(amplitudes(p_dev, st_dev, n), lim)
    d = varrer(p_dev, st_dev, taus, ok)
    cand = [r for r in d if r['duty'] <= DUTY_MAX and r['FA_min_h'] <= FA_ALVO]
    e = max(cand, key=lambda r: r['sens'])
    return dict(taus=taus, n=n, lim=lim, ganho_dev=melhor_g, tau=e['limiar'], sens_dev=e['sens']), ref


# ───────────────────────── dados e treino ─────────────────────────
def sair_se_memoria_alta():
    with open("/proc/self/status") as f:
        rss = next(int(l.split()[1]) for l in f if l.startswith("VmRSS")) / 1024 / 1024
    if rss > LIMITE_RSS_GB:
        print(f"RSS={rss:.1f}GB > {LIMITE_RSS_GB}GB: a reiniciar o processo", flush=True)
        os._exit(75)


def dados_direcao(dr):
    d = hq.preparar(dr)
    dev, novos = hq.carregar(f"{base.RAIZ_DADOS}/dev_ppg.csv", dr)
    _, _, y2, _, subj2, sess_d, blo_d = hq.construir_sequencias(dev, base.FEAT_TAB + novos)
    del dev
    assert (y2 == d['y_d']).all() and (subj2 == d['subj_d']).all(), "desenvolvimento não alinha"
    d['sess_d'], d['blo_d'] = sess_d, blo_d
    return d


def hiperparametros(dr):
    return {m: json.load(open(f"{hq.CACHE_DIR}/{m}_{dr}.json"))['best_params'] for m in MODELOS}


def score_dobra(nome, hp, s, dr, d, tr, te):
    Xs, Xt, alvo, y, subj, F = d['Xs_d_n'], d['Xt_d'], d['dl_d'], d['y_d'], d['subj_d'], d['F']
    tau = hq.TAU[dr]
    hq.SEMENTE = s
    if nome == 'XGB':
        m = hq.xgb_quantilico(hp, tau).fit(Xt[tr], alvo[tr])
        return hq.score_de(m.predict(Xt[te]), dr)
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(Xs[tr], y[tr], subj[tr]))
    tr2, va2 = tr[a], tr[b]
    net, xgbp = hq.separar_params(nome, hp)
    mod = hq.treinar_rede(nome, net, Xs[tr2], alvo[tr2], Xs[va2], alvo[va2], F, tau)
    ex = Model(mod.input, mod.get_layer('emb').output)
    a_tr = np.hstack([Xt[tr], ex.predict(Xs[tr], batch_size=1024, verbose=0)])
    a_te = np.hstack([Xt[te], ex.predict(Xs[te], batch_size=1024, verbose=0)])
    del mod, ex
    tf.keras.backend.clear_session()
    m = hq.xgb_quantilico(xgbp, tau).fit(a_tr, alvo[tr])
    return hq.score_de(m.predict(a_te), dr)


def oof(dr, d, hp):
    """Previsões OOF no desenvolvimento, por modelo e semente (cache por dobra)."""
    saida = {}
    folds = list(GroupKFold(n_splits=3).split(d['Xt_d'], d['y_d'], d['subj_d']))
    for s in SEMENTES:
        for nome in MODELOS:
            p = np.full(len(d['y_d']), np.nan)
            for k, (tr, te) in enumerate(folds):
                cp = f"{CACHE}/{dr}_{nome}_s{s}_dobra{k}.npy"
                if os.path.exists(cp):
                    p[te] = np.load(cp)
                    continue
                t0 = time.time()
                sc = score_dobra(nome, hp[nome], s, dr, d, tr, te)
                np.save(cp, sc)
                p[te] = sc
                print(f"[{dr}] OOF {nome} s{s} dobra {k} ({time.time()-t0:.0f}s)", flush=True)
                gc.collect()
                sair_se_memoria_alta()
            saida[(nome, s)] = p
    return saida


# ───────────────────────── análise ─────────────────────────
def ic_ganho(st_h, contagens, ref, hib, rng):
    """IC95 do ganho médio (e por nível) do híbrido sobre o XGBoost, reamostrando sujeitos."""
    S = st_h.n_subj
    pesos = np.stack([np.bincount(rng.integers(0, S, S), minlength=S) for _ in range(B)]).astype(float)

    def curva(c, w):
        det = w @ c['det']; fa = w @ c['fa']; al = w @ c['al']
        nev = w @ st_h.nev_subj; bl = w @ st_h.blocos_subj
        return fronteira_interp(det / np.maximum(nev, 1), fa * BLOCO_SEG / 60.0 / (bl * BLOCO_SEG / 3600.0), al / bl)

    g = np.stack([curva(contagens[hib], w) - curva(contagens[ref], w) for w in pesos])
    medias = np.nanmean(g, axis=1)
    return (np.nanpercentile(medias, [2.5, 97.5]).tolist(),
            [np.nanpercentile(g[:, i], [2.5, 97.5]).tolist() for i in range(g.shape[1])])


def contagens_holdout(p, st_h, taus, ok):
    linhas = varrer(p, st_h, taus, ok, por_sujeito=True)
    return dict(det=np.stack([r['det_s'] for r in linhas], 1), fa=np.stack([r['fa_blocos_s'] for r in linhas], 1),
                al=np.stack([r['alarme_blocos_s'] for r in linhas], 1)), linhas


def analisar(dr, d, scores_dev, hold):
    st_dev = Estrutura(d['sess_d'], d['y_d'])
    suj = pd.factorize(hold['subject_id'])[0]
    st_h = Estrutura(hold['session_id'].values, hold['y'].values, suj)
    rng = np.random.default_rng(42)
    linhas = []
    for s in SEMENTES:
        cont, info, fr = {}, {}, {}
        for nome in MODELOS:
            t0 = time.time()
            p_dev = scores_dev[(nome, s)]
            cfg, ref_dev = afinar_no_dev(p_dev, st_dev)
            p_h = hold[f'score_{nome}_s{s}'].values
            ok = portao(amplitudes(p_h, st_h, cfg['n']), cfg['lim'])
            c, lin = contagens_holdout(p_h, st_h, cfg['taus'], ok)
            cont[nome] = {k: c[k] for k in ('det', 'fa', 'al')}
            fr[nome] = frontier_de(lin)
            pt = st_h.avaliar((p_h >= cfg['tau']) & ok)
            sem = frontier_de(varrer(p_h, st_h, cfg['taus']))
            info[nome] = dict(cfg=cfg, ponto=pt, sem_portao=sem)
            print(f"[{dr}] s{s} {nome}: portão {cfg['n']} blocos <= {cfg['lim']} (ganho dev {cfg['ganho_dev']:+.4f}), "
                  f"limiar {cfg['tau']:.2f} -> holdout sens {pt['sens']:.3f} FA {pt['FA_min_h']:.2f} min/h "
                  f"duty {pt['duty']*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)
        for nome in MODELOS:
            cfg, pt = info[nome]['cfg'], info[nome]['ponto']
            linha = dict(direcao=dr, semente=s, modelo=nome, portao_blocos=cfg['n'], portao_limite=cfg['lim'],
                         limiar=cfg['tau'], sens_dev=round(cfg['sens_dev'], 4), sens_holdout=round(pt['sens'], 4),
                         FA_min_h_holdout=round(pt['FA_min_h'], 3), duty_holdout=round(pt['duty'], 4),
                         sens_media_com_portao=round(float(np.nanmean(fr[nome])), 4),
                         sens_media_sem_portao=round(float(np.nanmean(info[nome]['sem_portao'])), 4),
                         **{f'sens_{a}minh': round(float(v), 4) for a, v in zip(ALVOS_FA, fr[nome])})
            if nome != 'XGB':
                ic, ic_niveis = ic_ganho(st_h, cont, 'XGB', nome, rng)
                g = fr[nome] - fr['XGB']
                linha.update(ganho_medio=round(float(np.nanmean(g)), 4), ic95_inf=round(ic[0], 4), ic95_sup=round(ic[1], 4),
                             exclui_zero=bool(ic[0] > 0 or ic[1] < 0),
                             ganho_ponto_op=round(pt['sens'] - info['XGB']['ponto']['sens'], 4),
                             **{f'ganho_{a}minh': round(float(v), 4) for a, v in zip(ALVOS_FA, g)})
            linhas.append(linha)
    return linhas


def principal():
    os.makedirs(CACHE, exist_ok=True)
    todas = []
    for dr in ('descida', 'subida'):
        t0 = time.time()
        d = dados_direcao(dr)
        hp = hiperparametros(dr)
        print(f"[{dr}] dados prontos em {time.time()-t0:.0f}s | dev {len(d['y_d'])} | holdout {len(d['y_h'])}", flush=True)
        scores_dev = oof(dr, d, hp)
        hold = pd.read_csv(f"{ORIG}/holdout_preds_hibrido_quantilico_sementes_{dr}.csv.gz")
        assert (hold['session_id'].astype(str).values == np.asarray(d['sess_h']).astype(str)).all() \
            and (hold['bloco_id'].values == d['blo_h']).all(), "holdout não alinha com as previsões guardadas"
        todas += analisar(dr, d, scores_dev, hold)
        pd.DataFrame(todas).to_csv(f"{SAIDA}/sistema_completo_quantilica.csv", index=False)
        del d, scores_dev
        gc.collect()
    tab = pd.DataFrame(todas)
    json.dump(dict(n_limiares=N_LIMIARES, bootstrap=B, alvos_min_h=ALVOS_FA.tolist()),
              open(f"{SAIDA}/sistema_completo_quantilica.json", "w"), indent=2)
    h = tab[tab.modelo != 'XGB']
    print(f"\n{'='*74}\nRESULTADO FINAL (sistema completo, quantilica)\n{'='*74}")
    print(h[['direcao', 'semente', 'modelo', 'sens_media_com_portao', 'ganho_medio', 'ic95_inf', 'ic95_sup',
             'exclui_zero', 'ganho_ponto_op']].to_string(index=False))


if __name__ == "__main__":
    principal()
