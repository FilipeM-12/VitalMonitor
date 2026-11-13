"""
Verificação de robustez da comparação XGBoost contra híbridos na formulação de
TENDÊNCIA, com validação cruzada de 3 dobras por sujeito sobre TODOS os sujeitos
(desenvolvimento + holdout), 3 sementes por modelo.

Substitui cv5_sujeitos_hibridos_vs_xgboost.py e cv5_sementes_xgboost.py (5
dobras, que não estão neste repositório). As 3 dobras são as mesmas do resto da tese, para
os métodos serem uniformes.

Corre para uma das duas formulações (argumento obrigatório):
  classificador  hiperparâmetros de resumo_arquiteturas.csv (XGBoost) e
                 cache_hibrido_conjunto/*.json (híbridos)
  quantilica     hiperparâmetros de cache_hibrido_quantilico/*.json (os três)

Em cada dobra treina o XGBoost e os dois híbridos (BiLSTM+XGB, BiTCN+XGB) com
os hiperparâmetros JÁ afinados, cada um com as sementes 0, 1 e 2. O híbrido da
semente s compara-se com o XGBoost da mesma semente (cada semente é UM modelo,
nunca se faz média entre sementes). Análise: AUC por dobra e agregado fora da
dobra, e bootstrap de sujeitos (1000) da diferença emparelhada.

Ressalva: os hiperparâmetros foram afinados no desenvolvimento, que aqui entra em
dobras de teste. Isso favorece todos os modelos por igual, pelo que a diferença
emparelhada se mantém válida, mas os AUC absolutos são otimistas.

Retoma sozinho: cada (dobra, modelo, semente) fica em cache_cv3_<formulacao>/.
Sai com código 75 quando a memória passa o limite, para o lançador reiniciar.
Saída: cv3_<formulacao>_hibridos_vs_xgboost.json (nesta pasta).
"""
import json
import os
import sys

ORIG = "/home/filipe/Desktop/Tese/algoritmo-ml/2026-09-13_modelo_temporal_causal"
SAIDA = os.path.dirname(os.path.abspath(__file__))
os.chdir(ORIG)
sys.path.insert(0, ORIG)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from tensorflow.keras.models import Model
from xgboost import XGBClassifier

import comparacao_arquiteturas_classificador as base
import hibrido_classificador_busca_conjunta as hc
import hibrido_quantilico_busca_conjunta as hq

K, SEMENTES, N_BOOT = 3, (0, 1, 2), 1000
LIMITE_RSS_GB = 15
FORMULACAO = sys.argv[1] if len(sys.argv) > 1 else ''
assert FORMULACAO in ('classificador', 'quantilica'), "uso: cv3_hibridos_vs_xgboost.py classificador|quantilica"
CACHE = f"{SAIDA}/cache_cv3_{FORMULACAO}"
os.makedirs(CACHE, exist_ok=True)


def sair_se_memoria_alta():
    with open("/proc/self/status") as f:
        rss = next(int(l.split()[1]) for l in f if l.startswith("VmRSS")) / 1024 / 1024
    if rss > LIMITE_RSS_GB:
        print(f"RSS={rss:.1f}GB > {LIMITE_RSS_GB}GB: a reiniciar o processo", flush=True)
        os._exit(75)


def dados_todos(dr):
    """Junta desenvolvimento e holdout. Na quantílica o alvo de regressão é o Delta."""
    if FORMULACAO == 'quantilica':
        d = hq.preparar(dr)
        subj_h, alvo = d['subj_h'], np.concatenate([d['dl_d'], d['dl_h']])
    else:
        d = hc.preparar(dr)
        hold, _ = base.carregar(f"{base.RAIZ_DADOS}/holdout_ppg.csv", dr)
        _, _, y2, subj_h, _, _ = base.construir_sequencias(hold, d['feat_tab'])
        assert len(subj_h) == len(d['y_h']) and (y2 == d['y_h']).all()
        alvo = None
    Xs = np.concatenate([d['Xs_d_n'], d['Xs_h_n']])
    Xt = np.concatenate([d['Xt_d'], d['Xt_h']])
    y = np.concatenate([d['y_d'], d['y_h']])
    subj = np.concatenate([d['subj_d'], subj_h])
    assert not (set(d['subj_d']) & set(subj_h)), "desenvolvimento e holdout partilham sujeitos"
    return Xs, Xt, y, alvo, subj, d['F']


def hiperparametros(dr):
    if FORMULACAO == 'quantilica':
        ler = lambda m: json.load(open(f"{hq.CACHE_DIR}/{m}_{dr}.json"))['best_params']
        return {m: ler(m) for m in ('XGB', 'BiLSTM', 'BiTCN')}
    resumo = pd.read_csv(f"{ORIG}/resumo_arquiteturas.csv")
    xgb = json.loads(resumo[(resumo.direcao == dr) & (resumo.modelo == 'XGB_tabular')].best_params.iloc[0])
    ler = lambda m: json.load(open(f"{hc.CACHE_DIR}/{m}_{dr}.json"))['best_params']
    return {'XGB': xgb, 'BiLSTM': ler('BiLSTM'), 'BiTCN': ler('BiTCN')}


def score_xgboost(hp, s, dr, Xt, y, alvo, tr, te):
    if FORMULACAO == 'quantilica':
        hq.SEMENTE = s
        m = hq.xgb_quantilico(hp, hq.TAU[dr]).fit(Xt[tr], alvo[tr])
        return hq.score_de(m.predict(Xt[te]), dr)
    ytr = y[tr]
    pw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    m = XGBClassifier(**hp, scale_pos_weight=pw, eval_metric='auc', n_jobs=-1, tree_method='hist',
                      random_state=s).fit(Xt[tr], ytr)
    return m.predict_proba(Xt[te])[:, 1]


def score_hibrido(nome, hp, s, dr, Xs, Xt, y, alvo, subj, F, tr, te):
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(Xs[tr], y[tr], subj[tr]))
    tr2, va2 = tr[a], tr[b]
    if FORMULACAO == 'quantilica':
        net, xgbp = hq.separar_params(nome, hp)
        hq.SEMENTE = s
        mod = hq.treinar_rede(nome, net, Xs[tr2], alvo[tr2], Xs[va2], alvo[va2], F, hq.TAU[dr])
    else:
        net, xgbp = hc.separar_params(hp)
        tf.keras.utils.set_random_seed(100 + s)
        mod = hc.treinar_rede(nome, net, Xs[tr2], y[tr2], Xs[va2], y[va2], F)
    ex = Model(mod.input, mod.get_layer('emb').output)
    a_tr = np.hstack([Xt[tr], ex.predict(Xs[tr], batch_size=1024, verbose=0)])
    a_te = np.hstack([Xt[te], ex.predict(Xs[te], batch_size=1024, verbose=0)])
    del mod, ex
    tf.keras.backend.clear_session()
    if FORMULACAO == 'quantilica':
        m = hq.xgb_quantilico(xgbp, hq.TAU[dr]).fit(a_tr, alvo[tr])
        return hq.score_de(m.predict(a_te), dr)
    return hc.xgb_topo(xgbp, y[tr]).fit(a_tr, y[tr]).predict_proba(a_te)[:, 1]


def bootstrap_emparelhado(y, subj, a, b):
    rng = np.random.default_rng(42)
    us = np.unique(subj)
    idx = {u: np.where(subj == u)[0] for u in us}
    difs = []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        if y[ii].min() == y[ii].max():
            continue
        difs.append(roc_auc_score(y[ii], a[ii]) - roc_auc_score(y[ii], b[ii]))
    return [float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))], float((np.array(difs) > 0).mean())


def main():
    res = {}
    for dr in ('descida', 'subida'):
        Xs, Xt, y, alvo, subj, F = dados_todos(dr)
        hp = hiperparametros(dr)
        print(f"[{FORMULACAO}/{dr}] {len(y)} janelas, {len(np.unique(subj))} sujeitos, {int(y.sum())} positivos",
              flush=True)
        oof, auc_dobra = {}, {}
        for k, (tr, te) in enumerate(GroupKFold(n_splits=K).split(Xs, y, subj)):
            for s in SEMENTES:
                for nome in ('XGB', 'BiLSTM', 'BiTCN'):
                    chave = f"XGBoost s{s}" if nome == 'XGB' else f"{nome}+XGB s{s}"
                    cp = f"{CACHE}/{dr}_dobra{k}_{chave.replace(' ', '_').replace('+', '')}.npy"
                    if os.path.exists(cp):
                        p = np.load(cp)
                    else:
                        p = (score_xgboost(hp['XGB'], s, dr, Xt, y, alvo, tr, te) if nome == 'XGB' else
                             score_hibrido(nome, hp[nome], s, dr, Xs, Xt, y, alvo, subj, F, tr, te))
                        np.save(cp, p)
                        print(f"[{FORMULACAO}/{dr} dobra {k}] {chave} AUC {roc_auc_score(y[te], p):.4f}", flush=True)
                        sair_se_memoria_alta()
                    oof.setdefault(chave, np.full(len(y), np.nan))[te] = p
                    auc_dobra.setdefault(chave, []).append(float(roc_auc_score(y[te], p)))
        assert not any(np.isnan(v).any() for v in oof.values())

        agregado = {n: float(roc_auc_score(y, v)) for n, v in oof.items()}
        ganhos = {}
        for n in oof:
            if n.startswith('XGBoost'):
                continue
            ref = f"XGBoost s{n[-1]}"
            ic, p_pos = bootstrap_emparelhado(y, subj, oof[n], oof[ref])
            ganhos[n] = dict(ganho=agregado[n] - agregado[ref], ic95=ic, p_pos=p_pos,
                             ganho_por_dobra=[a - b for a, b in zip(auc_dobra[n], auc_dobra[ref])])
            print(f"[{FORMULACAO}/{dr}] {n}: ganho {ganhos[n]['ganho']:+.4f}  IC95 [{ic[0]:+.4f}, {ic[1]:+.4f}]",
                  flush=True)
        xgb = [agregado[f"XGBoost s{s}"] for s in SEMENTES]
        res[dr] = dict(formulacao=FORMULACAO, n=int(len(y)), n_sujeitos=int(len(np.unique(subj))),
                       n_positivos=int(y.sum()), k=K, sementes=list(SEMENTES),
                       auc_por_dobra=auc_dobra, auc_agregado=agregado, ganho_mesma_semente=ganhos,
                       amplitude_xgboost_entre_sementes=float(max(xgb) - min(xgb)))
        json.dump(res, open(f"{SAIDA}/cv3_{FORMULACAO}_hibridos_vs_xgboost.json", "w"), indent=2)
    print(f"\n{'='*74}\nRESULTADO FINAL (cv3, {FORMULACAO})\n{'='*74}", flush=True)


if __name__ == "__main__":
    main()
