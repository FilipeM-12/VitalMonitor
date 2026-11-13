"""Verifica a afirmacao do Cap. 7: o ganho dos hibridos sobre o XGBoost e da ordem da
incerteza de amostragem. Reavalia XGBoost tabular e os dois hibridos (hiperparametros
da busca conjunta em cache_hibrido_conjunto/) no holdout causal so-PPG, com N_SEMENTES
treinos da rede por hibrido, e calcula por bootstrap de SUJEITOS (1000 reamostragens):
  - IC95% do AUC de cada modelo;
  - IC95% da DIFERENCA emparelhada hibrido - XGBoost.
Cada semente e avaliada como modelo unico (comparavel com o XGBoost unico), sem media entre sementes.
Saida: bootstrap_hibridos_vs_xgboost.json e .log (nesta pasta)."""
import json, os, sys
ORIG = "/home/filipe/Desktop/Tese/algoritmo-ml/2026-09-13_modelo_temporal_causal"
SAIDA = os.path.dirname(os.path.abspath(__file__))
os.chdir(ORIG); sys.path.insert(0, ORIG)
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model
from xgboost import XGBClassifier
import hibrido_classificador_busca_conjunta as hc
import comparacao_arquiteturas_classificador as base

N_SEMENTES, N_BOOT = 3, 1000


def auc_ic(y, subj, ps, nomes):
    rng = np.random.default_rng(42)
    us = np.unique(subj); idx = {s: np.where(subj == s)[0] for s in us}
    res = {n: [] for n in nomes}; dif = {n: [] for n in nomes if n != 'XGBoost'}
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[s] for s in rng.choice(us, len(us))])
        if y[ii].min() == y[ii].max(): continue
        a = {n: roc_auc_score(y[ii], ps[n][ii]) for n in nomes}
        for n in nomes: res[n].append(a[n])
        for n in dif: dif[n].append(a[n] - a['XGBoost'])
    ic = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return ({n: dict(auc=float(roc_auc_score(y, ps[n])), ic95=ic(res[n]), semiamplitude=float((ic(res[n])[1]-ic(res[n])[0])/2)) for n in nomes},
            {n: dict(ganho=float(roc_auc_score(y, ps[n]) - roc_auc_score(y, ps['XGBoost'])), ic95=ic(dif[n]),
                     p_pos=float((np.array(dif[n]) > 0).mean())) for n in dif})


def main():
    resumo = pd.read_csv(f"{ORIG}/resumo_arquiteturas.csv")
    out = {}
    for dr in ('descida', 'subida'):
        dados = hc.preparar(dr)
        y = dados['y_h']
        # preparar() nao devolve o sujeito de cada linha do holdout: reconstroi-se com as mesmas funcoes
        hold, _ = base.carregar(f"{base.RAIZ_DADOS}/holdout_ppg.csv", dr)
        _, _, y2, subj, _, _ = base.construir_sequencias(hold, dados['feat_tab'])
        assert len(subj) == len(y) and (y2 == y).all()
        bp = json.loads(resumo[(resumo.direcao == dr) & (resumo.modelo == 'XGB_tabular')].best_params.iloc[0])
        yd = dados['y_d']; pw = (len(yd) - yd.sum()) / max(yd.sum(), 1)
        x = XGBClassifier(**bp, scale_pos_weight=pw, eval_metric='auc', n_jobs=-1, tree_method='hist').fit(dados['Xt_d'], yd)
        ps = {'XGBoost': x.predict_proba(dados['Xt_h'])[:, 1]}
        por_semente = {}
        for nome in ('BiLSTM', 'BiTCN'):
            best = json.load(open(f"{hc.CACHE_DIR}/{nome}_{dr}.json"))['best_params']
            net, xgbp = hc.separar_params(best)
            sc, pl = [], []
            for s in range(N_SEMENTES):
                tf.keras.utils.set_random_seed(100 + s)
                mod = hc.treinar_rede(nome, net, dados['Xs_d_n'][dados['tr']], yd[dados['tr']],
                                      dados['Xs_d_n'][dados['va']], yd[dados['va']], dados['F'])
                ex = Model(mod.input, mod.get_layer('emb').output)
                e_d = ex.predict(dados['Xs_d_n'], batch_size=1024, verbose=0); e_h = ex.predict(dados['Xs_h_n'], batch_size=1024, verbose=0)
                m = hc.xgb_topo(xgbp, yd).fit(np.hstack([dados['Xt_d'], e_d]), yd)
                p = m.predict_proba(np.hstack([dados['Xt_h'], e_h]))[:, 1]
                sc.append(float(roc_auc_score(y, p))); pl.append(p)
                print(f"[{dr}] {nome} semente {s}: AUC holdout {sc[-1]:.4f}", flush=True)
            por_semente[nome] = sc
            for i, p in enumerate(pl): ps[f'{nome}+XGB s{i}'] = p   # cada semente e um modelo (sem media de sementes: seria um ensemble)
        modelos, difs = auc_ic(y, subj, ps, list(ps))
        out[dr] = dict(n_holdout=int(len(y)), n_sujeitos=int(len(np.unique(subj))), auc_por_semente=por_semente,
                       modelos=modelos, diferenca_vs_xgboost=difs)
        print(dr, json.dumps(out[dr], indent=1), flush=True)
        json.dump(out, open(f"{SAIDA}/bootstrap_hibridos_vs_xgboost.json", "w"), indent=2)


if __name__ == "__main__":
    main()
