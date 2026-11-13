"""
Sementes adicionais dos híbridos quantílicos, para saber se o ganho sobre o
XGBoost resiste à aleatoriedade do treino.

Reutiliza os hiperparâmetros escolhidos com a semente 0 em
hibrido_quantilico_busca_conjunta.py (cache_hibrido_quantilico/*.json) e repete
só o treino final, com as SEMENTES abaixo. O XGBoost quantílico é retreinado com
a mesma semente de cada híbrido, para a comparação ser emparelhada como no
classificador (cada semente é UM modelo, nunca se faz média entre sementes).

Saída: holdout_preds_hibrido_quantilico_sementes_{direcao}.csv.gz (scores por
semente) e resumo_hibrido_quantilico_sementes.csv (AUC e ganho emparelhado por
semente, com IC95 por reamostragem de sujeitos). A semente 0 vem das previsões
já guardadas, sem retreino.
"""
import gc
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import hibrido_quantilico_busca_conjunta as q

SEMENTES = (1, 2)
MODELOS = ('XGB', 'BiLSTM', 'BiTCN')
B = 1000


def ganho_bootstrap(y, s_hib, s_xgb, sujeitos, rng):
    grupos = {s: np.where(sujeitos == s)[0] for s in np.unique(sujeitos)}
    chaves = list(grupos)
    difs = []
    for _ in range(B):
        idx = np.concatenate([grupos[s] for s in rng.choice(chaves, len(chaves))])
        if y[idx].min() == y[idx].max():
            continue
        difs.append(roc_auc_score(y[idx], s_hib[idx]) - roc_auc_score(y[idx], s_xgb[idx]))
    return np.percentile(difs, [2.5, 97.5])


def main():
    linhas = []
    rng = np.random.default_rng(0)
    for direcao in ('descida', 'subida'):
        t0 = time.time()
        dados = q.preparar(direcao)
        print(f"[{direcao}] dados prontos em {time.time()-t0:.0f}s", flush=True)
        base = pd.read_csv(q.PREDS.format(direcao=direcao))
        assert len(base) == len(dados['y_h']) and (base['y'].values == dados['y_h']).all()
        preds = base[['subject_id', 'session_id', 'bloco_id', 'y', 'delta']].copy()
        for m in MODELOS:
            preds[f'score_{m}_s0'] = base[f'score_{m}']

        for semente in SEMENTES:
            q.SEMENTE = semente
            for m in MODELOS:
                best = json.load(open(f"{q.CACHE_DIR}/{m}_{direcao}.json"))['best_params']
                t0 = time.time()
                auc, s = q.avaliar_final(m, best, dados, direcao)
                preds[f'score_{m}_s{semente}'] = s
                print(f"[{direcao}] semente {semente} {m}: AUC {auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
                gc.collect()
            preds.to_csv(f"{q.OUT}/holdout_preds_hibrido_quantilico_sementes_{direcao}.csv.gz",
                         index=False, compression='gzip')

        y, suj = preds['y'].values, preds['subject_id'].values
        for semente in (0,) + SEMENTES:
            s_xgb = preds[f'score_XGB_s{semente}'].values
            auc_xgb = roc_auc_score(y, s_xgb)
            for m in ('BiLSTM', 'BiTCN'):
                s_hib = preds[f'score_{m}_s{semente}'].values
                auc_hib = roc_auc_score(y, s_hib)
                lo, hi = ganho_bootstrap(y, s_hib, s_xgb, suj, rng)
                linhas.append(dict(direcao=direcao, semente=semente, modelo=f'{m}+XGB',
                                   auc_xgb=round(auc_xgb, 4), auc_hibrido=round(auc_hib, 4),
                                   ganho=round(auc_hib - auc_xgb, 4),
                                   ic95_inf=round(lo, 4), ic95_sup=round(hi, 4),
                                   exclui_zero=bool(lo > 0 or hi < 0)))
                print(linhas[-1], flush=True)
        pd.DataFrame(linhas).to_csv(f"{q.OUT}/resumo_hibrido_quantilico_sementes.csv", index=False)
        del dados
        gc.collect()

    print(f"\n{'='*74}\nRESULTADO FINAL (sementes {(0,) + SEMENTES}, hiperparametros da semente 0)\n{'='*74}")
    print(pd.DataFrame(linhas).to_string(index=False))


if __name__ == "__main__":
    main()
