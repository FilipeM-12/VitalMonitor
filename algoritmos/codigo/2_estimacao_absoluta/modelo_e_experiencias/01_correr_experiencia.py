#!/usr/bin/env python3
"""

Uma experiência do absoluto: busca de hiperparâmetros (3 dobras, dados causais) e avaliação única no holdout.

    python 01_correr_experiencia.py FAMILIA VARIANTE [--trials 20] [--paralelo 3] [--ancoras causal]

FAMILIA   XGB | BILSTM | BITCN | HIBRIDO_BILSTM_XGB | HIBRIDO_BITCN_XGB
VARIANTE  baseline | nk | nk2 | fft | clinicas | diagnostico   (ver nucleo/config.py)

Escreve em resultados/experiencias/<FAMILIA>_<VARIANTE>/
    metricas.json          MAE de CV e de holdout, coorte, hiperparâmetros, tempos
    holdout_predicoes.csv  uma linha por bloco previsto do holdout (usada pelas tabelas, BHS, SHAP e deriva)
    trials.csv             MAE de CV de cada tentativa do Optuna
Se metricas.json já existe, não faz nada. O estudo Optuna retoma do último trial gravado.
"""

# Operações deterministas do TensorFlow: a mesma semente dá o mesmo resultado. Sem isto, o híbrido BiLSTM-XGB varia até 0,3 mmHg
# entre corridas com a mesma semente (ver 14_reprodutibilidade.py). Os processos dos trials herdam estas variáveis.
import os as _os
_os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
_os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nucleo import config as C
from nucleo.avaliacao import avaliar_holdout
from nucleo.busca import procurar
from nucleo.dados import carregar, carregar_tabela

COLUNAS_PREVISOES = ['subject_id', 'record_id', 'segment_order', 'bloco_id', 'session_id', 'sbp_target', 'sbp_calib',
                     'preds_delta', 'preds_final', 'tempo_desde_calib_seg', 'age', 'pat_mean', 'hr_mean']


def correr(familia, variante, n_trials=C.N_TRIALS, paralelo=1, ancoras="causal"):
    pasta = f"{C.DIR_RESULTADOS}/experiencias/{familia}_{variante}" + ("" if ancoras == "causal" else f"_{ancoras}")
    if os.path.exists(f"{pasta}/metricas.json"):
        print(f"{familia}_{variante}: já feito", flush=True)
        return json.load(open(f"{pasta}/metricas.json"))
    os.makedirs(pasta, exist_ok=True)
    features = C.VARIANTES[variante].features

    # derivar a tabela UMA vez, num subprocesso, antes dos trials (que só leem a cache). Sem isto cada trial em paralelo
    # derivava a sua cópia ao mesmo tempo e a memória esgotava-se.
    ctx = mp.get_context('spawn')
    p = ctx.Process(target=carregar_tabela, args=(C.VARIANTES[variante], ancoras))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise SystemExit(f"a derivação de {variante} falhou (código {p.exitcode})")

    t0 = time.time()
    melhores, tentativas = procurar(familia, variante, ancoras, features, n_trials, trials_em_paralelo=paralelo)
    t_busca = time.time() - t0

    dev, teste, info = carregar(variante, ancoras)
    t0 = time.time()
    previsoes, m = avaliar_holdout(familia, melhores, dev, teste, features)
    t_holdout = time.time() - t0
    print(f"[{familia}_{variante}] HOLDOUT MAE = {m['mae_sbp']:.4f} mmHg (n={m['n']}, sujeitos={m['n_sujeitos']})", flush=True)

    previsoes[[c for c in COLUNAS_PREVISOES if c in previsoes.columns]].to_csv(f"{pasta}/holdout_predicoes.csv", index=False)
    pd.DataFrame([(t.number, t.value) for t in tentativas], columns=["number", "value"]).to_csv(f"{pasta}/trials.csv", index=False)
    saida = dict(familia=familia, variante=variante, ancoras=ancoras, n_trials=len(tentativas), mae_cv=float(json.load(open(f"{C.DIR_OPTUNA}/{familia}_{variante}" + ("" if ancoras == "causal" else f"_{ancoras}") + ".best_params.json"))["mae_cv"]),
                 mae_holdout=m['mae_sbp'], mae_delta_holdout=m['mae_delta'], n_holdout=m['n'], n_sujeitos_holdout=m['n_sujeitos'],
                 coorte=info, n_features=len(features), hiperparametros=melhores,
                 tempo_busca_s=round(t_busca, 1), tempo_holdout_s=round(t_holdout, 1))
    json.dump(saida, open(f"{pasta}/metricas.json", "w"), indent=1)
    return saida


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("familia", choices=["XGB", "BILSTM", "BITCN", "HIBRIDO_BILSTM_XGB", "HIBRIDO_BITCN_XGB"])
    ap.add_argument("variante", choices=sorted(C.VARIANTES))
    ap.add_argument("--trials", type=int, default=C.N_TRIALS)
    ap.add_argument("--paralelo", type=int, default=1, help="trials em paralelo (só para o XGBoost)")
    ap.add_argument("--ancoras", default="causal", choices=["causal", "doente_inteiro"])
    a = ap.parse_args()
    correr(a.familia, a.variante, a.trials, a.paralelo, a.ancoras)
