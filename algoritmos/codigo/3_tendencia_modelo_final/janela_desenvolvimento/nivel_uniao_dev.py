"""nivel_uniao_dev.py — tabela dos níveis do quantil (e do classificador) com o custo da UNIÃO dos dois detetores (substitui a leitura da tabela de nivel_quantil_dev.py, que mede cada detetor sozinho).

Para cada formulação, previsões fora da dobra no desenvolvimento (3 dobras por doente, as mesmas de nivel_quantil_dev.py), com os hiperparâmetros do modelo adotado (os do classificador para o classificador):
  · classificador (score = probabilidade);
  · regressor quantílico com tau = e na descida e 1 − e na subida, para e = 0,05; 0,10; 0,20; 0,30 e 0,50 (e = 0,10 é o modelo adotado; hiperparâmetros afinados com tau = 0,10/0,90, sem reafinar por nível).
Ponto de operação como no notebook 03 (funções extraídas dele): grelha de 30 x 30 pares de limiares entre os percentis 90 e 99,8 de cada score (a de 02_classificador_vs_quantilica.py), e escolha do par que maximiza a menor das duas sensibilidades com a união
a no máximo 10 min/h de falso alarme e ciclo de serviço <= 30 %. Escreve resultados_nivel_uniao_dev.csv (AUC por bloco, sensibilidade de cada direção e VPP da união). Previsões em cache (previsoes_nivel_uniao.npz).
Uso: python3 nivel_uniao_dev.py [pasta_dados] [n_threads]
"""
import ast, json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier, XGBRegressor

AQUI = os.path.dirname(os.path.abspath(__file__))
A = "/home/filipe/Desktop/Tese/algoritmo-ml"
TF = os.path.normpath(os.path.join(AQUI, "..", "tendencia_tempo_fixo"))        # tempo_fixo.py e os best_params_*.json só existem em tendencia_tempo_fixo/
sys.path.insert(0, TF)
import tempo_fixo as tf  # noqa: E402

DADOS = sys.argv[1] if len(sys.argv) > 1 else f"{AQUI}/dados"
NJ = int(sys.argv[2]) if len(sys.argv) > 2 else 14
D, BLOCO_SEG, GAP_MERGE, PRE_CREDITO, DUTY_MAX = 15, 15, 4, 8, 0.30
DIRS = ('descida', 'subida'); SINAL = {'descida': -1.0, 'subida': 1.0}
NB = f"{A}/ENTREGA_novo/03_tendencia_regressao_quantilica.ipynb"
NS = dict(np=np, pd=pd, DIRS=DIRS, BLOCO_SEG=BLOCO_SEG, GAP_MERGE=GAP_MERGE, PRE_CREDITO=PRE_CREDITO, DUTY_MAX=DUTY_MAX)
cells = json.load(open(NB))["cells"]
for i in (17, 19):
    for no in ast.parse("".join(cells[i]["source"])).body:
        if isinstance(no, ast.FunctionDef) or (isinstance(no, ast.Assign) and getattr(no.targets[0], "id", "") in ("SEP",)):
            exec(compile(ast.Module([no], []), NB, "exec"), NS)
varrer, concatenar = NS["varrer"], NS["concatenar"]

dev, V = tf.preparar_grade(DADOS, 'dev'); FEATS = V['tudo']
ok = dev.delta.notna().values
dv = dev[ok]; X = dv[FEATS].values; suj = dv['subject_id'].astype(str).values
Y = {d: dv[f'y_{d}'].values.astype(float) for d in DIRS}
folds = {d: list(GroupKFold(n_splits=3).split(X, Y[d], suj)) for d in DIRS}
HPQ = {d: json.load(open(f"{A}/Tentativa_22_10/tendencia_tempo_fixo/best_params_quant_{d}_tudo.json"))["best_params"] for d in DIRS}
HPC = {d: json.load(open(f"{A}/Tentativa_22_10/tendencia_tempo_fixo/best_params_clf_{d}_tudo.json"))["best_params"] for d in DIRS}
NIVEIS = (0.05, 0.10, 0.20, 0.30, 0.50)
cache = f"{AQUI}/previsoes_nivel_uniao.npz"
PR = dict(np.load(cache)) if os.path.exists(cache) else {}


def oof_clf(d):
    yb = Y[d]; pw = (len(yb) - yb.sum()) / max(yb.sum(), 1); o = np.zeros(len(yb))
    for tr, va in folds[d]:
        o[va] = XGBClassifier(**HPC[d], scale_pos_weight=pw, eval_metric='auc', n_jobs=NJ, tree_method='hist').fit(X[tr], yb[tr]).predict_proba(X[va])[:, 1]
    return o


def oof_q(d, tau):
    yr = dv['delta'].values; o = np.zeros(len(yr))
    for tr, va in folds[d]:
        o[va] = SINAL[d] * XGBRegressor(**HPQ[d], objective='reg:quantileerror', quantile_alpha=tau, n_jobs=NJ, tree_method='hist').fit(X[tr], yr[tr]).predict(X[va])
    return o


for d in DIRS:
    if f'clf_{d}' not in PR:
        t0 = time.time(); PR[f'clf_{d}'] = oof_clf(d); print('clf', d, f'{time.time() - t0:.0f} s', flush=True); np.savez_compressed(cache, **PR)
    for e in NIVEIS:
        k = f'q{e:.2f}_{d}'
        if k not in PR:
            t0 = time.time(); PR[k] = oof_q(d, e if d == 'descida' else 1 - e); print(k, f'{time.time() - t0:.0f} s', flush=True); np.savez_compressed(cache, **PR)

linhas = []
base = dev[['session_id', 'bloco_id', 'y_descida', 'y_subida']].copy()
for d in DIRS: base[f'y_{d}'] = base[f'y_{d}'].fillna(0)
for nome, ks in [('classificador', {d: f'clf_{d}' for d in DIRS})] + [(f'quantílica e={e:.2f}', {d: f'q{e:.2f}_{d}' for d in DIRS}) for e in NIVEIS]:
    x = base.copy()
    for d in DIRS:
        p = np.full(len(dev), np.nan); p[ok] = PR[ks[d]]; x[f'p_{d}'] = p
    cat = concatenar(x, ['p_descida', 'p_subida', 'y_descida', 'y_subida'])
    taus = {d: np.unique(np.round(np.nanpercentile(x[f'p_{d}'], np.linspace(90, 99.8, 30)), 4)) for d in DIRS}     # a grelha de 02_classificador_vs_quantilica.py (a da Tabela da comparação)
    graf, M, ev, cs_any, n_real = varrer(cat, taus)
    c = graf[(graf.u_FA_min_h <= 10) & (graf.u_duty <= DUTY_MAX)]
    r = c.loc[np.minimum(c.sens_d, c.sens_s).idxmax()]
    al = NS["carga_uniao"] if "carga_uniao" in NS else None
    row = dict(formulacao=nome, auc_descida=roc_auc_score(Y['descida'], PR[ks['descida']]), auc_subida=roc_auc_score(Y['subida'], PR[ks['subida']]),
               sens_descida=r.sens_d, sens_subida=r.sens_s, fa_min_h=r.u_FA_min_h, duty=r.u_duty)
    linhas.append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(linhas).to_csv(f"{AQUI}/resultados_nivel_uniao_dev.csv", index=False)
print("FIM")
