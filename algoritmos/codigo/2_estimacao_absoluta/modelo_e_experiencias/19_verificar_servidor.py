#!/usr/bin/env python3
"""19 — O servidor calcula o mesmo que o treino, com as 27 características.

Corre depois do 18, que dá as medianas de reserva da normalização (comum.SD_RESERVA tem de ser igual).
Duas verificações independentes, porque características iguais não garantem previsões iguais.
  A. Derivação. comum.derivar_features_ml, o código que corre no worker, recebe janela a janela as janelas de extração
     do treino (os CSV de DIR_FEATURES) e cada bloco que devolve compara-se com a linha da tabela do treino
     (cache baseline, âncoras causais). Entram as sessões do conjunto de teste com um só segmento e sem blocos fora de
     70 < PAS < 190, porque o servidor não tem o alvo para aplicar esse filtro.
  B. Inferência. worker_absoluto.prever_ta, com o modelo_sbp.joblib instalado e no ambiente do servidor (o xgboost e o
     numpy de lá), recebe as linhas do teste pela ordem e compara-se com a reconstrução do treino
     (nucleo.avaliacao.reconstruir) sobre as previsões do mesmo pacote. O worker arredonda à décima de mmHg.
Escreve resultados/tabelas/verificacao_servidor.json.
  venv/bin/python 19_verificar_servidor.py
"""
import json
import os
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from nucleo import config as C
from nucleo import dados as D
from nucleo.avaliacao import metricas, reconstruir

SERVIDOR = "/home/filipe/Desktop/Tese/server"
PY_SERVIDOR = f"{SERVIDOR}/venv/bin/python"
PACOTE = f"{SERVIDOR}/modelo_sbp.joblib"
sys.path.insert(0, SERVIDOR)
import comum

FEATS = list(C.FEATURES_V3)
SAIDA = f"{C.DIR_RESULTADOS}/tabelas/verificacao_servidor.json"

assert comum.FEATURES_ML_ORDER == FEATS, "a ordem do servidor não é a do treino"
RESERVA = json.load(open(f"{C.DIR_RESULTADOS}/tabelas/sd_reserva_calibracao.json"))["reserva"]
assert all(comum.SD_RESERVA[k] == RESERVA[k] for k in comum.SD_RESERVA), "comum.SD_RESERVA não é a do 18"
assert set(comum.SD_RESERVA) == {'pat_mean', 'hr_mean', *comum.ANCORAS}
dev, teste, info = D.carregar("baseline")
del dev
print(f"teste: {info['n_sujeitos_teste']} sujeitos, {len(teste)} blocos", flush=True)


# ── A. derivação ────────────────────────────────────────────────────────────────────────────────────────────────────
def janelas_do_teste():
    ids = set(teste.subject_id.astype(str))
    partes = []
    for f in sorted(os.listdir(C.DIR_FEATURES)):
        if not (f.startswith("features_") and f.endswith("_baseline.csv")):
            continue
        for d in pd.read_csv(f"{C.DIR_FEATURES}/{f}", chunksize=400_000):
            d = d.rename(columns={'subject_folder': 'subject_id', 'segment': 'segment_name'})
            d['subject_id'] = d['subject_id'].astype(str)
            d = d[d.subject_id.isin(ids)]
            if 'gender' not in d.columns:
                d['gender'] = 'M' if 'homens' in f.lower() else 'F'
            partes.append(d)
    return pd.concat(partes, ignore_index=True)


jan = janelas_do_teste()
print(f"janelas do teste: {len(jan)} (sem PAT: {int(jan.pat_mean.isna().sum())})", flush=True)
jan['record_id'] = jan['segment_name'].astype(str).str.rsplit('_', n=1).str[0]
blocos = D._colapsar_e_sessoes(jan.copy())           # as sessões do treino, antes dos filtros

RAW = ['pat_mean', 'hr_mean', 'w50', 'area_mean', 'ppg_valley_intensity', 'ppg_ac_amplitude', 'alpha_womersley',
       'alpha_womersley_b', 'qt_interval', 'tq_interval', 'qtc_interval', 'sdi_interaction', 'qrs_duration',
       'r_amplitude', 'qrs_amplitude', 'qrs_area']
NOMES_EXTRACAO = {'ppg_crest_time': 'crest_time', 'ppg_decay_time': 'decay_time'}   # o servidor renomeia ao fechar o bloco

linhas_cmp, sessoes_a, excluidas = [], [], dict(varios_segmentos=0, fora_70_190=0)
for sid, g in teste.groupby('session_id', sort=False):
    pre = blocos[blocos.session_id == sid]
    if pre.segment_order.nunique() != 1:
        excluidas['varios_segmentos'] += 1
        continue
    if not ((pre.sbp_target > 70) & (pre.sbp_target < 190)).all():
        excluidas['fora_70_190'] += 1
        continue
    seg = pre.segment_order.iloc[0]
    rec = pre.record_id.iloc[0]
    b0, b1 = int(pre.bloco_id.min()), int(pre.bloco_id.max())
    w = jan[(jan.record_id == rec) & (jan.segment_order == seg)
            & (jan.window_idx >= 3 * b0) & (jan.window_idx <= 3 * b1 + 2)].set_index('window_idx')
    assert w.index.is_unique
    st = comum.SessionState()
    pinfo = {'idade': float(g.age.iloc[0]), 'genero': g.gender.iloc[0]}
    saidas = {}
    for wi in range(3 * b0, 3 * b1 + 4):               # uma janela a mais fecha o último bloco
        raw = None
        if wi in w.index and np.isfinite(w.at[wi, 'pat_mean']):
            r = w.loc[wi]
            raw = {k: (None if pd.isna(r[k]) else float(r[k])) for k in RAW}
            for nome, nome_ext in NOMES_EXTRACAO.items():
                raw[nome_ext] = None if pd.isna(r[nome]) else float(r[nome])
        out = comum.derivar_features_ml(raw, st, pinfo)
        if out is not None:
            saidas[b0 + out.pop('_bid')] = out
    srv = pd.DataFrame.from_dict(saidas, orient='index')
    trn = g.set_index('bloco_id')
    comuns = trn.index.intersection(srv.index)
    linhas_cmp.append(pd.concat({'treino': trn.loc[comuns, FEATS], 'servidor': srv.loc[comuns, FEATS]}, axis=1)
                      .assign(session_id=sid))
    # o servidor só começa ao 10.º bloco válido (janela de calibração); a partir daí tem de cobrir todas as linhas
    ok_pre = pre[(pre.pat_mean > comum.PAT_MIN) & (pre.pat_mean < comum.PAT_MAX)].bloco_id.tolist()
    depois_calib = trn.index[trn.index >= ok_pre[comum.BLOCOS_CALIB - 1]]
    sessoes_a.append(dict(session_id=sid, linhas_treino=len(trn), linhas_servidor=len(srv), comparadas=len(comuns),
                          em_falta=int(len(depois_calib.difference(srv.index))),
                          a_mais=int(len(srv.index.difference(trn.index)))))

cmp = pd.concat(linhas_cmp)
por_feat = {}
for f in FEATS:
    t, s = cmp[('treino', f)].astype(float).values, cmp[('servidor', f)].astype(float).values
    dif = np.abs(t - s)
    difere = dif > 1e-6 * np.maximum(1.0, np.abs(t))
    por_feat[f] = dict(n=int(len(t)), dif_max=float(dif.max()), dif_mediana=float(np.median(dif)),
                       dif_rel_max=float((dif / np.maximum(np.abs(t), 1e-9)).max()),
                       r=float(np.corrcoef(t, s)[0, 1]) if np.std(t) > 0 and np.std(s) > 0 else None,
                       linhas_diferentes=int(difere.sum()),
                       sessoes_diferentes=int(cmp.loc[difere, 'session_id'].nunique()))
sa = pd.DataFrame(sessoes_a)
print(f"\nA. derivação: {len(sa)} sessões ({excluidas}), {len(cmp)} blocos comparados, "
      f"em falta {int(sa.em_falta.sum())}, a mais {int(sa.a_mais.sum())}")
print(pd.DataFrame(por_feat).T.to_string(float_format=lambda x: f"{x:.3g}"), flush=True)

# ── B. inferência ───────────────────────────────────────────────────────────────────────────────────────────────────
P = joblib.load(PACOTE)
assert list(P["features"]) == FEATS
X = (teste[FEATS].values - P["scaler_mean"]) / P["scaler_scale"]
bruto = np.zeros(len(teste))
for nome, m in (("booster_young", teste.age.values < P["idade_corte"]), ("booster_senior", teste.age.values >= P["idade_corte"])):
    b = xgb.Booster()
    b.load_model(bytearray(P[nome]))
    bruto[m] = b.predict(xgb.DMatrix(X[m]))
d = reconstruir(teste, pd.Series(bruto, index=teste.index))
M = metricas(d)
print(f"\nB. reconstrução do treino com o pacote instalado: MAE {M['mae_sbp']:.4f} ({M['n']} blocos), "
      f"no pacote {P['meta']['mae_holdout']}", flush=True)

AJUDA = r'''
import sys, numpy as np
sys.path.insert(0, sys.argv[3])
import worker_absoluto as W, comum
z = np.load(sys.argv[1], allow_pickle=True)
feats, X, sess, cal = list(z["feats"]), z["X"], z["sess"], z["cal"]
assert W.MODELO is not None and list(W.MODELO["features"]) == feats
out = np.full(len(X), np.nan)
st, atual = None, None
for i in range(len(X)):
    if sess[i] != atual:
        st, atual = comum.SessionState(), sess[i]
    v = W.prever_ta(dict(zip(feats, X[i])), {"sbp": float(cal[i])}, st)[0]
    if v is not None:
        out[i] = v
np.save(sys.argv[2], out)
import xgboost, numpy
print("servidor:", "xgboost", xgboost.__version__, "numpy", numpy.__version__)
'''
with tempfile.TemporaryDirectory() as tmp:
    np.savez(f"{tmp}/entrada.npz", feats=np.array(FEATS), X=teste[FEATS].values.astype(float),
             sess=pd.factorize(teste.session_id)[0], cal=teste.sbp_calib.values.astype(float))
    open(f"{tmp}/ajuda.py", "w").write(AJUDA)
    r = subprocess.run([PY_SERVIDOR, f"{tmp}/ajuda.py", f"{tmp}/entrada.npz", f"{tmp}/saida.npy", SERVIDOR],
                       capture_output=True, text=True, cwd=tmp)
    print(r.stdout.strip(), r.stderr.strip()[-1500:] if r.returncode else "")
    assert r.returncode == 0
    srv = np.load(f"{tmp}/saida.npy")

tem = np.isfinite(srv)
pos = teste.groupby('session_id', sort=False).cumcount().values
trn = d['preds_final'].reindex(teste.index).values
dif = np.abs(srv[tem] - trn[tem])
igual_arred = float(np.mean(srv[tem] == np.round(trn[tem], 1)))
mae_srv = float(np.mean(np.abs(srv[tem] - teste.sbp_target.values[tem])))
mae_trn = float(np.mean(np.abs(trn[tem] - teste.sbp_target.values[tem])))
print(f"   worker: {int(tem.sum())} previsões (as duas primeiras de cada sessão formam a âncora: "
      f"{int((~tem).sum())} sem valor, todas na posição < 2: {bool((pos[~tem] < 2).all())})")
print(f"   |worker - treino|: máx {dif.max():.4f}, mediana {np.median(dif):.4f} mmHg; "
      f"iguais ao treino arredondado à décima: {100 * igual_arred:.3f} %")
print(f"   MAE nas mesmas linhas: worker {mae_srv:.4f}, treino {mae_trn:.4f}", flush=True)

res = dict(
    data="2026-10-14", pacote=PACOTE, treinado_em=P["meta"]["treinado_em"],
    derivacao=dict(sessoes=len(sa), excluidas=excluidas, blocos_comparados=int(len(cmp)),
                   em_falta_depois_da_calibracao=int(sa.em_falta.sum()), a_mais=int(sa.a_mais.sum()),
                   dif_max_global=max(v['dif_max'] for v in por_feat.values()),
                   r_min=min(v['r'] for v in por_feat.values() if v['r'] is not None), por_caracteristica=por_feat),
    inferencia=dict(blocos=int(tem.sum()), sem_valor_fora_da_ancora=int(((~tem) & (pos >= 2)).sum()),
                    dif_max_mmHg=float(dif.max()), iguais_arredondado=igual_arred,
                    mae_holdout_reconstrucao=round(M['mae_sbp'], 4), mae_holdout_pacote=P['meta']['mae_holdout'],
                    mae_worker_mesmas_linhas=round(mae_srv, 4), mae_treino_mesmas_linhas=round(mae_trn, 4)))
os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
json.dump(res, open(SAIDA, "w"), indent=1, ensure_ascii=False)
print(f"\nGuardado: {SAIDA}")
