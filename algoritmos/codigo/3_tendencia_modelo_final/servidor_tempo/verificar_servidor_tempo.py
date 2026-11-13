#!/usr/bin/env python3
"""verificar_servidor_tempo.py — derivação do servidor (derivacao_ppg.py) contra a tabela da tendência do treino, no conjunto de teste inteiro.

Adaptado de 5_servidor_e_carga/verificacao_da_extracao_e_derivacao_contra_o_treino/05_verificar_servidor_teste_completo.py:
  entrada   as janelas extraídas no treino com as características novas (Features_morfologia/features_*_ppg_apg_morf.csv) dos sujeitos do teste
  servidor  derivacao_ppg.derivar_features_ppg, janela a janela, uma sessão do treino de cada vez
  treino    a tabela da tendência (tendencia_tempo_fixo/dados/holdout_ppg.csv) com as variações por TEMPO e a volatilidade (tempo_fixo.preparar_grade)
Compara bloco a bloco todas as características do modelo (as 43), incluindo onde ficam em falta: têm de estar em falta nos mesmos blocos.
Entram as sessões de um só segmento e sem blocos fora de 70 < PAS < 190 (o servidor não tem o alvo para esse filtro). Escreve resultado_servidor_tempo.json.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = "/home/filipe/Desktop/Tese/algoritmo-ml"
TF = f"{RAIZ}/Tentativa_22_10/tendencia_tempo_fixo"
sys.path.insert(0, TF)
import tempo_fixo as tfx  # noqa: E402

DERIVACAO = f"{AQUI}/../../../../servidor/workers/derivacao_ppg.py"
spec = importlib.util.spec_from_file_location("dpp", DERIVACAO)
dpp = importlib.util.module_from_spec(spec); spec.loader.exec_module(dpp)
RESERVA = json.load(open(f"{AQUI}/sd_reserva_ppg_morf.json"))["reserva"]
assert set(dpp.SD_RESERVA) == set(RESERVA) and all(abs(dpp.SD_RESERVA[k] - RESERVA[k]) <= 1e-12 * max(1, abs(RESERVA[k])) for k in RESERVA), \
    "SD_RESERVA de derivacao_ppg não é a do 06"
RAW = ['hr_mean', 'w50', 'area_mean', 'ppg_valley_intensity', 'ppg_ac_amplitude', 'ppg_crest_time', 'ppg_decay_time', 'alpha_womersley',
       'w25', 'w75', 'ppg_max_slope', 'ppg_max_slope_norm', 'reflection_index', 'apg_b_a', 'apg_d_a']   # o que a extração do servidor devolve
SAIDA = f"{AQUI}/resultado_servidor_tempo.json"

grade, V = tfx.preparar_grade(f"{TF}/dados", 'holdout')
FEATS = V['tudo']
treino = grade[grade.sbp_target.notna()].copy()
treino['age'] = pd.read_csv(f"{TF}/dados/holdout_ppg.csv", usecols=['session_id', 'bloco_id', 'age']).set_index(['session_id', 'bloco_id']).loc[
    list(zip(treino.session_id, treino.bloco_id)), 'age'].values
tab = pd.read_csv(f"{TF}/dados/holdout_ppg.csv", usecols=['session_id', 'bloco_id', 'gender'])
treino = treino.merge(tab, on=['session_id', 'bloco_id'])
ids = set(treino.subject_id.astype(str))
print(f"treino: {len(treino)} blocos, {treino.session_id.nunique()} sessões, {len(ids)} sujeitos", flush=True)

partes = []
for g in ("homens", "mulheres"):
    for d in pd.read_csv(f"/media/filipe/Dados/Features_morfologia/features_{g}_ppg_apg_morf.csv", chunksize=400_000):
        d['subject_id'] = d['subject_id'].astype(str)
        partes.append(d[d.subject_id.isin(ids)])
jan = pd.concat(partes, ignore_index=True)
jan['record_id'] = jan['segment_name'].astype(str).str.rsplit('_', n=1).str[0]
jan['bloco_id'] = jan['window_idx'] // 3
meta = ['subject_id', 'record_id', 'segment_order', 'bloco_id', 'gender', 'dataset', 'age']
blocos = (jan.groupby(meta)[['sbp_target', 'hr_mean']].mean().reset_index()
          .sort_values(['subject_id', 'record_id', 'segment_order', 'bloco_id']).reset_index(drop=True))
blocos['seg_diff'] = blocos.groupby('record_id')['segment_order'].diff()
blocos['win_diff'] = blocos.groupby(['record_id', 'segment_order'])['bloco_id'].diff()
nova = ((blocos['seg_diff'] > 1) | (blocos['win_diff'] > 4) | blocos['seg_diff'].isna() | blocos['win_diff'].isna())
blocos['session_id'] = blocos['record_id'] + "_" + nova.groupby(blocos['record_id']).cumsum().astype(str)
print(f"janelas: {len(jan)}", flush=True)
jan = jan.set_index(['record_id', 'segment_order'])

linhas, sessoes, excluidas = [], [], dict(varios_segmentos=0, fora_70_190=0)
for sid, g in treino.groupby('session_id', sort=False):
    pre = blocos[blocos.session_id == sid]
    if pre.segment_order.nunique() != 1:
        excluidas['varios_segmentos'] += 1
        continue
    if not ((pre.sbp_target > 70) & (pre.sbp_target < 190)).all():
        excluidas['fora_70_190'] += 1
        continue
    rec, seg = pre.record_id.iloc[0], pre.segment_order.iloc[0]
    b0, b1 = int(pre.bloco_id.min()), int(pre.bloco_id.max())
    w = jan.loc[(rec, seg)]
    w = w[(w.window_idx >= 3 * b0) & (w.window_idx <= 3 * b1 + 2)].set_index('window_idx')
    assert w.index.is_unique
    est = dpp.EstadoSessaoPPG()
    pinfo = {'idade': float(g.age.iloc[0]), 'genero': g.gender.iloc[0]}
    saidas = {}
    for wi in range(3 * b0, 3 * b1 + 4):               # uma janela a mais fecha o último bloco
        raw = None
        if wi in w.index:
            r = w.loc[wi]
            raw = {k: (None if pd.isna(r[k]) else float(r[k])) for k in RAW}
        out = dpp.derivar_features_ppg(raw, est, pinfo)
        if out is not None:
            saidas[b0 + out.pop('_bid')] = out
    srv = pd.DataFrame.from_dict(saidas, orient='index')
    trn = g.set_index('bloco_id')
    obrig = ['hr_mean_z', 'hr_lag1'] + [f'{a}_global_z' for a in dpp.ANCORAS]
    completas = trn.index[trn[obrig].notna().all(axis=1)] if len(trn) else trn.index[:0]
    comuns = completas.intersection(srv.index)
    feats = [c for c in FEATS if c in srv.columns]
    if len(comuns):
        linhas.append(pd.concat({'treino': trn.loc[comuns, feats], 'servidor': srv.loc[comuns, feats]}, axis=1).assign(session_id=sid))
    sessoes.append(dict(session_id=sid, linhas_treino_completas=len(completas), linhas_servidor=len(srv), comparadas=len(comuns),
                        so_treino=int(len(completas.difference(srv.index))), so_servidor=int(len(srv.index.difference(completas)))))

cmp = pd.concat(linhas)
sa = pd.DataFrame(sessoes)
por_feat = {}
for f in FEATS:
    if ('treino', f) not in cmp.columns:
        por_feat[f] = dict(erro='não produzida pelo servidor'); continue
    t, s = cmp[('treino', f)].astype(float).values, cmp[('servidor', f)].astype(float).values
    falta_t, falta_s = np.isnan(t), np.isnan(s)
    ambos = ~falta_t & ~falta_s
    dif = np.abs(t[ambos] - s[ambos])
    difere = dif > 1e-6 * np.maximum(1.0, np.abs(t[ambos]))
    por_feat[f] = dict(n=int(len(t)), em_falta_treino=int(falta_t.sum()), em_falta_servidor=int(falta_s.sum()),
                       em_falta_so_num_dos_dois=int((falta_t != falta_s).sum()),
                       dif_max=float(dif.max()) if len(dif) else None,
                       r=float(np.corrcoef(t[ambos], s[ambos])[0, 1]) if ambos.sum() > 2 and np.std(t[ambos]) > 0 and np.std(s[ambos]) > 0 else None,
                       linhas_diferentes=int(difere.sum()))
print(f"\n{len(sa)} sessões ({excluidas}), {len(cmp)} blocos comparados | só no treino {int(sa.so_treino.sum())} "
      f"({int((sa.so_treino > 0).sum())} sessões), só no servidor {int(sa.so_servidor.sum())} ({int((sa.so_servidor > 0).sum())} sessões)")
print(pd.DataFrame(por_feat).T.to_string(float_format=lambda x: f"{x:.3g}"), flush=True)
json.dump(dict(data="2026-10-23", derivacao=os.path.normpath(DERIVACAO), sessoes=len(sa), excluidas=excluidas,
               blocos_comparados=int(len(cmp)), so_treino=int(sa.so_treino.sum()), so_servidor=int(sa.so_servidor.sum()),
               por_caracteristica=por_feat), open(SAIDA, "w"), indent=1, ensure_ascii=False)
print(f"Guardado: {SAIDA}")
