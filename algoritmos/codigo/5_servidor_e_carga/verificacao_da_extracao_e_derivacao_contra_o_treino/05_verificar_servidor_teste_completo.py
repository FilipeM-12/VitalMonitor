#!/usr/bin/env python3
"""05 — Derivação da tendência: servidor contra treino no conjunto de teste inteiro.

O 04 verifica a derivação em 5 registos, cada um tratado como um conjunto de dados à parte. Isso esconde o que o treino
tira de outras sessões (a mediana de reserva do desvio de calibração, quando o desvio é nulo) e as linhas que o treino
deita fora. Aqui, como no Absoluto_27/19_verificar_servidor.py do absoluto:
  entrada   as janelas extraídas no treino (features_*_ppg_apg.csv) dos sujeitos do conjunto de teste
  servidor  servidor/workers/derivacao_ppg_classificador.derivar_features_ppg, janela a janela, uma sessão do treino de cada vez
  treino    a tabela única da tendência, holdout_ppg.csv (construir_tabela_ppg.py), com as variações e
            a volatilidade calculadas nessa tabela
Compara bloco a bloco (mesmo bloco_id) todas as características que o servidor produz e a tabela do treino tem.
Entram as sessões de um só segmento e sem blocos fora de 70 < PAS < 190 (o servidor não tem o alvo para esse filtro).
Escreve resultado_servidor_teste_completo.json.
  venv/bin/python 05_verificar_servidor_teste_completo.py
"""
import importlib.util
import json
import os
import types

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = "/home/filipe/Desktop/Tese"
DERIV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "servidor", "workers", "derivacao_ppg_classificador.py")
JANELAS = "/media/filipe/Dados/Features"
SA = f"{RAIZ}/algoritmo-ml/RESULTADOS_RELATORIO/Sem_apg"
TABELA = f"{RAIZ}/algoritmo-ml/2026-09-09_normalizacao_causal_tendencia/dados/holdout_ppg.csv"   # a tabela única, no sítio canónico
AVALIAR = f"{RAIZ}/algoritmo-ml/RESULTADOS_RELATORIO/tudo_3_dobras/causal09/02_avaliar.py"
SAIDA = f"{AQUI}/resultado_servidor_teste_completo.json"
RAW = ['hr_mean', 'w50', 'area_mean', 'ppg_valley_intensity', 'ppg_ac_amplitude', 'ppg_crest_time', 'ppg_decay_time',
       'alpha_womersley']   # o que a extração do servidor devolve

spec = importlib.util.spec_from_file_location("dpp", DERIV)
dpp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dpp)
RESERVA = json.load(open(f"{AQUI}/sd_reserva_ppg.json"))["reserva"]
assert dpp.SD_RESERVA == {k: RESERVA[k] for k in dpp.SD_RESERVA}, "derivacao_ppg_classificador.SD_RESERVA não é a do 06"
assert set(dpp.SD_RESERVA) == {'hr_mean', *dpp.ANCORAS}
av = types.ModuleType("av")
exec(compile(open(AVALIAR, encoding="utf-8").read().replace('if __name__ == "__main__":', 'if False:'),
             "02_avaliar.py", "exec"), av.__dict__)

# ── treino: a tabela do conjunto de teste com as variações multi-escala ─────────────────────────────────────────────
treino = pd.read_csv(TABELA).sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
treino, multi = av.add_multi(treino)
ids = set(treino.subject_id.astype(str))
print(f"treino: {len(treino)} blocos, {treino.session_id.nunique()} sessões, {len(ids)} sujeitos", flush=True)

# ── janelas dos mesmos sujeitos e as sessões do treino antes dos filtros ────────────────────────────────────────────
partes = []
for f in ("features_homens_ppg_apg.csv", "features_mulheres_ppg_apg.csv"):
    for d in pd.read_csv(f"{JANELAS}/{f}", chunksize=400_000):
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

# ── servidor, sessão a sessão ──────────────────────────────────────────────────────────────────────────────────────
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
    w = jan[(jan.record_id == rec) & (jan.segment_order == seg)
            & (jan.window_idx >= 3 * b0) & (jan.window_idx <= 3 * b1 + 2)].set_index('window_idx')
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
    feats = [c for c in srv.columns if c in trn.columns] if len(srv) else []
    completas = trn.index[trn[feats or ['hr_mean_z']].notna().all(axis=1)] if feats else trn.index[:0]
    comuns = completas.intersection(srv.index)
    if len(comuns):
        linhas.append(pd.concat({'treino': trn.loc[comuns, feats], 'servidor': srv.loc[comuns, feats]}, axis=1)
                      .assign(session_id=sid))
    sessoes.append(dict(session_id=sid, linhas_treino_completas=len(completas), linhas_servidor=len(srv),
                        comparadas=len(comuns), so_treino=int(len(completas.difference(srv.index))),
                        so_servidor=int(len(srv.index.difference(completas)))))

cmp = pd.concat(linhas)
sa = pd.DataFrame(sessoes)
por_feat = {}
for f in sorted({c[1] for c in cmp.columns if c[0] == 'treino'}):
    t, s = cmp[('treino', f)].astype(float).values, cmp[('servidor', f)].astype(float).values
    dif = np.abs(t - s)
    difere = dif > 1e-6 * np.maximum(1.0, np.abs(t))
    por_feat[f] = dict(n=int(len(t)), dif_max=float(dif.max()), dif_mediana=float(np.median(dif)),
                       r=float(np.corrcoef(t, s)[0, 1]) if np.std(t) > 0 and np.std(s) > 0 else None,
                       linhas_diferentes=int(difere.sum()), sessoes_diferentes=int(cmp.loc[difere, 'session_id'].nunique()))
print(f"\n{len(sa)} sessões ({excluidas}), {len(cmp)} blocos comparados | só no treino {int(sa.so_treino.sum())} "
      f"({int((sa.so_treino > 0).sum())} sessões), só no servidor {int(sa.so_servidor.sum())} "
      f"({int((sa.so_servidor > 0).sum())} sessões)")
print(pd.DataFrame(por_feat).T.to_string(float_format=lambda x: f"{x:.3g}"), flush=True)
json.dump(dict(data="2026-10-14", derivacao=DERIV, sessoes=len(sa), excluidas=excluidas, blocos_comparados=int(len(cmp)),
               so_treino=int(sa.so_treino.sum()), so_servidor=int(sa.so_servidor.sum()),
               sessoes_com_diferenca_de_linhas=int(((sa.so_treino > 0) | (sa.so_servidor > 0)).sum()),
               por_caracteristica=por_feat), open(SAIDA, "w"), indent=1, ensure_ascii=False)
print(f"Guardado: {SAIDA}")
