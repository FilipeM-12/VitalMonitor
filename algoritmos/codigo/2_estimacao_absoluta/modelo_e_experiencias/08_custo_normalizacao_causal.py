#!/usr/bin/env python3
"""Quanto custa normalizar as âncoras morfológicas com a janela de calibração do doente (causal) em vez do doente
inteiro (que olha para o futuro e não se pode calcular à cabeceira)? Para cada variante, mesma coorte, mesmos
hiperparâmetros (os da busca causal), muda só a normalização. Na variante de referência, diferença emparelhada por
sujeito com bootstrap. É o único sítio onde as âncoras `doente_inteiro` são usadas."""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nucleo import config as C
from nucleo.avaliacao import avaliar_holdout, bootstrap_sujeitos
from nucleo.busca import melhores_parametros
from nucleo.dados import carregar

def ganho_ancora(d):
    """Erro da âncora (repetir a última calibração) menos o erro do modelo; positivo = o modelo ganha à âncora."""
    return float((d['sbp_calib'] - d['sbp_target']).abs().mean() - (d['preds_final'] - d['sbp_target']).abs().mean())


linhas, ref = [], None
for variante in ("baseline", "nk", "nk2", "fft", "clinicas"):
    params = melhores_parametros("XGB", variante)
    resultado = {}
    for ancoras in ("causal", "doente_inteiro"):
        dev, teste, info = carregar(variante, ancoras)
        d, m = avaliar_holdout("XGB", params, dev, teste, C.VARIANTES[variante].features)
        resultado[ancoras] = (d, m)
    mc, mn = resultado["causal"][1], resultado["doente_inteiro"][1]
    linhas.append(dict(variante=variante, mae_causal=round(mc['mae_sbp'], 4), mae_doente_inteiro=round(mn['mae_sbp'], 4),
                       custo=round(mc['mae_sbp'] - mn['mae_sbp'], 4), n_causal=mc['n'], n_doente_inteiro=mn['n'],
                       sujeitos_causal=mc['n_sujeitos'], sujeitos_doente_inteiro=mn['n_sujeitos'],
                       ganho_vs_ancora_causal=round(ganho_ancora(resultado["causal"][0]), 4),
                       ganho_vs_ancora_doente_inteiro=round(ganho_ancora(resultado["doente_inteiro"][0]), 4)))
    print(linhas[-1], flush=True)
    if variante == "baseline":
        chave = ['subject_id', 'record_id', 'segment_order', 'bloco_id']
        d_c, d_n = (resultado[k][0].set_index(chave) for k in ("causal", "doente_inteiro"))
        comum = d_c.index.intersection(d_n.index)
        erro = {k: (d.loc[comum, 'sbp_target'] - d.loc[comum, 'preds_final']).abs() for k, d in (("causal", d_c), ("doente_inteiro", d_n))}
        sujeito = pd.Series(comum.get_level_values('subject_id'), index=comum)
        ref = bootstrap_sujeitos(erro, sujeito, pares={"causal": "doente_inteiro"})
        por_sujeito = pd.DataFrame(erro).groupby(level='subject_id').mean()
        ref['n_doentes_que_pioram'] = int((por_sujeito['causal'] > por_sujeito['doente_inteiro']).sum())
        ref['n_doentes'] = int(len(por_sujeito))
os.makedirs(f"{C.DIR_RESULTADOS}/tabelas", exist_ok=True)
pd.DataFrame(linhas).to_csv(f"{C.DIR_RESULTADOS}/tabelas/custo_normalizacao_causal.csv", index=False)
json.dump(dict(variantes=linhas, baseline_emparelhado=ref), open(f"{C.DIR_RESULTADOS}/tabelas/custo_normalizacao_causal.json", "w"), indent=1)
print(json.dumps(ref, indent=1))
