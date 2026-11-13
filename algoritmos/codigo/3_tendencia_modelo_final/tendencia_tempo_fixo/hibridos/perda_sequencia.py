"""perda_sequencia.py — quantas linhas com alvo ficam sem previsão nas redes por lhes faltar um bloco na sequência de 20 (5 min).
Mesma regra de hibrido_quantilico_tempo.construir_sequencias: a linha só entra se os 20 blocos seguidos que acabam nela (na grade de 15 s) têm as 11 características da sequência.
Escreve perda_sequencia.json."""
import json, os, sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
AQUI = os.path.dirname(os.path.abspath(__file__)); TF = os.path.dirname(AQUI)
sys.path.insert(0, TF)
import tempo_fixo as tf
T = 20
FEAT_SEQ = ['hr_mean_z', 'hr_lag1', 'alpha_womersley_global_z', 'area_mean_global_z', 'w50_global_z', 'ppg_crest_time_global_z', 'ppg_decay_time_global_z',
            'w25_global_z', 'w75_global_z', 'ppg_max_slope_norm_global_z', 'apg_b_a_global_z']
R = {}
for nome in ('dev', 'holdout'):
    g, _ = tf.preparar_grade(f"{TF}/dados", nome)
    com_alvo = ent = 0
    for sid, s in g.groupby('session_id'):
        s = s.sort_values('bloco_id')
        alvo = s['delta'].notna().values; com_alvo += int(alvo.sum())
        if len(s) < T: continue
        sw = sliding_window_view(s[FEAT_SEQ].values.astype('float32'), T, axis=0)      # (n-T+1, F, T)
        completo = ~np.isnan(sw.reshape(len(sw), -1)).any(axis=1)
        ent += int((completo & alvo[T - 1:]).sum())
    R[nome] = dict(linhas_com_alvo=com_alvo, linhas_com_sequencia_completa=ent, fracao_perdida=1 - ent / com_alvo)
    print(nome, R[nome], flush=True)
json.dump(R, open(f"{AQUI}/perda_sequencia.json", "w"), indent=1)
