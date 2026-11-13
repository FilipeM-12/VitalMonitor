"""verificar_extracao_morf.py — a extração do servidor (extracao_ppg.py) devolve EXATAMENTE o que a do treino (09b) devolve?

Para N registos do MIMIC, filtra-se o registo inteiro como no treino (passa-baixo de 8 Hz) e passa-se cada janela de 15 s (passo de 5 s) às duas
funções processar_janela_so_ppg. Compara-se, janela a janela, as chaves que o servidor produz: tem de haver a mesma decisão (janela aceite ou
rejeitada) e os mesmos valores. Escreve resultado_extracao_morf.json.   venv/bin/python verificar_extracao_morf.py [n_registos]
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import wfdb

AQUI = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def carregar(caminho, nome):
    spec = importlib.util.spec_from_file_location(nome, caminho)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


treino = carregar(f"{AQUI}/../extracao_morfologia/09b_extrair_features_ppg_morfologia.py", "treino")
srv = carregar(f"{AQUI}/../../../../servidor/workers/extracao_ppg.py", "srv")
df = pd.read_csv(treino.INPUT_CSVS[0]).sample(N * 4, random_state=7)
n_jan = n_ac = n_dif = 0
por_chave, divergencias = {}, []
registos = 0
for _, row in df.iterrows():
    if registos >= N:
        break
    base = row['absolute_path'].replace('.dat', '').replace('.hea', '')
    if not os.path.exists(base + '.hea'):
        continue
    try:
        hdr = wfdb.rdheader(base); nomes = [s.upper() for s in hdr.sig_name]
        canais = ['PLETH', 'PPG']
        i = next(k for k, s in enumerate(nomes) if any(x in s for x in canais))
        x = wfdb.rdrecord(base, channels=[i], sampto=int(hdr.fs * 1500)).p_signal[:, 0].astype(np.float32)
    except Exception:
        continue
    x = pd.Series(x).interpolate().ffill().bfill().values
    filt = treino.butter_lowpass_filter(x, 8, hdr.fs)
    fs = hdr.fs
    w, st = int(15 * fs), int(5 * fs)
    registos += 1
    for a in range(0, len(filt) - w + 1, st):
        win = filt[a:a + w]
        if np.std(win) < 0.005:
            continue
        t = treino.processar_janela_so_ppg(win, fs)
        s = srv.processar_janela_so_ppg(win, fs)
        n_jan += 1
        if (t is None) != (s is None):
            n_dif += 1; divergencias.append(dict(registo=base, janela=a, treino=t is not None, servidor=s is not None)); continue
        if t is None:
            continue
        n_ac += 1
        for k in s:
            tv, sv = t.get(k), s[k]
            ok = (tv is None and sv is None) or (tv is not None and sv is not None and abs(tv - sv) <= 1e-9)
            por_chave.setdefault(k, [0, 0]); por_chave[k][0] += 1; por_chave[k][1] += (not ok)
            if not ok and len(divergencias) < 20:
                divergencias.append(dict(registo=base, janela=a, chave=k, treino=tv, servidor=sv))
res = dict(registos=registos, janelas=n_jan, aceites=n_ac, decisao_diferente=n_dif, por_chave={k: dict(n=v[0], diferentes=v[1]) for k, v in por_chave.items()},
           divergencias=divergencias[:20])
print(json.dumps({k: v for k, v in res.items() if k != 'divergencias'}, indent=1)); print(divergencias[:5])
json.dump(res, open(f"{AQUI}/resultado_extracao_morf.json", "w"), indent=1, ensure_ascii=False, default=float)
