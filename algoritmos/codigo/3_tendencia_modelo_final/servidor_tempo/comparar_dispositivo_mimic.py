"""comparar_dispositivo_mimic.py — as características novas têm no dispositivo a mesma variabilidade (dentro da sessão) que no MIMIC?

O modelo vê cada característica normalizada pela calibração da sessão (z-score dos 10 primeiros blocos), por isso o nível absoluto não conta, mas a
variabilidade do z conta. Para os 3 ensaios com a pulseira (gravacoes_mqtt, canal IR) corre-se a extração e a derivação do servidor (as _morf/_tempo) e
compara-se, com os blocos 10 a 60 das sessões do MIMIC do conjunto de teste (os ensaios têm 28 a 60 blocos), o desvio padrão e os percentis do z de cada
característica e a taxa de blocos em que existe. Escreve comparacao_dispositivo_mimic.csv.   venv/bin/python comparar_dispositivo_mimic.py
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = "/home/filipe/Desktop/Tese/algoritmo-ml"


def carregar(caminho, nome):
    spec = importlib.util.spec_from_file_location(nome, caminho)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


TRABALHADORES = f"{AQUI}/../../../../servidor/workers"
ext, dpp = carregar(f"{TRABALHADORES}/extracao_ppg.py", "ext"), carregar(f"{TRABALHADORES}/derivacao_ppg.py", "dpp")
RAW = ['hr_mean', 'w50', 'area_mean', 'ppg_valley_intensity', 'ppg_ac_amplitude', 'ppg_crest_time', 'ppg_decay_time', 'alpha_womersley',
       'w25', 'w75', 'ppg_max_slope', 'ppg_max_slope_norm', 'reflection_index', 'apg_b_a', 'apg_d_a']
COLS = [f'{a}_global_z' for a in dpp.TODAS_ANCORAS if a != 'ppg_max_slope'] + ['hr_mean_z']

linhas = []
for f in sorted(glob.glob('/home/filipe/Desktop/Tese/arduino/Dados/gravacoes_mqtt/*PULSEIRA_01*.csv')):
    d = pd.read_csv(f)
    fs = 1000.0 / d.time_ms.diff().replace(0, np.nan).median()
    x = d.ir.values.astype(float)
    w, st = int(15 * fs), int(5 * fs)
    filt = ext.butter_lowpass_filter(x, 8.0, fs)
    est, saidas = dpp.EstadoSessaoPPG(), []
    for a in range(0, len(x) - w + 1, st):
        r = ext.processar_janela_so_ppg(filt[a:a + w], fs)
        out = dpp.derivar_features_ppg(r, est, {'idade': 30, 'genero': 'M'})
        if out is not None:
            saidas.append(out)
    s = pd.DataFrame(saidas)
    s['registo'] = os.path.basename(f)[:6]
    linhas.append(s)
disp = pd.concat(linhas, ignore_index=True)
print('dispositivo: blocos com saída', disp.groupby('registo').size().to_dict())

tab = pd.read_csv(f"{RAIZ}/Tentativa_22_10/tendencia_tempo_fixo/dados/holdout_ppg.csv")
tab = tab.sort_values(['session_id', 'bloco_id'])
tab['k'] = tab.groupby('session_id').cumcount()
mim = tab[(tab.k >= 10) & (tab.k < 60)]

linhas = []
for c in COLS:
    d, m = disp[c].dropna(), mim[c].dropna()
    linhas.append(dict(caracteristica=c, existe_disp=round(disp[c].notna().mean(), 3), existe_mimic=round(mim[c].notna().mean(), 3),
                       dp_disp=round(d.std(), 2), dp_mimic=round(m.std(), 2), razao_dp=round(d.std() / m.std(), 2),
                       p5_disp=round(d.quantile(.05), 2), p95_disp=round(d.quantile(.95), 2),
                       p5_mimic=round(m.quantile(.05), 2), p95_mimic=round(m.quantile(.95), 2)))
r = pd.DataFrame(linhas)
r.to_csv(f"{AQUI}/comparacao_dispositivo_mimic.csv", index=False)
print(r.to_string(index=False))
