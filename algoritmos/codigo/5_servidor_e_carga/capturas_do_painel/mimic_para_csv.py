#!/usr/bin/env python3
"""Converte segmentos reais do MIMIC (PLETH e ECG derivação II) para o CSV que o simulador reproduz (formato da bancada:
time_ms, ecg, ir, red a 100 Hz), para as capturas do painel. Só para ilustração; não entra em resultados.

  PLETH 125 Hz -> 100 Hz; IR = 37 300 - 300·z (o MAX30102 dá o pico sistólico como mínimo; o worker corrige a orientação);
  RED com DC 55 850 e amplitude relativa 0,5 da do IR (SpO2 de cerca de 97 %); ECG em contagens do ADC (base 2 184,
  onda R +410). Temperatura: o MIMIC não tem temperatura de pele e o simulador gera-a.

Escolha: doentes do conjunto de teste, segmentos com .dat no disco, >= 45 min, com PLETH e II; de cada doente o troço de
30 min (a começar aos 5 min) com a maior variação de PA sistólica em 5 min, desde que o PLETH tenha < 1 % de falhas;
só blocos com sistólica entre 70 e 190 mmHg (>= 90 % do troço) e variação máxima em 5 min entre 15 e
40 mmHg (uma descida ou subida real, não um artefacto da linha arterial); pelo menos 70 % das janelas aceites pelos portões
de qualidade do servidor (extracao_ppg_classificador.py); os três doentes com maior variação."""
import os, sys
import numpy as np, pandas as pd, wfdb
from scipy.signal import resample_poly, find_peaks
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "servidor", "workers"))
import extracao_ppg_classificador as epp

AQUI = os.path.dirname(os.path.abspath(__file__))
L = pd.read_csv("/media/filipe/Dados/MIMIC_DATASET_CONSOLIDADO/dataset_para_extração_features.csv")
H = set(open("/home/filipe/Desktop/Tese/algoritmo-ml/RESULTADOS_RELATORIO/codigo_dependencias/particao_sujeitos/pacientes_holdout.txt").read().split())
L = L[L.subject_folder.isin(H) & (L.duration_min >= 45) & L.channels.str.contains("PLETH") & L.channels.str.contains(r"\bII\b")]
L = L[[os.path.exists(p.replace(".hea", "") + ".dat") for p in L.absolute_path]]
INI_MIN, DUR_MIN = 5, 30


def sbp_por_bloco(abp, fs):
    out = []
    for k in range(0, len(abp) - int(15 * fs), int(15 * fs)):
        w = abp[k:k + int(15 * fs)]
        pk, _ = find_peaks(w, distance=int(fs * 0.4), height=np.nanpercentile(w, 50))
        out.append(np.nanmean(w[pk]) if len(pk) >= 3 else np.nan)
    return np.array(out)


cands = []
for subj, g in L.groupby("subject_folder"):
    for _, r in g.head(3).iterrows():
        base = r.absolute_path.replace(".hea", "").replace(".dat", "")
        try:
            h = wfdb.rdheader(base); fs = h.fs; nomes = [s.upper() for s in h.sig_name]
            ch = [nomes.index("PLETH"), nomes.index("II"), nomes.index("ABP")]
            rec = wfdb.rdrecord(base, channels=ch, sampfrom=int(INI_MIN * 60 * fs), sampto=int((INI_MIN + DUR_MIN) * 60 * fs))
        except Exception:
            continue
        sig = rec.p_signal
        if np.isnan(sig[:, 0]).mean() > 0.01 or np.isnan(sig[:, 1]).mean() > 0.01:
            continue
        s = sbp_por_bloco(sig[:, 2], fs)
        if np.mean((s > 70) & (s < 190)) < 0.9:            # linha arterial com muitos artefactos: fora
            continue
        s[(s <= 70) | (s >= 190)] = np.nan
        d5 = s[20:] - s[:-20]
        var = np.nanmax(np.abs(d5)) if np.isfinite(d5).any() else 0
        pl = resample_poly(pd.Series(sig[:, 0]).interpolate().bfill().ffill().values, 4, 5)
        ir = 37300 - 300 * (pl - np.median(pl)) / ((np.percentile(pl, 95) - np.percentile(pl, 5)) / 2)
        snap = [dict(ir=v) for v in ir]
        aceite = np.mean([epp.extrair_features_janela_ppg(snap[k:k + 2500], 100, 1500, 500) is not None
                          for k in range(0, len(snap) - 2500, 3000)])
        if aceite < 0.7:                                    # o servidor rejeitaria a maior parte (taquicardia, arritmia)
            continue
        if 15 <= var <= 40:                                 # uma descida ou subida real, não um artefacto
            cands.append((var, subj, base, fs))
            break
cands.sort(reverse=True)
escolhidos = cands[:3]
for n, (var, subj, base, fs) in enumerate(escolhidos, 1):
    h = wfdb.rdheader(base); nomes = [s.upper() for s in h.sig_name]
    rec = wfdb.rdrecord(base, channels=[nomes.index("PLETH"), nomes.index("II")],
                        sampfrom=int(INI_MIN * 60 * fs), sampto=int((INI_MIN + DUR_MIN) * 60 * fs))
    pl = pd.Series(rec.p_signal[:, 0]).interpolate().bfill().ffill().values
    ecg = pd.Series(rec.p_signal[:, 1]).interpolate().bfill().ffill().values
    pl = resample_poly(pl, 4, 5); ecg = resample_poly(ecg, 4, 5)          # 125 -> 100 Hz
    z = (pl - np.median(pl)) / ((np.percentile(pl, 95) - np.percentile(pl, 5)) / 2)
    ir = 37300 - 300 * z
    red = 55850 - 300 * 0.5 * (55850 / 37300) * z
    e = (ecg - np.median(ecg)) / max(np.percentile(ecg, 99.5) - np.median(ecg), 1e-6)
    ecg_c = np.clip(2184 + 410 * e, 0, 4095)
    t = np.arange(len(ir)) * 10
    out = f"{AQUI}/mimic_paciente_{n}.csv"
    pd.DataFrame(dict(time_ms=t, ecg=ecg_c.round(), ir=ir.round(), red=red.round(), bpm=-1, spo2=-1, tempC=-1000)).to_csv(out, index=False)
    print(f"Paciente {n}: {subj} {os.path.basename(base)} (variação máx. de PA sistólica em 5 min: {var:.1f} mmHg) -> {os.path.basename(out)}")
