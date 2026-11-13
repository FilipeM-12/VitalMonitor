#!/usr/bin/env python3
"""Mede, sobre sinal do MIMIC, em que a extração das características do absoluto no servidor
(`comum.extrair_features_janela` / `comum.processar_janela_absoluto`) difere da do treino
(`processar_janela_teste1` do Extração_features.ipynb, variante baseline, com a filtragem e as janelas do mesmo caderno).

As diferenças medidas, uma de cada vez a partir da cadeia do treino (TT):
  glitch   remoção de picos de 1 amostra: limiar absoluto de 0,15 mV (treino) contra relativo, 0,18 x (p99 - p1) (servidor)
  filtro   o treino filtra o registo inteiro e corta as janelas; o servidor filtra um buffer de 25 s (margem de 5 s de cada lado)
           e usa os 15 s centrais (bordas do filtfilt)
  pol      polaridade do ECG decidida no registo (treino) ou em cada janela (servidor)
  ppg      correção de inversão do PPG (só existe no servidor)
  detR     portões do detetor de R: pisos absolutos em mV (treino) contra relação sinal/ruído (servidor)
  S_tudo   o servidor inteiro (glitch por buffer, filtro por buffer, polaridade por janela, inversão do PPG, SNR)
O sinal do MIMIC entra como está (mV e unidades do PLETH, a 125 Hz): as portas do sensor do servidor (média do IR, desvio
mínimo, em contagens do ADC) ficam de fora, porque não existem no treino. A frente do servidor está reproduzida aqui com
as funções do `comum.py` e confere-se que, a 100 Hz sobre sinal de bancada, dá exatamente o mesmo que `extrair_features_janela`.
Escolha dos registos: um segmento por doente do conjunto de teste (lista do absoluto), o de mais janelas aceites no treino
nos primeiros 40 min, e esses 40 min.
  /home/filipe/Desktop/Tese/algoritmo-ml/venv/bin/python 20_extracao_servidor_vs_treino.py [n_registos]
Escreve 20_extracao_servidor_vs_treino.json."""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import find_peaks
from scipy.stats import skew

AQUI = os.path.dirname(os.path.abspath(__file__))
S = "/home/filipe/Desktop/Tese/server"
sys.path.insert(0, S)
import comum  # noqa: E402

A = "/home/filipe/Desktop/Tese/algoritmo-ml"
NB = f"{A}/05_EXTRACAO_FEATURES/_notebooks_originais/Extração_features.ipynb"
MIMIC = "/media/filipe/Dados/MIMIC_DATASET_CONSOLIDADO"
FEATS = "/media/filipe/Dados/Features"
IDS = {l.strip() for l in open(f"{A}/RESULTADOS_RELATORIO/codigo_dependencias/particao_sujeitos/pacientes_holdout.txt") if l.strip()}
N_REG = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MIN_S = 40 * 60
JAN, PASSO = 15, 5

# ── o código do treino, tal e qual ───────────────────────────────────────────────────────────────────────────
nb = json.load(open(NB, encoding="utf-8"))
src = "".join(nb["cells"][3]["source"])
T = {}
exec(src[:src.index("def processar_linha_individual")], T)       # só as definições
T_glitch, T_proc, T_lp, T_hp = T["remover_picos_isolados"], T["processar_janela_teste1"], T["butter_lowpass_filter"], T["butter_highpass_filter"]


def detectar_r_treino(win_ecg, fs):
    """Deteção de R do treino (linhas de processar_janela_teste1 até aos pisos de amplitude, em mV)."""
    envelope = comum._envelope_qrs(win_ecg, fs)

    def env_local(p):
        return np.max(envelope[max(0, p - int(0.05 * fs)): min(len(envelope), p + int(0.05 * fs))])
    cand, _ = find_peaks(win_ecg, height=0.15 * np.max(win_ecg))
    r = []
    for p in cand:
        if r and (p - r[-1]) < int(fs * 0.4):
            ep, el = env_local(p), env_local(r[-1])
            if ep > el * 1.15 or (ep > el / 1.15 and win_ecg[p] > win_ecg[r[-1]]):
                r[-1] = p
        else:
            r.append(int(p))
    r = np.array(r)
    if len(r) >= 3:
        envs, amps = np.array([env_local(p) for p in r]), win_ecg[r]
        me, ma = np.median(envs), np.median(amps)
        if me > 0 and ma > 0:
            r = r[~((envs < 0.15 * me) & (amps < 0.40 * ma))]
    if len(r) < 8:
        return None
    amp = np.median(win_ecg[r])
    if amp < 0.08:
        return None
    rr = np.diff(r) / fs
    if amp < 0.15 and np.std(rr) / np.mean(rr) > 0.25:
        return None
    if np.max(np.abs(win_ecg)) > 4 * amp:
        return None
    return r


MAPA = {"pat_mean": "pat_mean", "hr_mean": "hr_mean", "qrs_duration_mean": "qrs_duration", "r_amplitude_mean": "r_amplitude",
        "qrs_amplitude_mean": "qrs_amplitude", "qrs_area_mean": "qrs_area", "w50_mean": "w50", "pulse_area_mean": "area_mean",
        "ppg_valley_intensity_mean": "ppg_valley_intensity", "ppg_ac_amplitude_mean": "ppg_ac_amplitude", "crest_time_mean": "crest_time",
        "decay_time_mean": "decay_time", "alpha_mean": "alpha_womersley", "alpha_b_mean": "alpha_womersley_b",
        "qt_interval_mean": "qt_interval", "tq_interval_mean": "tq_interval", "qtc_interval_mean": "qtc_interval",
        "sdi_interaction_mean": "sdi_interaction"}
NOMES = list(MAPA.values())


def feats_treino(win_ecg, win_ppg, fs):
    r = T_proc(win_ecg, win_ppg, fs)
    if r is None or r[1] is None:
        return None
    return {MAPA[k]: v for k, v in r[1].items() if k in MAPA}


# ── configurações ────────────────────────────────────────────────────────────────────────────────────────────
BASE = dict(glitch="T", filtro="T", pol="T", ppg=False, detR="T")
CFG = {"TT": BASE}
for k, v in (("glitch", "S"), ("filtro", "S"), ("pol", "S"), ("ppg", True), ("detR", "S")):
    CFG[k] = {**BASE, k: v}
CFG["pol_sessao"] = {**BASE, "pol": "sessao"}               # decisão da sessão (voto da maioria das janelas até agora)
CFG["ppg_sessao"] = {**BASE, "ppg": "sessao"}
CFG["glitch_sessao150"] = {**BASE, "glitch": "sessao150"}   # escala dos picos isolados: p99-p1 dos primeiros 150 s do registo
CFG["glitch_sessao25"] = {**BASE, "glitch": "sessao25"}     # idem, do primeiro buffer de 25 s
CFG["S_tudo"] = dict(glitch="Sbuf", filtro="S", pol="S", ppg=True, detR="S")
CFG["S_sessao"] = dict(glitch="Sbuf", filtro="S", pol="sessao", ppg="sessao", detR="S")    # o servidor (estado da sessão)


def glitch_com_escala(sinal, limiar):
    """O mesmo que comum.remover_picos_isolados, com o limiar absoluto dado (escala fixa da sessão)."""
    s = np.asarray(sinal, dtype=float)
    d_ant, d_prox, d_viz = np.abs(s[1:-1] - s[:-2]), np.abs(s[1:-1] - s[2:]), np.abs(s[:-2] - s[2:])
    idx = np.where((d_ant > limiar) & (d_prox > limiar) & (d_viz < limiar * 0.5))[0] + 1
    limpo = s.copy()
    limpo[idx] = (s[idx - 1] + s[idx + 1]) / 2
    return limpo


class Registo:
    def __init__(self, ecg, ppg, fs):
        self.fs, self.ecg, self.ppg = fs, ecg, ppg
        self.ppg_f = T_lp(ppg, 8, fs)
        self.ecg_g = {"T": T_glitch(ecg), "S": comum.remover_picos_isolados(ecg)}
        self.ecg_f = {k: T_hp(v, 0.5, fs) for k, v in self.ecg_g.items()}
        self.sinal = {k: (-1.0 if skew(v) < 0 else 1.0) for k, v in self.ecg_f.items()}
        self.n = int((len(ecg) - JAN * fs) / (PASSO * fs)) + 1
        esc = lambda x: np.percentile(x, 99) - np.percentile(x, 1)
        self.escala = {"sessao150": esc(ecg[: 150 * fs]), "sessao25": esc(ecg[: 25 * fs])}
        for k, e in self.escala.items():
            self.ecg_g[k] = glitch_com_escala(ecg, 0.18 * e)
            self.ecg_f[k] = T_hp(self.ecg_g[k], 0.5, fs)
        self.votos = {}

    def voto(self, chave, negativo):
        v = self.votos.setdefault(chave, [0, 0])
        v[0 if negativo else 1] += 1
        return -1.0 if v[0] > v[1] else 1.0


def janela(R, w, cfg, contagem, nome=None):
    fs = R.fs
    ini, fim, marg = w * PASSO * fs, w * PASSO * fs + JAN * fs, PASSO * fs
    if cfg["glitch"] == "Sbuf":                                       # como o servidor: glitch no buffer de 25 s
        buf = comum.remover_picos_isolados(R.ecg[ini - marg: fim + marg])
        e = comum.butter_highpass_filter(buf, 0.5, fs)[marg: marg + JAN * fs]
    elif cfg["filtro"] == "S" and cfg["glitch"] in R.ecg_g:           # buffer de 25 s filtrado, 15 s centrais
        buf = R.ecg_g[cfg["glitch"]][ini - marg: fim + marg]
        e = comum.butter_highpass_filter(buf, 0.5, fs)[marg: marg + JAN * fs]
    else:                                                             # registo inteiro filtrado
        e = R.ecg_f[cfg["glitch"]][ini:fim]
    if cfg["filtro"] == "S":
        p = comum.butter_lowpass_filter(R.ppg[ini - marg: fim + marg], 8.0, fs)[marg: marg + JAN * fs]
    else:
        p = R.ppg_f[ini:fim]
    if cfg["pol"] == "T":
        e = e * R.sinal["T"]
    elif cfg["pol"] == "sessao":
        e = e * R.voto((nome, "ecg"), skew(e) < 0)
    elif skew(e) < 0:
        e = -e
        contagem["pol_janela_inverte"] += 1
    if cfg["ppg"] == "sessao":
        sg = R.voto((nome, "ppg"), skew(np.diff(p)) < 0)
        p = p * sg
        contagem["ppg_inverte"] += int(sg < 0)
    elif cfg["ppg"] and skew(np.diff(p)) < 0:
        p = -p
        contagem["ppg_inverte"] += 1
    if cfg["detR"] == "T" and cfg["glitch"] == "T" and cfg["filtro"] == "T" and cfg["pol"] == "T" and not cfg["ppg"]:
        pass
    return comum.processar_janela_absoluto(e, p, fs, detectar_r=detectar_r_treino if cfg["detR"] == "T" else None), e, p


# ── confere a frente do servidor reproduzida contra a função real, a 100 Hz, sobre sinal de bancada ────────────
def conferir_replica():
    iguais = total = 0
    for f in sorted(glob.glob("/home/filipe/Desktop/Tese/arduino/Dados/ECG+PPG/Configuração 1/Todos_*_bruto_*.csv"))[:4]:
        d = pd.read_csv(f)
        t = (d.time_ms.values - d.time_ms.values[0]) / 1000
        tu = np.arange(0, t[-1], 1 / 100)
        ecg, ir = np.interp(tu, t, d.ecg.values), np.interp(tu, t, d.ir.values)
        for i0 in range(0, len(tu) - 2500, 500):
            snap = [{"ecg": float(e), "ir": float(i), "red": 0.0} for e, i in zip(ecg[i0:i0 + 2500], ir[i0:i0 + 2500])]
            real = comum.extrair_features_janela(snap)
            buf_e = comum.remover_picos_isolados(ecg[i0:i0 + 2500])
            e = comum.butter_highpass_filter(buf_e, 0.5, 100)[500:2000]
            p = comum.butter_lowpass_filter(ir[i0:i0 + 2500], 8.0, 100)[500:2000]
            if skew(e) < 0:
                e = -e
            if skew(np.diff(p)) < 0:
                p = -p
            if np.mean(ir[i0:i0 + 2500]) < 10_000 or np.std(p) < 10:
                rep = None
            else:
                rep = comum.processar_janela_absoluto(e, p, 100)
            total += 1
            iguais += (real == rep)
    return dict(janelas=total, iguais=iguais)


def escolher_registos():
    cons = pd.read_csv(f"{FEATS.rsplit('/', 1)[0]}/features_todos_consolidado.csv")
    cons["caminho"] = cons.absolute_path.str.replace("/media/filipe/Dados/MIMIC_DATASET_CONSOLIDADO", MIMIC)
    conta = []
    for g in ("homens", "mulheres"):
        d = pd.read_csv(f"{FEATS}/features_{g}_baseline.csv", usecols=["subject_id", "segment_name", "window_idx"])
        d = d[d.subject_id.astype(str).isin(IDS) & (d.window_idx < MIN_S // PASSO)]
        conta.append(d.groupby(["subject_id", "segment_name"]).size().rename("n").reset_index())
    c = pd.concat(conta).sort_values("n", ascending=False).drop_duplicates("subject_id")
    c = c.merge(cons[["segment", "caminho"]], left_on="segment_name", right_on="segment", how="left")
    c = c[c.caminho.notna() & c.n.ge(60)]
    return c.head(N_REG)


def carregar(caminho):
    hdr = wfdb.rdheader(caminho)
    nomes = [s.upper() for s in hdr.sig_name]
    if "II" not in nomes:
        return None
    i_e = nomes.index("II")
    i_p = next((i for i, s in enumerate(nomes) if "PLETH" in s or "PPG" in s), None)
    if i_p is None:
        return None
    rec = wfdb.rdrecord(caminho, channels=[i_e, i_p], sampto=int(MIN_S * hdr.fs))
    e, p = rec.p_signal[:, 0].astype(float), rec.p_signal[:, 1].astype(float)
    if np.isnan(e).any() or np.isnan(p).any():
        e, p = pd.Series(e).interpolate().ffill().bfill().values, pd.Series(p).interpolate().ffill().bfill().values
    return Registo(e, p, int(hdr.fs))


if __name__ == "__main__":
    t0 = time.time()
    rep = conferir_replica()
    print("réplica da frente do servidor contra comum.extrair_features_janela (bancada, 100 Hz):", rep, flush=True)
    assert rep["iguais"] == rep["janelas"], rep
    regs = escolher_registos()
    print(len(regs), "registos escolhidos", flush=True)
    linhas = []
    cont = {k: dict(pol_janela_inverte=0, ppg_inverte=0) for k in CFG}
    n_reg = 0
    for _, r in regs.iterrows():
        try:
            R = carregar(r.caminho)
        except Exception as e:
            print("falhou", r.caminho, str(e)[:60])
            continue
        if R is None or R.n < 5:
            continue
        n_reg += 1
        for w in range(1, R.n - 1):                                    # com margem de 5 s de cada lado, como no servidor
            ini, fim = w * PASSO * R.fs, w * PASSO * R.fs + JAN * R.fs
            lin = {"subject_id": r.subject_id, "segmento": r.segment_name, "w": w}
            # o treino, a sério: as funções do caderno sobre o registo inteiro filtrado
            e_t = (R.ecg_f["T"] * R.sinal["T"])[ini:fim]
            ref = feats_treino(e_t, R.ppg_f[ini:fim], R.fs)
            lin["treino"] = ref
            for nome, cfg in CFG.items():
                f, _, _ = janela(R, w, cfg, cont[nome], nome)
                lin[nome] = f
            linhas.append(lin)
        print(f"  registo {n_reg}/{len(regs)} {r.segment_name}: {R.n} janelas, {time.time() - t0:.0f} s", flush=True)

    # ── resultados ────────────────────────────────────────────────────────────────────────────────────────────
    saida = {"registos": n_reg, "sujeitos": int(len({l['subject_id'] for l in linhas})), "janelas": len(linhas),
             "replica_bancada": rep, "contagens": cont, "variantes": {}}
    ref_ok = [l for l in linhas if l["treino"] is not None]
    saida["janelas_aceites_treino"] = len(ref_ok)
    # TT tem de ser igual ao treino
    ig = sum(1 for l in linhas if (l["TT"] is None) == (l["treino"] is None) and (l["TT"] is None or all(
        abs((l["TT"].get(k) if l["TT"].get(k) is not None else np.nan) - (l["treino"].get(k) if l["treino"].get(k) is not None else np.nan)) < 1e-9
        or (l["TT"].get(k) is None and l["treino"].get(k) is None) for k in NOMES)))
    saida["TT_igual_ao_treino"] = f"{ig} de {len(linhas)}"
    for nome in CFG:
        if nome == "TT":
            continue
        ambos = [l for l in linhas if l["treino"] is not None and l[nome] is not None]
        so_t = sum(1 for l in linhas if l["treino"] is not None and l[nome] is None)
        so_v = sum(1 for l in linhas if l["treino"] is None and l[nome] is not None)
        v = dict(aceites=sum(1 for l in linhas if l[nome] is not None), so_treino=so_t, so_variante=so_v, comuns=len(ambos), features={})
        for k in NOMES:
            a = np.array([l["treino"].get(k) if l["treino"].get(k) is not None else np.nan for l in ambos], dtype=float)
            b = np.array([l[nome].get(k) if l[nome].get(k) is not None else np.nan for l in ambos], dtype=float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 10:
                continue
            d = np.abs(b[m] - a[m])
            esc = np.maximum(np.abs(a[m]), 1e-9)
            v["features"][k] = dict(n=int(m.sum()), r=round(float(np.corrcoef(a[m], b[m])[0, 1]), 6) if np.std(a[m]) > 0 and np.std(b[m]) > 0 else None,
                                    dif_mediana_rel=float(np.median(d / esc)), dif_p95_rel=float(np.quantile(d / esc, 0.95)),
                                    pct_dif_acima_1e3=round(100 * float(np.mean(d / esc > 1e-3)), 2), dif_max_abs=float(d.max()))
        saida["variantes"][nome] = v
    json.dump(saida, open(f"{AQUI}/20_extracao_servidor_vs_treino.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in saida.items() if k != "variantes"}, indent=1))
    for nome, v in saida["variantes"].items():
        ps = [f for f in v["features"].values()]
        print(f"{nome:8s} aceites {v['aceites']:7d} só treino {v['so_treino']:6d} só variante {v['so_variante']:6d} comuns {v['comuns']:7d} | "
              f"r mínimo {min((x['r'] for x in ps if x['r'] is not None), default=None)} | "
              f"mediana da dif. relativa máx {max((x['dif_mediana_rel'] for x in ps), default=0):.2e} | % com dif>1e-3 máx {max((x['pct_dif_acima_1e3'] for x in ps), default=0):.2f}")
    print(f"{time.time() - t0:.0f} s")
