#!/usr/bin/env python3
"""analisar_ensaio_ppg.py — compara a FC, a SpO2 e a FR do servidor com as medições apontadas à mão num ensaio com a
pulseira só-PPG, e gera a figura e os números para o relatório.

ENTRADAS
  --gravacao  CSV do gravar_mqtt.py (por omissão, o mais recente em Dados/gravacoes_mqtt). Dá o período do ensaio
              (rx_utc) e o sinal em bruto.
  --registo   folha preenchida durante o ensaio, de preferência com o anotar_ensaio.py (por omissão, o
              ensaios/registo_*.csv mais recente); à mão, segue o modelo ensaios/modelo_registo_ppg.csv, uma linha por medição:
                hora            hora LOCAL do relógio do PC, HH:MM ou HH:MM:SS
                respiracoes     respirações contadas no minuto que COMEÇA nessa hora (vazio se não contou)
                fc_smartband    FC lida na smartband nessa hora (vazio se não leu)
                spo2_smartband  SpO2 lida na smartband nessa hora (vazio se não leu)
                notas           texto livre (ex.: "respiração lenta")
              Aceita vírgula ou ponto e vírgula como separador (Excel em português usa ';').
  Os valores do servidor vêm da tabela medicao (FC, SpO2 e FR a ~1 Hz, timestamp em UTC), para a pulseira e o período
  da gravação.

EMPARELHAMENTO (cada estimativa do servidor olha para trás no tempo)
  FC e SpO2: a via rápida usa os últimos 8 s de sinal (comum.MIN_AMOSTRAS = 800 a 100 Hz). Compara-se a leitura da
             smartband à hora h com a mediana do servidor entre h - 10 s e h + 10 s.
  FR:        o servidor usa os últimos 90 s (frequencia_respiratoria.JANELA_MAX_S). O minuto contado [h, h + 60 s[ compara-se
             com a mediana do servidor entre h + 55 s e h + 65 s, ou seja, a estimativa que acaba no fim desse minuto
             (cobre o minuto contado e os 30 s antes dele; por isso convém manter o mesmo ritmo uns minutos seguidos).

SAÍDAS (em ensaios/ppg_<nome da gravação>/)
  ensaio_ppg.png       FC, SpO2 e FR ao longo do ensaio: servidor (linha) contra a referência (pontos), texto em inglês
  pares.csv            cada medição de referência com o valor do servidor emparelhado
  resultados.json      por grandeza: n, viés (servidor - referência), MAE, desvio padrão do erro, erro máximo; cobertura
                       (fração dos segundos com valor no servidor); janelas aceites pelos portões (recalculado do sinal em
                       bruto com extracao_ppg_classificador, como no worker); buracos na gravação
  resumo.md            o mesmo em tabela

Uso:
  /home/filipe/Desktop/Tese/algoritmo-ml/venv/bin/python analisar_ensaio_ppg.py
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

AQUI = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(os.path.dirname(os.path.dirname(AQUI)), "servidor", "workers")
sys.path.insert(0, SERVER)
import extracao_ppg_classificador as epp          # mesmos portões do worker só-PPG

FUSO = ZoneInfo("Europe/Lisbon")
FS = 100
JANELA, MARGEM, PASSO = 1500, 500, 500   # 15 s com 5 s de margem de cada lado, passo de 5 s (comum.py)
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://hospital:hospital_pass@localhost:5432/hospital")


def ler_registo(caminho, data_local):
    r = pd.read_csv(caminho, sep=None, engine="python", dtype=str).fillna("")
    r.columns = [c.strip().lower() for c in r.columns]
    for c in ("hora", "respiracoes", "fc_smartband", "spo2_smartband"):
        if c not in r.columns:
            sys.exit(f"erro: falta a coluna '{c}' em {caminho}")
    if "notas" not in r.columns:
        r["notas"] = ""
    r = r[r.hora.str.strip() != ""].copy()
    num = lambda s: pd.to_numeric(s.str.replace(",", ".").str.strip(), errors="coerce")
    for c in ("respiracoes", "fc_smartband", "spo2_smartband"):
        r[c] = num(r[c])

    def para_utc(h):
        partes = [int(x) for x in h.strip().split(":")]
        hh, mm, ss = (partes + [0])[:3]
        local = datetime(data_local.year, data_local.month, data_local.day, hh, mm, ss, tzinfo=FUSO)
        return local.astimezone(timezone.utc).replace(tzinfo=None)
    r["hora_utc"] = r.hora.map(para_utc)
    return r.sort_values("hora_utc").reset_index(drop=True)


def mediana(med, col, a, b):
    v = med.loc[(med.timestamp >= a) & (med.timestamp <= b), col].dropna()
    return float(v.median()) if len(v) else np.nan


def estatisticas(serv, ref):
    ok = ~(np.isnan(serv) | np.isnan(ref))
    e = serv[ok] - ref[ok]
    if not ok.any():
        return dict(n=0)
    return dict(n=int(ok.sum()), vies=round(float(e.mean()), 2), mae=round(float(np.abs(e).mean()), 2),
                dp_erro=round(float(e.std(ddof=1)), 2) if ok.sum() > 1 else None,
                erro_max=round(float(np.abs(e).max()), 2))


def portoes(g):
    """Janelas de 15 s com passo de 5 s sobre o sinal em bruto, como no worker (filtro sobre 25 s, janela central)."""
    ir = g.ir.to_numpy(dtype=float)
    n_ac = n = 0
    for i in range(0, len(ir) - JANELA - 2 * MARGEM + 1, PASSO):
        snap = [{"ir": x} for x in ir[i:i + JANELA + 2 * MARGEM]]
        n += 1
        n_ac += epp.extrair_features_janela_ppg(snap, FS, JANELA, MARGEM) is not None
    return dict(janelas=n, aceites=n_ac, fracao_aceites=round(n_ac / n, 3) if n else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gravacao", default=None)
    ap.add_argument("--registo", default=None, help="por omissão, o registo_*.csv mais recente em ensaios/")
    ap.add_argument("--pulseira", default="PULSEIRA_01")
    a = ap.parse_args()

    a.registo = a.registo or max(glob.glob(os.path.join(AQUI, "ensaios", "registo_*.csv")), key=os.path.getmtime)
    print("registo:", a.registo)
    grav = a.gravacao or max(glob.glob(os.path.join(AQUI, "Dados", "gravacoes_mqtt", "*.csv")), key=os.path.getmtime)
    nome = os.path.splitext(os.path.basename(grav))[0]
    saida = os.path.join(AQUI, "ensaios", f"ppg_{nome}")
    os.makedirs(saida, exist_ok=True)

    g = pd.read_csv(grav)
    g["rx"] = pd.to_datetime(g.rx_utc, utc=True).dt.tz_convert(None)
    t0, t1 = g.rx.min(), g.rx.max()
    dt = np.diff(g.time_ms.to_numpy())
    buracos = int(((dt > 50) | (dt < -1000)).sum())

    with create_engine(DATABASE_URL).connect() as c:
        med = pd.read_sql(text("SELECT timestamp, bpm, spo2, resp FROM medicao WHERE pulseira_id = :p "
                               "AND timestamp BETWEEN :a AND :b ORDER BY timestamp"),
                          c, params=dict(p=a.pulseira, a=t0 - timedelta(seconds=5), b=t1 + timedelta(seconds=5)))
    if med.empty:
        sys.exit(f"erro: a tabela medicao não tem valores de {a.pulseira} entre {t0} e {t1} UTC "
                 "(a pulseira estava atribuída a um doente?)")

    reg = ler_registo(a.registo, t0.replace(tzinfo=timezone.utc).astimezone(FUSO).date())
    s10, s55, s65 = timedelta(seconds=10), timedelta(seconds=55), timedelta(seconds=65)
    reg["fc_servidor"] = [mediana(med, "bpm", h - s10, h + s10) for h in reg.hora_utc]
    reg["spo2_servidor"] = [mediana(med, "spo2", h - s10, h + s10) for h in reg.hora_utc]
    reg["fr_servidor"] = [mediana(med, "resp", h + s55, h + s65) for h in reg.hora_utc]
    reg.to_csv(os.path.join(saida, "pares.csv"), index=False)

    dur_s = (t1 - t0).total_seconds()
    seg = med.set_index("timestamp").resample("1s").first()
    res = dict(
        gravacao=grav, registo=os.path.abspath(a.registo), pulseira=a.pulseira,
        inicio_utc=str(t0), fim_utc=str(t1), duracao_min=round(dur_s / 60, 1),
        amostras=int(len(g)), fs_media_hz=round(len(g) / dur_s, 1) if dur_s else None, buracos_gravacao=buracos,
        fc=estatisticas(reg.fc_servidor.to_numpy(float), reg.fc_smartband.to_numpy(float)),
        spo2=estatisticas(reg.spo2_servidor.to_numpy(float), reg.spo2_smartband.to_numpy(float)),
        fr=estatisticas(reg.fr_servidor.to_numpy(float), reg.respiracoes.to_numpy(float)),
        cobertura={k: round(float(seg[c].notna().mean()), 3) for k, c in (("fc", "bpm"), ("spo2", "spo2"), ("fr", "resp"))},
        portoes=portoes(g),
    )
    json.dump(res, open(os.path.join(saida, "resultados.json"), "w"), indent=1, ensure_ascii=False)

    with open(os.path.join(saida, "resumo.md"), "w", encoding="utf-8") as f:
        f.write(f"# Ensaio PPG — {nome}\n\n{res['duracao_min']} min, {res['fs_media_hz']} Hz médios, "
                f"{buracos} buracos na gravação. Janelas aceites pelos portões: {res['portoes']['aceites']} de "
                f"{res['portoes']['janelas']} ({res['portoes']['fracao_aceites']}).\n\n"
                "| | n | viés | MAE | DP do erro | erro máx. | cobertura |\n|---|---|---|---|---|---|---|\n")
        for k, nomek in (("fc", "FC (bpm)"), ("spo2", "SpO2 (%)"), ("fr", "FR (resp/min)")):
            e = res[k]
            f.write(f"| {nomek} | {e.get('n')} | {e.get('vies', '')} | {e.get('mae', '')} | {e.get('dp_erro', '')} | "
                    f"{e.get('erro_max', '')} | {res['cobertura'][k]} |\n")

    # figura (texto em inglês)
    mins = lambda ts: (ts - t0).dt.total_seconds() / 60 if hasattr(ts, "dt") else (ts - t0).total_seconds() / 60
    xm = mins(med.timestamp)
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    for k, (col, ref, rot) in enumerate((("bpm", "fc_smartband", "HR (bpm)"), ("spo2", "spo2_smartband", "SpO$_2$ (%)"),
                                         ("resp", "respiracoes", "RR (breaths/min)"))):
        ax[k].plot(xm, med[col], color="#555555", lw=0.9, label="device (server)")
        r = reg[reg[ref].notna()]
        if col == "resp":       # cada contagem cobre um minuto: segmento horizontal
            for i, (_, l) in enumerate(r.iterrows()):
                x = mins(l.hora_utc)
                ax[k].plot([x, x + 1], [l[ref]] * 2, color="#c0392b", lw=2.5,
                           label="manual count (1 min)" if i == 0 else None)
        else:
            ax[k].plot(mins(r.hora_utc), r[ref], "o", color="#c0392b", ms=5, label="smartband")
        ax[k].set_ylabel(rot)
        if col == "resp":       # escala com folga: com valores parecidos, o eixo automático exagera as diferenças
            v = pd.concat([med[col], reg[ref]]).dropna()
            if len(v):
                ax[k].set_ylim(max(0, v.min() - 5), v.max() + 5)
        ax[k].legend(loc="upper right", fontsize=8)
        ax[k].grid(ls=":", alpha=0.6)
    for _, l in reg[reg.notas.str.strip() != ""].iterrows():
        for x_ in ax:
            x_.axvline(mins(l.hora_utc), color="#1f5fa8", ls="--", lw=0.7, alpha=0.6)
        ax[0].text(mins(l.hora_utc), 1.02, l.notas, transform=ax[0].get_xaxis_transform(), fontsize=7,
                   color="#1f5fa8", rotation=30, ha="left", va="bottom")
    ax[2].set_xlabel("Time since start of recording (min)")
    fig.tight_layout()
    fig.savefig(os.path.join(saida, "ensaio_ppg.png"), dpi=200, bbox_inches="tight")
    print(open(os.path.join(saida, "resumo.md"), encoding="utf-8").read())
    print("saída em", saida)


if __name__ == "__main__":
    main()
