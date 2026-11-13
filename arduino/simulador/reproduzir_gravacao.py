#!/usr/bin/env python3
"""reproduzir_gravacao.py — volta a publicar no broker uma gravação do gravar_mqtt.py, com o sinal e o time_ms originais.

O simulador_pulseira.py gera o seu próprio relógio a 100 Hz. Esta gravação foi feita a ~104 Hz e o worker mede a cadência
real a partir do time_ms, por isso aqui vai cada lote tal como chegou (25 amostras, mesmo time_ms) e à mesma cadência
de chegada (rx_utc). Serve para avaliar uma gravação antiga com o código atual do servidor.

Uso:
  python reproduzir_gravacao.py FICHEIRO.csv --pulseira PULSEIRA_REPLAY_X [--velocidade 1]
A pulseira tem de estar atribuída a um doente na base de dados. Os valores ficam na tabela medicao com a hora do replay.
"""
import argparse
import json
import os
import time

import pandas as pd
import paho.mqtt.client as mqtt

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--pulseira", required=True)
ap.add_argument("--token", default=os.environ.get("AUTH_TOKEN", "segredo123"))
ap.add_argument("--broker", default="localhost")
ap.add_argument("--porta", type=int, default=1883)
ap.add_argument("--utilizador", default="pulseira")
ap.add_argument("--password", default="pulseira_pass")
ap.add_argument("--velocidade", type=float, default=1.0)
a = ap.parse_args()

g = pd.read_csv(a.csv)
g["rx"] = pd.to_datetime(g.rx_utc, utc=True)
lotes = [x for _, x in g.groupby("rx", sort=True)]
c = mqtt.Client(client_id=a.pulseira, clean_session=True)
c.username_pw_set(a.utilizador, a.password)
c.connect(a.broker, a.porta, keepalive=30)
c.loop_start()
topico = f"pulseiras/raw/{a.pulseira}"
t0, w0 = lotes[0].rx.iloc[0], time.monotonic()
print(f"{len(lotes)} lotes, {len(g)} amostras, {(lotes[-1].rx.iloc[0] - t0).total_seconds() / 60:.1f} min", flush=True)
inicio_utc = pd.Timestamp.utcnow()
for k, b in enumerate(lotes):
    atraso = (b.rx.iloc[0] - t0).total_seconds() / a.velocidade - (time.monotonic() - w0)
    if atraso > 0:
        time.sleep(atraso)
    am = [{"t": int(r.time_ms), "ecg": int(round(r.ecg)), "ir": int(round(r.ir)), "red": int(round(r.red)),
           "temp": round(float(r.temp), 2)} for r in b.itertuples()]
    c.publish(topico, json.dumps({"pulseira_id": a.pulseira, "token": a.token, "amostras": am}, separators=(",", ":")), qos=0)
    if k % 400 == 0:
        print(f"  lote {k}/{len(lotes)}", flush=True)
time.sleep(2)
c.loop_stop()
print("início do replay (UTC):", inicio_utc.isoformat(), "· fim:", pd.Timestamp.utcnow().isoformat())
