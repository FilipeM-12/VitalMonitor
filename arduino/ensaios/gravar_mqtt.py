#!/usr/bin/env python3
"""gravar_mqtt.py — grava em CSV o sinal em bruto que as pulseiras publicam no mosquitto.

É mais um cliente do broker, ao lado dos workers: subscreve pulseiras/raw/<ID> e escreve cada amostra numa linha. Não
interfere com o processamento (os workers continuam a receber tudo) e pode ser ligado e desligado a qualquer altura,
sem reiniciar nada.

Ao arrancar pede o nome do paciente (ou recebe-o com --paciente), que entra no nome do ficheiro.

Ficheiro: <saida>/<paciente>_<PULSEIRA_ID>_<AAAAMMDD_HHMMSS>.csv, um por pulseira e por execução, com as colunas
    rx_utc    hora de chegada do lote ao PC (UTC, ISO 8601), a mesma referência do timestamp da tabela medicao
    time_ms   millis() da pulseira (relógio da própria pulseira, reinicia num reboot)
    ecg, ir, red, temp    como no payload (no firmware só-PPG, ecg = 0)
As colunas time_ms,ecg,ir,red,temp são as dos ficheiros de bancada (arduino/Dados), por isso os scripts de análise
existentes leem estes ficheiros sem alterações.

Uso (com o venv do algoritmo-ml, que tem o paho-mqtt):
    /home/filipe/Desktop/Tese/algoritmo-ml/venv/bin/python gravar_mqtt.py                    # PULSEIRA_01, até Ctrl+C
    /home/filipe/Desktop/Tese/algoritmo-ml/venv/bin/python gravar_mqtt.py --duracao 15       # para sozinho aos 15 min
    /home/filipe/Desktop/Tese/algoritmo-ml/venv/bin/python gravar_mqtt.py --pulseira '#'     # todas as pulseiras

A cada 10 s mostra quantas amostras chegaram, a frequência de amostragem real e os buracos no time_ms (lotes perdidos).
"""
import argparse
import csv
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

AQUI = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pulseira", default="PULSEIRA_01", help="ID da pulseira, ou '#' para todas")
    ap.add_argument("--broker", default=os.environ.get("MQTT_BROKER", "localhost"))
    ap.add_argument("--porta", type=int, default=int(os.environ.get("MQTT_PORT", "1883")))
    # credenciais dos workers (comum.py); o broker não aceita ligações anónimas
    ap.add_argument("--utilizador", default=os.environ.get("MQTT_USER", "worker"))
    ap.add_argument("--password", default=os.environ.get("MQTT_PASS", "worker_pass"))
    ap.add_argument("--saida", default=os.path.join(AQUI, "Dados", "gravacoes_mqtt"))
    ap.add_argument("--duracao", type=float, default=None, help="minutos de gravação (por omissão, até Ctrl+C)")
    ap.add_argument("--paciente", default=None, help="nome do paciente (se faltar, é pedido ao arrancar)")
    a = ap.parse_args()

    nome = a.paciente
    while not nome:
        nome = input("Nome do paciente: ").strip()
    # só letras (com acentos), algarismos, '-' e '_' no nome do ficheiro; espaços passam a '_'
    nome = "".join(c if c.isalnum() or c in "-_" else "_" for c in nome.replace(" ", "_")).strip("_")

    os.makedirs(a.saida, exist_ok=True)
    arranque = datetime.now().strftime("%Y%m%d_%H%M%S")
    ficheiros, estado, lock = {}, {}, threading.Lock()

    def escritor(pid):
        if pid not in ficheiros:
            caminho = os.path.join(a.saida, f"{nome}_{pid}_{arranque}.csv")
            f = open(caminho, "w", newline="")
            w = csv.writer(f)
            w.writerow(["rx_utc", "time_ms", "ecg", "ir", "red", "temp"])
            ficheiros[pid] = (f, w)
            estado[pid] = dict(n=0, n_janela=0, ultimo_t=None, buracos=0, t0=time.time(), caminho=caminho)
            print(f"[{pid}] a gravar em {caminho}", flush=True)
        return ficheiros[pid][1], estado[pid]

    def on_connect(cliente, userdata, flags, rc, properties=None):
        if rc != 0:
            print(f"erro: o broker recusou a ligação (rc={rc}); verificar --utilizador/--password", file=sys.stderr)
            return
        topico = f"pulseiras/raw/{a.pulseira}"
        cliente.subscribe(topico, qos=0)
        print(f"ligado a {a.broker}:{a.porta}, à escuta em {topico}", flush=True)

    def on_message(cliente, userdata, msg):
        rx = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        try:
            d = json.loads(msg.payload)
            pid = d.get("pulseira_id") or msg.topic.split("/")[2]
            amostras = d["amostras"]
        except Exception as e:
            print(f"lote ignorado em {msg.topic}: {e!r}", file=sys.stderr)
            return
        with lock:
            w, e = escritor(pid)
            for s in amostras:
                t = s.get("t")
                # buraco: mais de 50 ms entre amostras seguidas (a 100 Hz são 10 ms); um salto grande para trás é reboot
                if e["ultimo_t"] is not None and t is not None and (t - e["ultimo_t"] > 50 or t < e["ultimo_t"] - 1000):
                    e["buracos"] += 1
                e["ultimo_t"] = t
                w.writerow([rx, t, s.get("ecg"), s.get("ir"), s.get("red"), s.get("temp")])
            e["n"] += len(amostras)
            e["n_janela"] += len(amostras)

    cliente = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                          client_id=f"gravador-{os.getpid()}", clean_session=True)   # nunca o ID da pulseira
    cliente.username_pw_set(a.utilizador, a.password)
    cliente.on_connect = on_connect
    cliente.on_message = on_message
    try:
        cliente.connect(a.broker, a.porta, keepalive=30)
    except OSError as e:
        sys.exit(f"erro: não foi possível ligar a {a.broker}:{a.porta} ({e})")

    parar = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: parar.set())
    signal.signal(signal.SIGTERM, lambda *_: parar.set())
    cliente.loop_start()
    inicio = time.time()
    try:
        while not parar.wait(10):
            with lock:
                if not estado:
                    print("ainda sem dados da pulseira…", flush=True)
                for pid, e in estado.items():
                    for f, _ in [ficheiros[pid]]:
                        f.flush()
                    print(f"[{pid}] {e['n']} amostras, {e['n_janela'] / 10:.0f} Hz nos últimos 10 s, "
                          f"{e['buracos']} buracos, {(time.time() - e['t0']) / 60:.1f} min", flush=True)
                    e["n_janela"] = 0
            if a.duracao and time.time() - inicio >= a.duracao * 60:
                break
    finally:
        cliente.loop_stop()
        cliente.disconnect()
        with lock:
            for pid, (f, _) in ficheiros.items():
                f.close()
                e = estado[pid]
                print(f"[{pid}] terminado: {e['n']} amostras ({e['n'] / 100 / 60:.1f} min a 100 Hz), "
                      f"{e['buracos']} buracos -> {e['caminho']}")


if __name__ == "__main__":
    main()
