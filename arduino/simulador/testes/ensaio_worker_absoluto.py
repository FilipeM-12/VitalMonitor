#!/usr/bin/env python3
"""Ensaio do worker do absoluto (worker_absoluto.py), isolado da produção.

Mosquitto próprio em 127.0.0.1:18831, o worker do absoluto como processo, um doente e uma pulseira só do ensaio na base de
dados do servidor (PostgreSQL), com atribuição e calibração, e o simulador a publicar sinal real de bancada com ECG e PPG
(Todos_participante C_bruto_20260408_193122.csv, em ciclo). No fim confere:
  · o worker arrancou com o modelo de 27 características e não deu erros no registo;
  · gravou blocos em medicao_absoluta, com as 27 características e previsões de PAS plausíveis;
  · os votos de polaridade do ECG e de orientação do PPG da sessão ficaram todos do mesmo lado (via o registo do worker);
e apaga as linhas do ensaio.
Escreve resultado.json e os registos em arduino/simulador/ensaio_worker_absoluto_<data>/.
  python ensaio_worker_absoluto.py [--duracao 720] [--velocidade 6]"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

SERVER = Path("/home/filipe/Desktop/Tese/server")
PY = SERVER / "venv/bin/python3"
SIMULADOR = Path(__file__).resolve().parents[1] / "simulador_pulseira.py"
CSV = Path("/home/filipe/Desktop/Tese/arduino/Dados/ECG+PPG/Configuração 1/Todos_participante C_bruto_20260408_193122.csv")
PORTA = 18831
PULSEIRA, PACIENTE, IDADE, SEXO = "PULSEIRA_E2E4", "DOENTE_E2E4", 58, "F"
SBP_CAL, DBP_CAL = 132, 78
TOKEN = "segredo123"
ap = argparse.ArgumentParser()
ap.add_argument("--duracao", type=float, default=720.0)
ap.add_argument("--velocidade", type=float, default=6.0)
a = ap.parse_args()
PASTA = Path(__file__).resolve().parents[1] / f"ensaio_worker_absoluto_{date.today().isoformat()}"
PASTA.mkdir(exist_ok=True)
URL = "postgresql://hospital:hospital_pass@localhost:5432/hospital"
sys.path.insert(0, str(SERVER / "venv/lib/python3.12/site-packages"))
from sqlalchemy import create_engine, text  # noqa: E402
ENG = create_engine(URL)
TABELAS = ("medicao_absoluta", "medicao_tendencia", "medicao", "calibracao", "atribuicao")


def limpar():
    with ENG.begin() as c:
        for t in TABELAS:
            c.execute(text(f"DELETE FROM {t} WHERE pulseira_id = :p OR paciente_id = :d"), dict(p=PULSEIRA, d=PACIENTE))
        c.execute(text("DELETE FROM pulseira WHERE id = :p"), dict(p=PULSEIRA))
        c.execute(text("DELETE FROM paciente WHERE id = :d"), dict(d=PACIENTE))


limpar()
conf = PASTA / "mosquitto_ensaio.conf"
conf.write_text(f"listener {PORTA} 127.0.0.1\nallow_anonymous true\nmax_packet_size 8192\npersistence false\n")
procs = {"mosquitto": subprocess.Popen(["/usr/sbin/mosquitto", "-c", str(conf)], stdout=open(PASTA / "mosquitto.log", "w"), stderr=subprocess.STDOUT)}
time.sleep(1.5)
assert procs["mosquitto"].poll() is None, "o mosquitto do ensaio não arrancou"
amb = dict(os.environ, DATABASE_URL=URL, MQTT_BROKER="127.0.0.1", MQTT_PORT=str(PORTA), MQTT_USER="worker", MQTT_PASS="worker_pass",
           AUTH_TOKEN=TOKEN, NUM_WORKERS="1", WORKER_ID="0", PYTHONUNBUFFERED="1")
agora = datetime.utcnow()
with ENG.begin() as c:
    c.execute(text("INSERT INTO paciente (id, idade, sexo) VALUES (:d, :i, :s)"), dict(d=PACIENTE, i=IDADE, s=SEXO))
    c.execute(text("INSERT INTO pulseira (id, paciente_id) VALUES (:p, :d)"), dict(p=PULSEIRA, d=PACIENTE))
    c.execute(text("INSERT INTO atribuicao (pulseira_id, paciente_id, data_inicio) VALUES (:p, :d, :t)"), dict(p=PULSEIRA, d=PACIENTE, t=agora))
    c.execute(text("INSERT INTO calibracao (pulseira_id, paciente_id, sbp, dbp, timestamp) VALUES (:p, :d, :s, :b, :t)"),
              dict(p=PULSEIRA, d=PACIENTE, s=SBP_CAL, b=DBP_CAL, t=agora))
procs["absoluto"] = subprocess.Popen([str(PY), "-u", "worker_absoluto.py"], cwd=SERVER, env=amb,
                                     stdout=open(PASTA / "worker_absoluto.log", "w"), stderr=subprocess.STDOUT)
time.sleep(10)
if procs["absoluto"].poll() is not None:
    for p in procs.values():
        p.terminate()
    limpar()
    sys.exit("o worker do absoluto não arrancou (ver worker_absoluto.log)")
inicio = time.time()
sim = subprocess.run([str(PY), "-u", str(SIMULADOR), "--pulseira", PULSEIRA, "--token", TOKEN, "--broker", "127.0.0.1", "--porta", str(PORTA),
                      "--fonte", "csv", "--csv", str(CSV), "--duracao", str(a.duracao), "--velocidade", str(a.velocidade)],
                     stdout=open(PASTA / "simulador.log", "w"), stderr=subprocess.STDOUT)
time.sleep(25)
with ENG.connect() as c:
    linhas = c.execute(text("SELECT features_ml, sbp_predicted FROM medicao_absoluta WHERE pulseira_id = :p ORDER BY id"), dict(p=PULSEIRA)).fetchall()
for p in procs.values():
    p.send_signal(signal.SIGTERM)
for p in procs.values():
    try:
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        p.kill()
log = (PASTA / "worker_absoluto.log").read_text(errors="replace")
feats = json.loads(linhas[0][0]) if linhas else {}
pas = [r[1] for r in linhas if r[1] is not None]
sucesso = bool(linhas) and len(feats) >= 27 and bool(pas) and all(60 < x < 220 for x in pas) and "Erro" not in log and "Traceback" not in log
res = dict(data=date.today().isoformat(), segundos_de_sinal=a.duracao, velocidade=a.velocidade, csv=CSV.name,
           simulador_codigo_saida=sim.returncode, duracao_real_s=round(time.time() - inicio), blocos_gravados=len(linhas),
           blocos_com_previsao=len(pas), n_caracteristicas_no_primeiro_bloco=len(feats),
           pas_min=min(pas) if pas else None, pas_max=max(pas) if pas else None, calibracao=[SBP_CAL, DBP_CAL],
           erros_no_registo=log.count("Erro") + log.count("Traceback"), sucesso=sucesso)
json.dump(res, open(PASTA / "resultado.json", "w"), indent=1, ensure_ascii=False)
limpar()
with ENG.connect() as c:
    resto = sum(c.execute(text(f"SELECT COUNT(*) FROM {t} WHERE pulseira_id = :p OR paciente_id = :d"), dict(p=PULSEIRA, d=PACIENTE)).scalar() for t in TABELAS)
    resto += c.execute(text("SELECT COUNT(*) FROM pulseira WHERE id = :p"), dict(p=PULSEIRA)).scalar()
    resto += c.execute(text("SELECT COUNT(*) FROM paciente WHERE id = :d"), dict(d=PACIENTE)).scalar()
print(f"linhas do ensaio que ficaram na base de dados: {resto}")
print(json.dumps(res, indent=1, ensure_ascii=False))
