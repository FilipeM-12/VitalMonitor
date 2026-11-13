"""Ensaio ponta a ponta do worker da tendência (janela por tempo exato, 43 características, extracao_ppg + derivacao_ppg + modelo_tendencia_ppg_quant_tempo.joblib).

Levanta um mosquitto próprio em 127.0.0.1:18830 (os workers do servidor, na porta 1883, não recebem nada),
um único worker (o novo, desta pasta) ligado à base de dados do servidor com um doente e uma pulseira só do ensaio (PULSEIRA_E2E4 / DOENTE_E2E4), publica sinal real de bancada com o ECG a zeros
(simulador --sem-ecg) e confirma que sai uma previsão por bloco com as 43 características, sem erros no registo. No fim exporta as linhas para CSV e apaga-as de todas as tabelas.
Uso: python ensaio_worker_tempo.py [--duracao 600] [--velocidade 1]
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

SERVER = Path("/home/filipe/Desktop/Tese/server")
PY = SERVER / "venv/bin/python3"
SIMULADOR = Path("/home/filipe/Desktop/Tese/arduino/simulador/simulador_pulseira.py")
AQUI = Path(__file__).resolve().parent
CSV = Path("/home/filipe/Desktop/Tese/arduino/Dados/ECG+PPG/Configuração 1/Todos_participante C_bruto_20260408_193122.csv")
PORTA = 18830
PULSEIRA, PACIENTE, IDADE, SEXO = "PULSEIRA_E2E4", "DOENTE_E2E4", 58, "F"      # idade e sexo do ensaio anterior
TOKEN = "segredo123"

ap = argparse.ArgumentParser()
ap.add_argument("--duracao", type=float, default=600.0, help="segundos de sinal")
ap.add_argument("--velocidade", type=float, default=1.0)
a = ap.parse_args()

# cada duração tem a sua própria pasta, para uma corrida nunca substituir outra
PASTA = AQUI / f"ensaio_worker_{date.today().isoformat()}_{int(a.duracao)}s"
PASTA.mkdir(exist_ok=True)
URL = "postgresql://hospital:hospital_pass@localhost:5432/hospital"
sys.path.insert(0, str(SERVER / "venv/lib/python3.12/site-packages"))
from sqlalchemy import create_engine, text                      # noqa: E402
ENG = create_engine(URL)
TABELAS = ("medicao_tendencia", "medicao_absoluta", "medicao", "calibracao", "atribuicao")


def limpar():
    with ENG.begin() as c:
        for t in TABELAS:
            c.execute(text(f"DELETE FROM {t} WHERE pulseira_id = :p OR paciente_id = :d"), dict(p=PULSEIRA, d=PACIENTE))
        c.execute(text("DELETE FROM pulseira WHERE id = :p"), dict(p=PULSEIRA))
        c.execute(text("DELETE FROM paciente WHERE id = :d"), dict(d=PACIENTE))


limpar()                                          # restos de uma corrida interrompida

# 1. broker próprio
conf = PASTA / "mosquitto_ensaio.conf"
conf.write_text(f"listener {PORTA} 127.0.0.1\nallow_anonymous true\nmax_packet_size 8192\npersistence false\n")
procs = {"mosquitto": subprocess.Popen(["/usr/sbin/mosquitto", "-c", str(conf)],
                                       stdout=open(PASTA / "mosquitto.log", "w"), stderr=subprocess.STDOUT)}
time.sleep(1.5)
assert procs["mosquitto"].poll() is None, "o mosquitto do ensaio não arrancou (ver mosquitto.log)"

# 2. base de dados: tabelas do servidor + doente e pulseira do ensaio
amb = dict(os.environ, DATABASE_URL=URL, MQTT_BROKER="127.0.0.1", MQTT_PORT=str(PORTA), MQTT_USER="worker",
           MQTT_PASS="worker_pass", AUTH_TOKEN=TOKEN, NUM_WORKERS="1", WORKER_ID="0", PYTHONUNBUFFERED="1")
with ENG.begin() as c:
    c.execute(text("INSERT INTO paciente (id, idade, sexo) VALUES (:d, :i, :s)"), dict(d=PACIENTE, i=IDADE, s=SEXO))
    c.execute(text("INSERT INTO pulseira (id, paciente_id) VALUES (:p, :d)"), dict(p=PULSEIRA, d=PACIENTE))

# 3. o worker instalado (servidor/workers: extração, derivação e comum.py; o modelo vem de servidor/modelos)
procs["tempo"] = subprocess.Popen([str(PY), "-u", "worker_tendencia_quantilica.py"], cwd=AQUI.parents[3] / "servidor" / "workers", env=amb,
                                  stdout=open(PASTA / "worker_tempo.log", "w"), stderr=subprocess.STDOUT)
time.sleep(10)
mortos = [n for n, p in procs.items() if p.poll() is not None]
if mortos:
    for p in procs.values():
        p.terminate()
    limpar()
    sys.exit(f"não arrancaram: {mortos} (ver os .log em {PASTA})")

# 4. um único fluxo de sinal real, ECG a zeros
inicio = time.time()
sim = subprocess.run([str(PY), "-u", str(SIMULADOR), "--pulseira", PULSEIRA, "--token", TOKEN, "--broker", "127.0.0.1",
                      "--porta", str(PORTA), "--fonte", "csv", "--csv", str(CSV), "--sem-ecg",
                      "--duracao", str(a.duracao), "--velocidade", str(a.velocidade)],
                     stdout=open(PASTA / "simulador.log", "w"), stderr=subprocess.STDOUT)
time.sleep(30)                                    # últimos blocos a sair dos workers

# 5. resultado
import csv                                        # noqa: E402
with ENG.connect() as c:
    linhas = c.execute(text("""SELECT variante, COUNT(*),
                                      SUM(CASE WHEN NOT estavel_descida THEN 1 ELSE 0 END),
                                      SUM(CASE WHEN NOT estavel_subida THEN 1 ELSE 0 END),
                                      SUM(CASE WHEN NOT estavel_descida OR NOT estavel_subida THEN 1 ELSE 0 END),
                                      MIN(score_descida), MAX(score_descida)
                               FROM medicao_tendencia WHERE pulseira_id = :p GROUP BY variante"""), dict(p=PULSEIRA)).fetchall()
    todas = c.execute(text("SELECT * FROM medicao_tendencia WHERE pulseira_id = :p ORDER BY id"), dict(p=PULSEIRA))
    with open(PASTA / "medicao_tendencia_ensaio.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(todas.keys())
        w.writerows(todas.fetchall())
for p in procs.values():
    p.send_signal(signal.SIGTERM)
for p in procs.values():
    try:
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        p.kill()

import json as _j                              # noqa: E402
rows = list(csv.DictReader(open(PASTA / "medicao_tendencia_ensaio.csv")))
n43 = [len(_j.loads(r["features_ml"])) for r in rows]
nnulos = [sum(v is None for v in _j.loads(r["features_ml"]).values()) for r in rows]
erros = sum("ERROR" in l or "Traceback" in l for l in open(PASTA / "worker_tempo.log"))
res = dict(data=date.today().isoformat(), segundos_de_sinal=a.duracao, velocidade=a.velocidade, simulador_codigo_saida=sim.returncode,
           duracao_real_s=round(time.time() - inicio), previsoes=len(rows), caracteristicas_por_previsao=sorted(set(n43)),
           caracteristicas_em_falta_por_previsao=dict(min=min(nnulos, default=None), max=max(nnulos, default=None)), erros_no_registo=erros,
           score_descida=[min((float(r["score_descida"]) for r in rows), default=None), max((float(r["score_descida"]) for r in rows), default=None)],
           sucesso=bool(rows) and set(n43) == {43} and erros == 0)
json.dump(res, open(PASTA / "resultado.json", "w"), indent=1, ensure_ascii=False)
limpar()
with ENG.connect() as c:
    resto = sum(c.execute(text(f"SELECT COUNT(*) FROM {t} WHERE pulseira_id = :p OR paciente_id = :d"),
                          dict(p=PULSEIRA, d=PACIENTE)).scalar() for t in TABELAS)
    resto += c.execute(text("SELECT COUNT(*) FROM pulseira WHERE id = :p"), dict(p=PULSEIRA)).scalar()
    resto += c.execute(text("SELECT COUNT(*) FROM paciente WHERE id = :d"), dict(d=PACIENTE)).scalar()
print(f"linhas do ensaio que ficaram na base de dados: {resto}")
print(json.dumps(res, indent=1, ensure_ascii=False))
