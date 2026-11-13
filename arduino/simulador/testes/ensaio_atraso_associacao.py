#!/usr/bin/env python3
"""Quanto tempo passa entre associar a pulseira a um doente e aparecerem os primeiros valores?

Mede-se no servidor real (workers e base de dados do servidor), com pulseiras simuladas de identificador ZZ_TESTE_ATRASO_<n> e doentes ZZ_TESTE_ATRASO_DOENTE_<n>.
Cada pulseira simulada começa a emitir ANTES de ser associada (é o caso do triage). Associa-se 10, 30 e 50 s depois do início, inserindo as linhas que o servidor insere
(paciente, pulseira, atribuicao) e mede-se, a cada 0,25 s, quando surgem na tabela `medicao` a primeira linha e o primeiro valor de cada sinal.

As linhas de teste apagam-se depois com limpar_ensaio_atraso.py (primeiro lista, depois apaga só os identificadores com o prefixo ZZ_TESTE_ATRASO_).
Uso (com o python do servidor): python3 ensaio_atraso_associacao.py <rotulo>   (escreve ensaio_atraso_associacao_<rotulo>.json ao lado)
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import psycopg2

SERVER = "/home/filipe/Desktop/Tese/server"
SIM = "/home/filipe/Desktop/Tese/arduino/simulador/simulador_pulseira.py"
AQUI = os.path.dirname(os.path.abspath(__file__))
DSN = os.environ.get("DATABASE_URL", "postgresql://hospital:hospital_pass@localhost:5432/hospital")
OFFS = [10, 30, 50]              # s depois do início da emissão
DURACAO_SIM = 260
CAMPOS = ["linha", "bpm", "spo2", "temp", "resp"]
rotulo = sys.argv[1] if len(sys.argv) > 1 else "teste"
ids = [f"ZZ_TESTE_ATRASO_{n}" for n in range(len(OFFS))]
pac = {i: f"ZZ_TESTE_ATRASO_DOENTE_{n}" for n, i in enumerate(ids)}

con = psycopg2.connect(DSN)
con.autocommit = True
cur = con.cursor()
cur.execute("SELECT count(*) FROM pulseira WHERE id LIKE 'ZZ\\_TESTE\\_ATRASO\\_%%'")
assert cur.fetchone()[0] == 0, "já há pulseiras de teste na base, correr primeiro limpar_ensaio_atraso.py"

sims = [subprocess.Popen([sys.executable, SIM, "--pulseira", i, "--duracao", str(DURACAO_SIM), "--velocidade", "1", "--silencioso"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for i in ids]
t0 = time.time()
print(f"{len(ids)} pulseiras simuladas a emitir; a associar aos {OFFS} s", flush=True)
assoc, achado = {}, {i: {} for i in ids}
try:
    fim = t0 + max(OFFS) + 60 + 150
    while time.time() < fim:
        agora = time.time()
        for i, off in zip(ids, OFFS):
            if i not in assoc and agora >= t0 + off:
                cur.execute("INSERT INTO paciente (id, idade, sexo, data_registo) VALUES (%s, 60, 'M', %s)", (pac[i], datetime.utcnow()))
                cur.execute("INSERT INTO pulseira (id, paciente_id) VALUES (%s, %s)", (i, pac[i]))
                cur.execute("INSERT INTO atribuicao (pulseira_id, paciente_id, data_inicio) VALUES (%s, %s, %s)", (i, pac[i], datetime.utcnow()))
                assoc[i] = time.time()
        for i in assoc:
            if len(achado[i]) == len(CAMPOS):
                continue
            for c in CAMPOS:
                if c in achado[i]:
                    continue
                cond = "" if c == "linha" else f" AND {c} IS NOT NULL"
                cur.execute(f"SELECT 1 FROM medicao WHERE pulseira_id=%s{cond} LIMIT 1", (i,))
                if cur.fetchone():
                    achado[i][c] = round(time.time() - assoc[i], 2)
        if len(assoc) == len(ids) and all(len(achado[i]) == len(CAMPOS) for i in ids):
            break
        time.sleep(0.25)
finally:
    for s in sims:
        s.terminate()
    con.close()
res = dict(rotulo=rotulo, associacao_s_apos_inicio=dict(zip(ids, OFFS)), atraso_s_ate_primeiro_valor={i: achado[i] for i in ids},
           nota="atraso = segundos desde a associação (linhas inseridas) até surgir em `medicao` a primeira linha / o primeiro valor não nulo de cada sinal")
json.dump(res, open(f"{AQUI}/ensaio_atraso_associacao_{rotulo}.json", "w"), indent=1)
print(json.dumps(res, indent=1, ensure_ascii=False))
