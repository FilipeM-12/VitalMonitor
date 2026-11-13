#!/usr/bin/env python3
"""Lista e, com --apagar, remove da base do servidor as linhas do ensaio de atraso da associação.
Só toca em identificadores que começam por ZZ_TESTE_ATRASO_ (pulseiras e doentes de teste). Primeiro mostra quantas linhas há em cada tabela e as linhas de
pulseira e paciente, e só apaga com --apagar, numa única transação, confirmando no fim que não ficou nada."""
import os
import sys

import psycopg2

DSN = os.environ.get("DATABASE_URL", "postgresql://hospital:hospital_pass@localhost:5432/hospital")
P = r"ZZ\_TESTE\_ATRASO\_%"
FILHAS = ["medicao", "medicao_absoluta", "medicao_tendencia", "calibracao", "atribuicao"]
con = psycopg2.connect(DSN)
cur = con.cursor()
tot = {}
for t in FILHAS:
    cur.execute(f"SELECT count(*) FROM {t} WHERE pulseira_id LIKE %s OR paciente_id LIKE %s", (P, P))
    tot[t] = cur.fetchone()[0]
cur.execute("SELECT count(*) FROM pulseira WHERE id LIKE %s", (P,)); tot["pulseira"] = cur.fetchone()[0]
cur.execute("SELECT count(*) FROM paciente WHERE id LIKE %s", (P,)); tot["paciente"] = cur.fetchone()[0]
cur.execute("SELECT id, paciente_id FROM pulseira WHERE id LIKE %s ORDER BY id", (P,)); print("pulseiras:", cur.fetchall())
cur.execute("SELECT id FROM paciente WHERE id LIKE %s ORDER BY id", (P,)); print("doentes:", [r[0] for r in cur.fetchall()])
print("linhas por tabela:", tot)
if "--apagar" not in sys.argv:
    print("(só listagem; usar --apagar para remover)")
    sys.exit(0)
try:
    for t in FILHAS:
        cur.execute(f"DELETE FROM {t} WHERE pulseira_id LIKE %s OR paciente_id LIKE %s", (P, P))
    cur.execute("DELETE FROM pulseira WHERE id LIKE %s", (P,))
    cur.execute("DELETE FROM paciente WHERE id LIKE %s", (P,))
    con.commit()
except Exception:
    con.rollback()
    raise
for t in FILHAS + ["pulseira", "paciente"]:
    col = "id" if t in ("pulseira", "paciente") else "pulseira_id"
    cur.execute(f"SELECT count(*) FROM {t} WHERE {col} LIKE %s", (P,))
    assert cur.fetchone()[0] == 0, t
print("apagado, nada ficou")
