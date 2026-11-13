"""Espera até cada um dos 3 doentes DEMO ter pelo menos N previsões de tendência (omissão 12, ou seja 3 min de blocos)."""
import sys, time
sys.path.insert(0, "/home/filipe/Desktop/Tese/server")
import comum as bp
from sqlalchemy import text
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
K = int(sys.argv[2]) if len(sys.argv) > 2 else 3          # quantos doentes têm de ter N previsões novas
while True:
    with bp.engine.connect() as c:
        r = dict(c.execute(text("SELECT paciente_id, count(*) FROM medicao_tendencia WHERE paciente_id LIKE 'PAC_00%' AND timestamp > now() at time zone 'utc' - interval '15 minutes' GROUP BY 1")).fetchall())
    print(time.strftime("%H:%M:%S"), r, flush=True)
    if sum(1 for v in r.values() if v >= N) >= K:
        break
    time.sleep(30)
