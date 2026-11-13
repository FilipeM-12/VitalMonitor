#!/usr/bin/env python3
"""Doentes de demonstração para as capturas do painel. python demo.py criar|apagar
Cria 3 doentes fictícios («Paciente 1, 2, 3») com pulseira, atribuição ativa e calibração; 'apagar' remove tudo o que tiver estes ids
(incluindo as medições que os workers gravarem entretanto)."""
import sys, datetime as dt
sys.path.insert(0, "/home/filipe/Desktop/Tese/server")
import comum as bp
from sqlalchemy import text
ANTIGOS = [("DEMO_01", "PULSEIRA_DEMO_1"), ("DEMO_02", "PULSEIRA_DEMO_2"), ("DEMO_03", "PULSEIRA_DEMO_3")]   # 1.ª versão, a apagar
DEMO = [("PAC_001", "Paciente 1", "M", 67, "PULSEIRA_P1", 142, 84),
        ("PAC_002", "Paciente 2", "F", 54, "PULSEIRA_P2", 118, 72),
        ("PAC_003", "Paciente 3", "M", 72, "PULSEIRA_P3", 156, 88)]
acao = sys.argv[1]
with bp.engine.begin() as c:
    for pid, pul in ANTIGOS:
        for t in ("medicao", "medicao_tendencia", "medicao_absoluta", "calibracao", "atribuicao"):
            c.execute(text(f"DELETE FROM {t} WHERE pulseira_id = :p OR paciente_id = :d"), dict(p=pul, d=pid))
        c.execute(text("DELETE FROM pulseira WHERE id = :p"), dict(p=pul))
        c.execute(text("DELETE FROM paciente WHERE id = :d"), dict(d=pid))
    for pid, nome, sexo, idade, pul, sbp, dbp in DEMO:
        for t in ("medicao", "medicao_tendencia", "medicao_absoluta", "calibracao", "atribuicao"):
            c.execute(text(f"DELETE FROM {t} WHERE pulseira_id = :p OR paciente_id = :d"), dict(p=pul, d=pid))
        c.execute(text("DELETE FROM pulseira WHERE id = :p"), dict(p=pul))
        c.execute(text("DELETE FROM paciente WHERE id = :d"), dict(d=pid))
        if acao == "criar":
            agora = dt.datetime.now(dt.UTC).replace(tzinfo=None)
            c.execute(text("INSERT INTO paciente (id, nome, sexo, idade, data_registo) VALUES (:d, :n, :s, :i, :t)"), dict(d=pid, n=nome, s=sexo, i=idade, t=agora))
            c.execute(text("INSERT INTO pulseira (id, paciente_id) VALUES (:p, :d)"), dict(p=pul, d=pid))
            c.execute(text("INSERT INTO atribuicao (pulseira_id, paciente_id, data_inicio) VALUES (:p, :d, :t)"), dict(p=pul, d=pid, t=agora))
            c.execute(text("INSERT INTO calibracao (pulseira_id, paciente_id, sbp, dbp, timestamp) VALUES (:p, :d, :s, :b, :t)"), dict(p=pul, d=pid, s=sbp, b=dbp, t=agora))
    restam = c.execute(text("SELECT count(*) FROM paciente WHERE id LIKE 'DEMO_%' OR id LIKE 'PAC_00%'")).scalar()
print(acao, "| doentes DEMO na base:", restam)
