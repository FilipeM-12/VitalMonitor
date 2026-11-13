#!/usr/bin/env python3
"""anotar_ensaio.py — folha de registo do ensaio PPG preenchida no terminal, com a hora posta automaticamente.

Correr numa segunda janela, ao mesmo tempo que o gravar_mqtt.py. Cada anotação fica logo gravada em
ensaios/registo_<paciente>_<AAAAMMDD_HHMMSS>.csv, no formato que o analisar_ensaio_ppg.py lê
(hora, respiracoes, fc_smartband, spo2_smartband, notas), com a hora local do relógio do PC.

Comandos (Enter no fim):
  r              começa a contar respirações: o programa marca a hora, conta 60 s, apita no fim e pergunta quantas
                 foram; a linha fica com a hora do INÍCIO do minuto
  f 72           FC da smartband agora
  s 98           SpO2 da smartband agora
  f 72 s 98      as duas na mesma linha
  n texto        nota agora (ex.: "n slow breathing"); aparece na figura, por isso em inglês se for para a tese
  q              terminar

Uso:
  /home/filipe/Desktop/Tese/algoritmo-ml/venv/bin/python anotar_ensaio.py
"""
import argparse
import csv
import os
import time
from datetime import datetime

AQUI = os.path.dirname(os.path.abspath(__file__))
COLS = ["hora", "respiracoes", "fc_smartband", "spo2_smartband", "notas"]


def numero(txt):
    try:
        return float(txt.replace(",", "."))
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paciente", default=None)
    ap.add_argument("--minuto", type=float, default=60.0, help=argparse.SUPPRESS)   # só para testar
    a = ap.parse_args()

    nome = a.paciente
    while not nome:
        nome = input("Nome do paciente: ").strip()
    nome = "".join(c if c.isalnum() or c in "-_" else "_" for c in nome.replace(" ", "_")).strip("_")
    os.makedirs(os.path.join(AQUI, "ensaios"), exist_ok=True)
    caminho = os.path.join(AQUI, "ensaios", f"registo_{nome}_{datetime.now():%Y%m%d_%H%M%S}.csv")
    f = open(caminho, "w", newline="")
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader(); f.flush()
    print(f"a registar em {caminho}\ncomandos: r (contar respirações), f <FC>, s <SpO2>, n <nota>, q (sair)")

    def linha(**kv):
        w.writerow({c: kv.get(c, "") for c in COLS}); f.flush()

    while True:
        try:
            cmd = input("> ").strip()
        except EOFError:
            break
        if not cmd:
            continue
        agora = datetime.now().strftime("%H:%M:%S")
        if cmd == "q":
            break
        if cmd == "r":
            print(f"{agora} a contar… (Ctrl+C cancela)")
            try:
                t0 = time.time()
                while (falta := a.minuto - (time.time() - t0)) > 0:
                    print(f"\r  faltam {falta:4.0f} s ", end="", flush=True)
                    time.sleep(min(1.0, falta))
                print("\r  fim do minuto \a")
            except KeyboardInterrupt:
                print("\n  contagem cancelada")
                continue
            while (n := numero(input("  respirações contadas: ").strip() or "x")) is None:
                print("  escreve só o número")
            linha(hora=agora, respiracoes=int(n))
            print(f"  registado: {agora}, {int(n)} respirações")
            continue
        if cmd.startswith("n "):
            linha(hora=agora, notas=cmd[2:].strip())
            print(f"  nota às {agora}")
            continue
        partes, kv, ok = cmd.split(), {}, True
        for i in range(0, len(partes), 2):
            chave = {"f": "fc_smartband", "s": "spo2_smartband"}.get(partes[i])
            v = numero(partes[i + 1]) if i + 1 < len(partes) else None
            if chave is None or v is None:
                ok = False
                break
            kv[chave] = int(v) if v == int(v) else v
        if not ok:
            print("  não percebi; exemplos: r | f 72 | s 98 | f 72 s 98 | n slow breathing | q")
            continue
        linha(hora=agora, **kv)
        print(f"  registado às {agora}: " + ", ".join(f"{k.split('_')[0].upper()} {v}" for k, v in kv.items()))

    f.close()
    print(f"terminado -> {caminho}")


if __name__ == "__main__":
    main()
