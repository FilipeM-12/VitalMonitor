#!/usr/bin/env python3
"""Sonda da extração de características com e sem o canal de ECG, com a extração só-PPG do worker instalado (extracao_ppg, 43 características).

Reproduz o que os workers fazem com o buffer de uma pulseira (janela de 15 s avaliada de 5 em 5 s, sobre um buffer de 25 s)
sobre o MESMO sinal do ensaio (o gerador do simulador: o ficheiro de bancada reproduzido por reflexão) e conta quantas
janelas cada extração aceita:

  com ECG        extração do modelo absoluto (comum.extrair_features_janela, ancorada nos picos R), canal de ECG real (controlo)
  ECG a zeros    a mesma extração, canal de ECG a zeros: o que se espera é 0 janelas aceites
  só-PPG         extração só-PPG (extracao_ppg, a do worker instalado), canal de ECG a zeros

O controlo mostra que o zero não vem do sinal de PPG nem do ficheiro, mas da falta do ECG. Escreve o resultado em JSON.

Uso: python sonda_extracao_ecg_morf.py <csv> <duracao_s> <saida.json> [janelas.json]   (correr com o python do servidor)
     janelas.json, se dado, recebe o instante (s desde o início do sinal) do fim de cada janela avaliada e quais
     foram aceites, para a figura do ensaio.
"""
import json
import sys
from collections import deque
from pathlib import Path


SERVIDOR = Path(__file__).resolve().parents[4] / "servidor" / "workers"
sys.path.insert(0, str(SERVIDOR))
sys.path.insert(0, "/home/filipe/Desktop/Tese/arduino/simulador")
import comum as bp                      # noqa: E402  (extração do modelo absoluto, ancorada nos picos R)
import extracao_ppg as epp    # noqa: E402  (a do worker instalado)
from simulador_pulseira import GeradorCSV   # noqa: E402

csv, duracao, saida = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
JANELAS = sys.argv[4] if len(sys.argv) > 4 else None
LOTE = 25


def contar(ecg_a_zeros: bool):
    ger = GeradorCSV(csv)
    buf = deque(maxlen=bp.BUFFER_MAX)
    cont = 0
    n = ecg_ok = ppg_ok = 0
    fim_s, aceite_ecg, aceite_ppg = [], [], []
    for k_lote in range(int(duracao * bp.FS / LOTE)):
        e, i, r = ger.gerar(LOTE)
        for k in range(LOTE):
            buf.append({"ecg": 0.0 if ecg_a_zeros else float(e[k]), "ir": float(i[k]), "red": float(r[k])})
        cont += LOTE
        if cont < bp.FEATURE_STRIDE or len(buf) < bp.FEATURE_WINDOW + 2 * bp.FEATURE_MARGEM:
            continue
        cont -= bp.FEATURE_STRIDE
        snap = list(buf)
        n += 1
        a_ecg = bp.extrair_features_janela(snap, fs=bp.FS) is not None
        a_ppg = epp.extrair_features_janela_ppg(snap, fs=bp.FS, feature_window=bp.FEATURE_WINDOW,
                                                feature_margem=bp.FEATURE_MARGEM) is not None
        ecg_ok += a_ecg
        ppg_ok += a_ppg
        fim_s.append((k_lote + 1) * LOTE / bp.FS)
        aceite_ecg.append(bool(a_ecg))
        aceite_ppg.append(bool(a_ppg))
    contar.janelas[ecg_a_zeros] = dict(fim_s=fim_s, extracao_com_ecg=aceite_ecg, extracao_so_ppg=aceite_ppg)
    return n, ecg_ok, ppg_ok


contar.janelas = {}


n1, ecg1, ppg1 = contar(False)
n0, ecg0, ppg0 = contar(True)
assert n0 == n1
res = dict(csv=csv.name, segundos_de_sinal=duracao, janelas_avaliadas=n0,
           extracao_com_ecg__ecg_real=ecg1, extracao_com_ecg__ecg_a_zeros=ecg0,
           extracao_so_ppg__ecg_real=ppg1, extracao_so_ppg__ecg_a_zeros=ppg0)
json.dump(res, open(saida, "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
if JANELAS:
    json.dump({"ecg_real": contar.janelas[False], "ecg_a_zeros": contar.janelas[True]}, open(JANELAS, "w"))
