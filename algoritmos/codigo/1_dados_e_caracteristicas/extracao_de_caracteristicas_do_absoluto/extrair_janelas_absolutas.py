"""
Constrói a "espinha dorsal" temporal: para CADA janela bruta (subject_id,
segment_name, segment_order, window_idx) de TODOS os pacientes com dados reais
(MIMIC-III + MIMIC-IV, não só os afetados por fármaco vasoativo), calcula o
instante absoluto de início (abs_start) e fim (abs_end = abs_start + 15s).

Reaproveita a maquinaria já validada em identificar_janelas_vasoativas.py
(headers mestre do PhysioNet com base_date/base_time + lista de segmentos/gaps)
e identificar_janelas_vasoativas_mimic4.py (headers mestre locais/PhysioNet).

Serve de base reutilizável para juntar QUALQUER variável clínica com timestamp
(fármacos, vitais, laboratório) às janelas de sinal -- não é específico de
fármacos vasoativos.

Saída: janelas_absolutas_TODOS.csv
  colunas: subject_id_str, dataset, record_id, segment_order, window_idx, abs_start, abs_end
"""
import os
import re
import requests
import pandas as pd

import identificar_janelas_vasoativas as v3
import identificar_janelas_vasoativas_mimic4 as v4

SAIDA_CSV = "../../06_FEATURES_DATASETS/janelas_absolutas_TODOS.csv"


def processar_mimic3(sessao):
    pacientes = v3.listar_pacientes_mimic3()
    print(f"[MIMIC-III] {len(pacientes)} pacientes a processar.")
    linhas = []
    for i, pasta in enumerate(pacientes):
        caminho_paciente = os.path.join(v3.PASTA_DATASET, pasta)
        try:
            heas_locais = sorted(f for f in os.listdir(caminho_paciente) if f.endswith(".hea"))
            heas_locais = [f for f in heas_locais if re.match(r"^\d+_\d+\.hea$", f)]
            if not heas_locais:
                continue
            inicios = v3.obter_inicios_absolutos_paciente(sessao, pasta)
            for hea in heas_locais:
                seg_name = hea[:-4]
                if seg_name not in inicios:
                    continue
                abs_start, fs = inicios[seg_name]
                try:
                    n_amostras = v3.ler_n_amostras_local(os.path.join(caminho_paciente, hea))
                except (IndexError, ValueError):
                    continue
                seg_order = int(seg_name.split("_")[-1])
                for w, ini_abs, fim_abs in v3.calcular_janelas_segmento(abs_start, fs, n_amostras):
                    linhas.append((pasta, "MIMIC-III", seg_name.rsplit("_", 1)[0], seg_order, w,
                                   ini_abs, fim_abs))
        except Exception as e:
            print(f"  [aviso] {pasta}: falha inesperada, ignorado ({e})")
            continue
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(pacientes)} ({len(linhas)} janelas até agora)")
    return linhas


def processar_mimic4(sessao):
    pacientes = v4.listar_pacientes_mimic4()
    print(f"[MIMIC-IV] {len(pacientes)} pacientes a processar.")
    linhas = []
    for i, pasta in enumerate(pacientes):
        caminho_paciente = os.path.join(v4.PASTA_DATASET, pasta)
        try:
            grupos = v4.listar_grupos_locais(caminho_paciente)
            if not grupos:
                continue
            inicios = v4.obter_inicios_absolutos_paciente(sessao, pasta, grupos)
            heas_locais = sorted(f for f in os.listdir(caminho_paciente)
                                  if re.match(r"^\d+_\d+\.hea$", f))
            for hea in heas_locais:
                seg_name = hea[:-4]
                if seg_name not in inicios:
                    continue
                abs_start, fs = inicios[seg_name]
                n_amostras = v3.ler_n_amostras_local(os.path.join(caminho_paciente, hea))
                seg_order = int(seg_name.split("_")[-1])
                for w, ini_abs, fim_abs in v3.calcular_janelas_segmento(abs_start, fs, n_amostras):
                    linhas.append((pasta, "MIMIC-IV", seg_name.rsplit("_", 1)[0], seg_order, w,
                                   ini_abs, fim_abs))
        except Exception as e:
            print(f"  [aviso] {pasta}: falha inesperada, ignorado ({e})")
            continue
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(pacientes)} ({len(linhas)} janelas até agora)")
    return linhas


def main():
    sessao = requests.Session()
    linhas = processar_mimic3(sessao) + processar_mimic4(sessao)
    df = pd.DataFrame(linhas, columns=["subject_id_str", "dataset", "record_id",
                                        "segment_order", "window_idx", "abs_start", "abs_end"])
    df.to_csv(SAIDA_CSV, index=False)
    print(f"\n{len(df)} janelas de {df['subject_id_str'].nunique()} pacientes guardadas em {SAIDA_CSV}")


if __name__ == "__main__":
    main()
