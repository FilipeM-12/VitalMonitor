"""
Equivalente a identificar_janelas_vasoativas.py, mas para os pacientes MIMIC-IV
(pastas com 8 dígitos, ex. p10126957) em MIMIC_DATASET_CONSOLIDADO.

Diferenças-chave face ao MIMIC-III:
- O header "mestre" (com base_date/base_time + lista de segmentos/gaps) já está
  MUITAS VEZES presente localmente, com o mesmo nome do grupo numérico (ex.
  "82924339.hea", ao lado dos segmentos "82924339_0001.hea" etc.) -- não é preciso
  ir sempre ao PhysioNet buscar um ficheiro com outro nome como no MIMIC-III.
  Quando falta localmente, descarrega-se de
  https://physionet.org/files/mimic4wdb/0.1.0/waves/{pasta[:4]}/{pasta}/{grupo}/{grupo}.hea
- Os fármacos vasoativos estão em icu/inputevents.csv (formato único, sem a
  distinção CV/MV do MIMIC-III) com STARTTIME/ENDTIME exato -- não é preciso
  reconstruir episódios por clustering.
- Os ITEMID coincidem em grande parte com os do MIMIC-III Metavision (o MIMIC-IV
  manteve o mesmo sistema), mais alguns itemid novos (confirmados via d_items.csv.gz).

Reaproveita a função de parsing do header mestre e o cálculo de janelas de
identificar_janelas_vasoativas.py (mesmo formato WFDB, mesma convenção de 15s/
stride 5s).

SAÍDA: janelas_vasoativas_excluir_mimic4.csv, com as mesmas colunas do ficheiro
MIMIC-III -- para depois serem concatenados num único ficheiro de exclusão.
"""
import os
import re
import requests
import pandas as pd

import identificar_janelas_vasoativas as v3  # reaproveita parse_master_header, calcular_janelas_segmento

PASTA_DATASET = v3.PASTA_DATASET
PASTA_MIMIC4 = "/media/filipe/Filipe/Datasets/mimic-iv-3.1/icu"
SAIDA_CSV = "../../06_FEATURES_DATASETS/janelas_vasoativas_excluir_mimic4.csv"
BASE_URL_V4 = "https://physionet.org/files/mimic4wdb/0.1.0/waves"

ITEMIDS_V4 = {
    221906: ("Norepinefrina", "vasopressor"),
    221289: ("Epinefrina/Adrenalina", "vasopressor"),
    229617: ("Epinefrina/Adrenalina", "vasopressor"),
    221662: ("Dopamina", "vasopressor"),
    221653: ("Dobutamina", "vasopressor"),
    221749: ("Fenilefrina", "vasopressor"),
    229630: ("Fenilefrina (50/250)", "vasopressor"),
    229631: ("Fenilefrina (200/250)_OLD", "vasopressor"),
    229632: ("Fenilefrina (200/250)", "vasopressor"),
    222315: ("Vasopressina", "vasopressor"),
    222056: ("Nitroglicerina", "vasodilatador"),
    222051: ("Nitroprussiato de sódio", "vasodilatador"),
}

RE_PASTA_MIMIC4 = re.compile(r"^p(\d{8})$")


def listar_pacientes_mimic4():
    pacientes = []
    for nome in os.listdir(PASTA_DATASET):
        caminho = os.path.join(PASTA_DATASET, nome)
        if not os.path.isdir(caminho):
            continue
        if not RE_PASTA_MIMIC4.match(nome):
            continue
        if os.listdir(caminho):
            pacientes.append(nome)
    return sorted(pacientes)


def listar_grupos_locais(caminho_paciente):
    """Nomes de grupo (ex: '82924339') que têm pelo menos um segmento local
    '<grupo>_<n>.hea'."""
    grupos = set()
    for f in os.listdir(caminho_paciente):
        m = re.match(r"^(\d+)_\d+\.hea$", f)
        if m:
            grupos.add(m.group(1))
    return sorted(grupos)


def obter_master_grupo(sessao, pasta_paciente, grupo):
    caminho_local = os.path.join(PASTA_DATASET, pasta_paciente, f"{grupo}.hea")
    if os.path.exists(caminho_local):
        with open(caminho_local) as f:
            return f.read()
    sub = pasta_paciente[:4]
    url = f"{BASE_URL_V4}/{sub}/{pasta_paciente}/{grupo}/{grupo}.hea"
    try:
        r = sessao.get(url, timeout=30)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        print(f"  [aviso] {pasta_paciente}/{grupo}: falha no download do master ({e})")
        return None


def obter_inicios_absolutos_paciente(sessao, pasta_paciente, grupos):
    resultado = {}
    for grupo in grupos:
        texto = obter_master_grupo(sessao, pasta_paciente, grupo)
        if texto is None:
            continue
        try:
            base_dt, fs, offsets = v3.parse_master_header(texto)
        except Exception as e:
            print(f"  [aviso] {pasta_paciente}/{grupo}: falha no parse do master ({e})")
            continue
        for seg_name, offset_frames in offsets.items():
            abs_start = base_dt + pd.Timedelta(seconds=offset_frames / fs)
            resultado[seg_name] = (abs_start, fs)
    return resultado


def construir_episodios_iv(subject_ids):
    print("A carregar icu/inputevents.csv (MIMIC-IV, STARTTIME/ENDTIME exato)...")
    caminho = os.path.join(PASTA_MIMIC4, "inputevents.csv")
    usecols = ["subject_id", "itemid", "starttime", "endtime"]
    partes = []
    for chunk in pd.read_csv(caminho, usecols=usecols, chunksize=2_000_000, low_memory=False):
        chunk = chunk[chunk["subject_id"].isin(subject_ids) & chunk["itemid"].isin(ITEMIDS_V4)]
        if len(chunk):
            partes.append(chunk)
    if not partes:
        return pd.DataFrame(columns=["subject_id", "farmaco", "tipo", "inicio", "fim"])
    df = pd.concat(partes, ignore_index=True)
    df["farmaco"] = df["itemid"].map(lambda i: ITEMIDS_V4[i][0])
    df["tipo"] = df["itemid"].map(lambda i: ITEMIDS_V4[i][1])
    df["inicio"] = pd.to_datetime(df["starttime"])
    # +1h de buffer pós-paragem, consistente com o MIMIC-III (v3.BUFFER_POS_CV_HORAS)
    df["fim"] = pd.to_datetime(df["endtime"]) + pd.Timedelta(hours=v3.BUFFER_POS_CV_HORAS)
    return df[["subject_id", "farmaco", "tipo", "inicio", "fim"]]


def main():
    pacientes = listar_pacientes_mimic4()
    print(f"{len(pacientes)} pacientes MIMIC-IV com dados reais a processar.")

    subject_ids = {int(p[1:]) for p in pacientes}
    episodios = construir_episodios_iv(subject_ids)
    print(f"{len(episodios)} episódios de administração vasoativa encontrados "
          f"({episodios['subject_id'].nunique()} pacientes distintos).")

    episodios_por_subject = {sid: g[["inicio", "fim", "farmaco"]].values.tolist()
                              for sid, g in episodios.groupby("subject_id")}

    sessao = requests.Session()
    linhas_excluir = []
    total_janelas = 0
    for i, pasta in enumerate(pacientes):
        subj = int(pasta[1:])
        caminho_paciente = os.path.join(PASTA_DATASET, pasta)
        eps = episodios_por_subject.get(subj, [])

        grupos = listar_grupos_locais(caminho_paciente)
        if not grupos:
            continue

        inicios = obter_inicios_absolutos_paciente(sessao, pasta, grupos)

        heas_locais = sorted(f for f in os.listdir(caminho_paciente)
                              if re.match(r"^\d+_\d+\.hea$", f))

        for hea in heas_locais:
            seg_name = hea[:-4]
            if seg_name not in inicios:
                continue
            abs_start, fs = inicios[seg_name]
            n_amostras = v3.ler_n_amostras_local(os.path.join(caminho_paciente, hea))
            seg_order = int(seg_name.split("_")[-1])
            janelas = v3.calcular_janelas_segmento(abs_start, fs, n_amostras)
            total_janelas += len(janelas)

            if not eps:
                continue
            for w, ini_abs, fim_abs in janelas:
                for ep_inicio, ep_fim, ep_farmaco in eps:
                    if ini_abs < ep_fim and fim_abs > ep_inicio:
                        linhas_excluir.append((subj, seg_name, seg_order, w, ep_farmaco))
                        break

        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(pacientes)} pacientes processados "
                  f"({len(linhas_excluir)} janelas marcadas até agora)")

    resultado = pd.DataFrame(linhas_excluir,
                              columns=["subject_id", "segment_name", "segment_order", "window_idx", "farmaco"])
    resultado.to_csv(SAIDA_CSV, index=False)
    print(f"\n{len(resultado)} de {total_janelas} janelas ({100*len(resultado)/max(total_janelas,1):.1f}%) "
          f"marcadas para exclusão em {resultado['subject_id'].nunique()} pacientes MIMIC-IV.")
    print(f"Guardado em: {SAIDA_CSV}")


if __name__ == "__main__":
    main()
