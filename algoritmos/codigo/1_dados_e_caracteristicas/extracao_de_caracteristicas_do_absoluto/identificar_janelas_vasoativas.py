"""
Identifica, ao nível da JANELA (15s, stride 5s -- mesma convenção do
Extração_features.ipynb), quais janelas de sinal se sobrepõem a uma administração
de fármaco vasoativo (vasopressor ou vasodilatador), para os pacientes MIMIC-III
(6 dígitos) em MIMIC_DATASET_CONSOLIDADO.

PROBLEMA RESOLVIDO: os headers .hea de cada segmento, já descarregados localmente,
só têm HORA (sem data) -- p.ex. "3010365_0001 5 125 125 00:14:04.072". A data
absoluta só existe no header "mestre" do registo multi-segmento, publicado à parte
no PhysioNet com o nome "p{subject}-AAAA-MM-DD-HH-MM.hea", que também lista os
GAPS (falhas de gravação) entre segmentos -- necessário para reconstruir o tempo
absoluto de forma correta mesmo quando há interrupções.

MÉTODO:
1. Para cada paciente, obtém a listagem remota da pasta no PhysioNet e descarrega
   os headers mestre (pequenos, só texto).
2. Cada header mestre dá: base_date + base_time (início do registo) e a lista
   ordenada de segmentos/gaps em nº de amostras -- permite calcular o instante
   absoluto de início de cada segmento por soma cumulativa.
3. Para cada segmento presente localmente, recalcula as janelas (mesma fórmula do
   notebook: samples_per_win = 15*fs, step = 5*fs, num_windows = ((n-samples_per_win)
   // step) + 1) e o instante absoluto de cada uma.
4. Cruza com os intervalos de administração de fármacos vasoativos:
   - INPUTEVENTS_MV: intervalo exato [STARTTIME, ENDTIME] por linha.
   - INPUTEVENTS_CV: eventos pontuais (RATE a cada CHARTTIME) -- agrupados por
     paciente+fármaco em "episódios" (gap máximo de 4h entre leituras consecutivas
     conta como o mesmo episódio contínuo), com buffer de +1h após a última leitura
     (fármacos de ação curta, mas a granularidade de registo do CareVue é ~horária).
   Esta reconstrução do CV é uma aproximação -- não há STARTTIME/ENDTIME explícito
   nessa tabela (INPUTEVENTS_CV).
5. Marca como "excluir" qualquer janela cujo intervalo [abs_start, abs_end] se
   sobreponha a um episódio de administração do MESMO subject_id.

SAÍDA: janelas_vasoativas_excluir.csv com a chave completa de janela
(subject_id, segment_name, segment_order, window_idx) + o(s) fármaco(s) responsável.
Este ficheiro NÃO apaga nada -- é uma lista de exclusão para usar como filtro no
pipeline de treino (o inverso de chaves_janelas_permitidas em Comparacao_cohort_nk.ipynb).
"""
import os
import re
import requests
import pandas as pd

PASTA_DATASET = "/media/filipe/Dados/MIMIC_DATASET_CONSOLIDADO"
PASTA_MIMIC = "/media/filipe/Filipe/Datasets/mimic-iii-1.4"
SAIDA_CSV = "../../06_FEATURES_DATASETS/janelas_vasoativas_excluir.csv"
BASE_URL = "https://physionet.org/files/mimic3wdb-matched/1.0"

WINDOW_SIZE_SEC = 15
STRIDE_SEC = 5
BUFFER_POS_CV_HORAS = 1

ITEMIDS = {
    221906: ("Norepinefrina", "vasopressor"),
    30047:  ("Norepinefrina (Levophed)", "vasopressor"),
    30120:  ("Norepinefrina (Levophed-k)", "vasopressor"),
    221289: ("Epinefrina/Adrenalina", "vasopressor"),
    30044:  ("Epinefrina/Adrenalina", "vasopressor"),
    30119:  ("Epinefrina/Adrenalina-k", "vasopressor"),
    30309:  ("Epinefrina/Adrenalina (drip)", "vasopressor"),
    221662: ("Dopamina", "vasopressor"),
    30043:  ("Dopamina", "vasopressor"),
    30307:  ("Dopamina (drip)", "vasopressor"),
    221653: ("Dobutamina", "vasopressor"),
    30042:  ("Dobutamina", "vasopressor"),
    30306:  ("Dobutamina (drip)", "vasopressor"),
    221749: ("Fenilefrina", "vasopressor"),
    222315: ("Vasopressina", "vasopressor"),
    30051:  ("Vasopressina", "vasopressor"),
    222056: ("Nitroglicerina", "vasodilatador"),
    30049:  ("Nitroglicerina", "vasodilatador"),
    30121:  ("Nitroglicerina-k", "vasodilatador"),
    222051: ("Nitroprussiato de sódio", "vasodilatador"),
    30050:  ("Nitroprussiato de sódio", "vasodilatador"),
}

RE_PASTA_MIMIC3 = re.compile(r"^p(\d{6})$")
RE_MASTER_HEA = re.compile(r"^p\d{6}-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}\.hea$")


def listar_pacientes_mimic3():
    pacientes = []
    for nome in os.listdir(PASTA_DATASET):
        caminho = os.path.join(PASTA_DATASET, nome)
        if not os.path.isdir(caminho):
            continue
        if not RE_PASTA_MIMIC3.match(nome):
            continue
        if os.listdir(caminho):
            pacientes.append(nome)
    return sorted(pacientes)


def obter_listagem_remota(sessao, pasta_paciente):
    sub = pasta_paciente[:3]
    url = f"{BASE_URL}/{sub}/{pasta_paciente}/"
    r = sessao.get(url, timeout=30)
    r.raise_for_status()
    return re.findall(r'href="([^"]+\.hea)"', r.text)


def parse_master_header(texto):
    # ignora comentários (incl. "#wfdb 10.7" no início dos masters do MIMIC-IV)
    linhas = [l for l in texto.strip().split("\n") if l.strip() and not l.startswith("#")]
    partes = linhas[0].split()
    fs = float(partes[2].split("/")[0])  # MIMIC-IV usa "freq/counter_freq"
    base_time_str, base_date_str = partes[4], partes[5]
    base_dt = pd.to_datetime(f"{base_date_str} {base_time_str}", format="%d/%m/%Y %H:%M:%S.%f")

    offsets = {}
    cursor = 0
    for linha in linhas[1:]:
        campos = linha.split()
        if not campos or linha.startswith("#"):
            continue
        nome_seg = campos[0]
        try:
            n_amostras = int(campos[1])
        except (ValueError, IndexError):
            continue
        if nome_seg == "~":
            cursor += n_amostras
            continue
        if nome_seg.endswith("_layout"):
            continue
        offsets[nome_seg] = cursor
        cursor += n_amostras
    return base_dt, fs, offsets


def obter_inicios_absolutos_paciente(sessao, pasta_paciente):
    """Devolve {segment_name: (abs_start_datetime, fs)} combinando todos os
    headers mestre encontrados para este paciente."""
    try:
        ficheiros = obter_listagem_remota(sessao, pasta_paciente)
    except requests.RequestException as e:
        print(f"  [aviso] {pasta_paciente}: falha na listagem remota ({e})")
        return {}

    masters = [f for f in ficheiros if RE_MASTER_HEA.match(f)]
    resultado = {}
    for m in masters:
        url = f"{BASE_URL}/{pasta_paciente[:3]}/{pasta_paciente}/{m}"
        try:
            r = sessao.get(url, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  [aviso] {pasta_paciente}/{m}: falha no download ({e})")
            continue
        try:
            base_dt, fs, offsets = parse_master_header(r.text)
        except Exception as e:
            print(f"  [aviso] {pasta_paciente}/{m}: falha no parse ({e})")
            continue
        for seg_name, offset_amostras in offsets.items():
            abs_start = base_dt + pd.Timedelta(seconds=offset_amostras / fs)
            resultado[seg_name] = (abs_start, fs)
    return resultado


def ler_n_amostras_local(caminho_hea):
    with open(caminho_hea) as f:
        primeira_linha = f.readline()
    campos = primeira_linha.split()
    return int(campos[3])


def calcular_janelas_segmento(abs_start, fs, n_amostras):
    samples_per_win = int(WINDOW_SIZE_SEC * fs)
    step_samples = int(STRIDE_SEC * fs)
    num_windows = int((n_amostras - samples_per_win) / step_samples) + 1
    if num_windows <= 0:
        return []
    janelas = []
    for w in range(num_windows):
        ini_amostra = w * step_samples
        ini_abs = abs_start + pd.Timedelta(seconds=ini_amostra / fs)
        fim_abs = ini_abs + pd.Timedelta(seconds=WINDOW_SIZE_SEC)
        janelas.append((w, ini_abs, fim_abs))
    return janelas


def construir_episodios_mv(subject_ids):
    print("A carregar INPUTEVENTS_MV.csv (episódios exatos STARTTIME/ENDTIME)...")
    caminho = os.path.join(PASTA_MIMIC, "INPUTEVENTS_MV.csv")
    usecols = ["SUBJECT_ID", "ITEMID", "STARTTIME", "ENDTIME"]
    partes = []
    for chunk in pd.read_csv(caminho, usecols=usecols, chunksize=2_000_000, low_memory=False):
        chunk = chunk[chunk["SUBJECT_ID"].isin(subject_ids) & chunk["ITEMID"].isin(ITEMIDS)]
        if len(chunk):
            partes.append(chunk)
    if not partes:
        return pd.DataFrame(columns=["subject_id", "farmaco", "tipo", "inicio", "fim"])
    df = pd.concat(partes, ignore_index=True)
    df["farmaco"] = df["ITEMID"].map(lambda i: ITEMIDS[i][0])
    df["tipo"] = df["ITEMID"].map(lambda i: ITEMIDS[i][1])
    df["inicio"] = pd.to_datetime(df["STARTTIME"])
    # +1h de buffer pós-paragem, igual ao usado no CV (consistência entre as
    # duas fontes -- generoso face à semivida real destes fármacos, minutos).
    df["fim"] = pd.to_datetime(df["ENDTIME"]) + pd.Timedelta(hours=BUFFER_POS_CV_HORAS)
    df = df.rename(columns={"SUBJECT_ID": "subject_id"})
    return df[["subject_id", "farmaco", "tipo", "inicio", "fim"]]


def construir_episodios_cv(subject_ids):
    print("A carregar INPUTEVENTS_CV.csv (reconstrução de episódios a partir do RATE entre leituras)...")
    caminho = os.path.join(PASTA_MIMIC, "INPUTEVENTS_CV.csv")
    usecols = ["SUBJECT_ID", "ITEMID", "CHARTTIME", "RATE"]
    partes = []
    for chunk in pd.read_csv(caminho, usecols=usecols, chunksize=2_000_000, low_memory=False):
        chunk = chunk[chunk["SUBJECT_ID"].isin(subject_ids) & chunk["ITEMID"].isin(ITEMIDS)]
        if len(chunk):
            partes.append(chunk)
    if not partes:
        return pd.DataFrame(columns=["subject_id", "farmaco", "tipo", "inicio", "fim"])
    df = pd.concat(partes, ignore_index=True)
    df["farmaco"] = df["ITEMID"].map(lambda i: ITEMIDS[i][0])
    df["tipo"] = df["ITEMID"].map(lambda i: ITEMIDS[i][1])
    df["CHARTTIME"] = pd.to_datetime(df["CHARTTIME"])

    # O CareVue regista RATE periodicamente (~hora a hora); o valor de cada
    # leitura mantém-se ATIVO até à leitura seguinte para o mesmo fármaco. Um
    # segmento [leitura_i, leitura_i+1) só conta como administração ATIVA se
    # RATE da leitura_i > 0 -- caso contrário fica-se com leituras "RATE=0,
    # ainda no doente mas parado" a esticarem o episódio por horas (encontrado
    # ao auditar p007654/Nitroglicerina: episódio real 08:30-10:30 com esta
    # lógica; um clustering ingénuo por gap de tempo, sem olhar ao RATE,
    # esticava-o até às 18:00).
    episodios = []
    for (subj, farmaco, tipo), grupo in df.groupby(["SUBJECT_ID", "farmaco", "tipo"]):
        grupo = grupo.sort_values("CHARTTIME")
        tempos = grupo["CHARTTIME"].tolist()
        rates = grupo["RATE"].tolist()

        segmentos = []
        for i in range(len(tempos) - 1):
            if pd.notna(rates[i]) and rates[i] > 0:
                segmentos.append([tempos[i], tempos[i + 1]])
        # última leitura sem leitura seguinte que a encerre -- assume-se ativa
        # só até ao buffer (não sabemos quando parou de facto).
        if pd.notna(rates[-1]) and rates[-1] > 0:
            segmentos.append([tempos[-1], tempos[-1] + pd.Timedelta(hours=BUFFER_POS_CV_HORAS)])
        if not segmentos:
            continue

        segmentos.sort()
        fundidos = [segmentos[0]]
        for ini, fim in segmentos[1:]:
            if ini <= fundidos[-1][1]:
                fundidos[-1][1] = max(fundidos[-1][1], fim)
            else:
                fundidos.append([ini, fim])

        for ini, fim in fundidos:
            episodios.append((subj, farmaco, tipo, ini, fim + pd.Timedelta(hours=BUFFER_POS_CV_HORAS)))

    return pd.DataFrame(episodios, columns=["subject_id", "farmaco", "tipo", "inicio", "fim"])


def main():
    pacientes = listar_pacientes_mimic3()
    print(f"{len(pacientes)} pacientes MIMIC-III com dados reais a processar.")

    subject_ids = {int(p[1:]) for p in pacientes}
    episodios_mv = construir_episodios_mv(subject_ids)
    episodios_cv = construir_episodios_cv(subject_ids)
    episodios = pd.concat([episodios_mv, episodios_cv], ignore_index=True)
    print(f"{len(episodios)} episódios de administração vasoativa reconstruídos "
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

        try:
            heas_locais = sorted(f for f in os.listdir(caminho_paciente) if f.endswith(".hea"))
            # só segmentos numerados (ex: 3010365_0007.hea) -- ignora ficheiros de numerics ("n.hea")
            heas_locais = [f for f in heas_locais if re.match(r"^\d+_\d+\.hea$", f)]
            if not heas_locais:
                continue

            inicios = obter_inicios_absolutos_paciente(sessao, pasta)

            for hea in heas_locais:
                seg_name = hea[:-4]
                if seg_name not in inicios:
                    continue
                abs_start, fs = inicios[seg_name]
                try:
                    n_amostras = ler_n_amostras_local(os.path.join(caminho_paciente, hea))
                except (IndexError, ValueError):
                    print(f"  [aviso] {pasta}/{hea}: header local vazio/corrompido -- ignorado")
                    continue
                seg_order = int(seg_name.split("_")[-1])
                janelas = calcular_janelas_segmento(abs_start, fs, n_amostras)
                total_janelas += len(janelas)

                if not eps:
                    continue
                for w, ini_abs, fim_abs in janelas:
                    for ep_inicio, ep_fim, ep_farmaco in eps:
                        if ini_abs < ep_fim and fim_abs > ep_inicio:
                            linhas_excluir.append((subj, seg_name, seg_order, w, ep_farmaco))
                            break
        except Exception as e:
            print(f"  [aviso] {pasta}: falha inesperada, paciente ignorado ({e})")
            continue

        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(pacientes)} pacientes processados "
                  f"({len(linhas_excluir)} janelas marcadas até agora)")
            pd.DataFrame(linhas_excluir,
                         columns=["subject_id", "segment_name", "segment_order", "window_idx", "farmaco"]
                         ).to_csv(SAIDA_CSV, index=False)

    resultado = pd.DataFrame(linhas_excluir,
                              columns=["subject_id", "segment_name", "segment_order", "window_idx", "farmaco"])
    resultado.to_csv(SAIDA_CSV, index=False)
    print(f"\n{len(resultado)} de {total_janelas} janelas ({100*len(resultado)/max(total_janelas,1):.1f}%) "
          f"marcadas para exclusão em {resultado['subject_id'].nunique()} pacientes.")
    print(f"Guardado em: {SAIDA_CSV}")


if __name__ == "__main__":
    main()
