"""
Extrai, para cada janela da "espinha dorsal" (janelas_absolutas_TODOS.csv),
variáveis clínicas com timestamp que plausivelmente modulam o tónus arterial
e por isso a relação PAT <-> pressão arterial:

  - temperatura_c            : última temperatura conhecida (°C), CHARTEVENTS
  - ventilacao_mecanica       : 1 se há registo de "Ventilator Mode" recente, senão 0
  - rass                      : última pontuação de sedação Richmond-RAS conhecida
  - vasopressor_ativo         : 1 se algum vasopressor/vasodilatador está ativo
  - n_vasoativos_ativos       : nº de fármacos vasoativos simultaneamente ativos

Fontes:
  MIMIC-III: CHARTEVENTS.csv(.gz) + INPUTEVENTS_CV/MV (via
             identificar_janelas_vasoativas.py)
  MIMIC-IV : chartevents.csv.gz + icu/inputevents.csv (via
             identificar_janelas_vasoativas_mimic4.py)

Junção "as-of" (último valor conhecido ANTES do início da janela, com um limite
de validade -- LOOKBACK_HORAS -- para não arrastar leituras muito antigas).

Saída: features_clinicas_extra.csv
  chave: subject_id_str, record_id, segment_order, window_idx
"""
import os
import pandas as pd
import numpy as np

import identificar_janelas_vasoativas as v3
import identificar_janelas_vasoativas_mimic4 as v4

ESPINHA_CSV = "../../06_FEATURES_DATASETS/janelas_absolutas_TODOS.csv"
SAIDA_CSV = "../../06_FEATURES_DATASETS/features_clinicas_extra.csv"

LOOKBACK_HORAS_VITAIS = 8      # temperatura/RASS/ventilação: última leitura válida até 8h atrás
ITEMIDS_TEMP_C = {223762: 1.0, 676: 1.0}          # já em Celsius
ITEMIDS_TEMP_F = {223761: 1.0, 678: 1.0}          # em Fahrenheit -- converter
ITEMIDS_VENT_MODE = {223849, 720}                  # presença = ventilado
ITEMIDS_RASS = {228096}                            # valuenum = pontuação -5..+4


def carregar_chartevents_mimic3(subject_ids):
    print("A carregar CHARTEVENTS.csv (MIMIC-III) -- pode demorar (ficheiro grande)...")
    caminho_gz = os.path.join(v3.PASTA_MIMIC, "CHARTEVENTS.csv.gz")
    caminho_csv = os.path.join(v3.PASTA_MIMIC, "CHARTEVENTS.csv")
    caminho = caminho_csv if os.path.exists(caminho_csv) else caminho_gz
    itemids_todos = set(ITEMIDS_TEMP_C) | set(ITEMIDS_TEMP_F) | ITEMIDS_VENT_MODE | ITEMIDS_RASS
    usecols = ["SUBJECT_ID", "ITEMID", "CHARTTIME", "VALUENUM"]
    partes = []
    for chunk in pd.read_csv(caminho, usecols=usecols, chunksize=3_000_000, low_memory=False):
        sub = chunk[chunk["SUBJECT_ID"].isin(subject_ids) & chunk["ITEMID"].isin(itemids_todos)]
        if len(sub):
            partes.append(sub)
    if not partes:
        return pd.DataFrame(columns=["subject_id", "itemid", "charttime", "valuenum"])
    df = pd.concat(partes, ignore_index=True)
    df.columns = ["subject_id", "itemid", "charttime", "valuenum"]
    df["charttime"] = pd.to_datetime(df["charttime"])
    return df


def carregar_chartevents_mimic4(subject_ids):
    print("A carregar chartevents.csv.gz (MIMIC-IV)...")
    caminho = os.path.join(v4.PASTA_MIMIC4, "chartevents.csv.gz")
    itemids_todos = set(ITEMIDS_TEMP_C) | set(ITEMIDS_TEMP_F) | ITEMIDS_VENT_MODE | ITEMIDS_RASS
    usecols = ["subject_id", "itemid", "charttime", "valuenum"]
    partes = []
    for chunk in pd.read_csv(caminho, compression="gzip", usecols=usecols,
                              chunksize=3_000_000, low_memory=False):
        sub = chunk[chunk["subject_id"].isin(subject_ids) & chunk["itemid"].isin(itemids_todos)]
        if len(sub):
            partes.append(sub)
    if not partes:
        return pd.DataFrame(columns=["subject_id", "itemid", "charttime", "valuenum"])
    df = pd.concat(partes, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"])
    return df


def construir_serie_vital(chart_df, itemids, nome_col, converter_f_para_c=False):
    """De um DataFrame CHARTEVENTS filtrado, extrai uma série (subject_id, charttime, valor)
    ordenada, para os itemids dados."""
    sub = chart_df[chart_df["itemid"].isin(itemids)].dropna(subset=["valuenum"]).copy()
    if converter_f_para_c:
        sub["valuenum"] = (sub["valuenum"] - 32) * 5.0 / 9.0
    sub = sub.rename(columns={"valuenum": nome_col})[["subject_id", "charttime", nome_col]]
    return sub.sort_values(["subject_id", "charttime"])


def asof_merge_vital(espinha, serie, nome_col, lookback_horas):
    """merge_asof: último valor conhecido ANTES (ou igual) ao abs_start da janela,
    dentro do limite de validade `lookback_horas`."""
    if serie.empty:
        espinha[nome_col] = np.nan
        return espinha
    espinha_ord = espinha.sort_values("abs_start")
    serie_ord = serie.sort_values("charttime")
    resultado = pd.merge_asof(
        espinha_ord, serie_ord,
        left_on="abs_start", right_on="charttime",
        left_by="subject_id_str", right_by="subject_id",
        direction="backward",
        tolerance=pd.Timedelta(hours=lookback_horas),
    )
    return resultado.drop(columns=["charttime", "subject_id"], errors="ignore")


def construir_contagem_vasoativos(espinha, episodios):
    """Para cada janela, conta quantos episódios de fármaco vasoativo a cobrem
    (overlap [abs_start, abs_end) com [inicio, fim))."""
    if episodios.empty:
        espinha["vasopressor_ativo"] = 0
        espinha["n_vasoativos_ativos"] = 0
        return espinha

    # zip(), NÃO .values.tolist(): .values numa coluna datetime64 seguido de
    # .tolist() dá inteiros (ns brutos), não Timestamps -- comparação ini_j<ef
    # rebenta (DateTime64 vs Int64). Descoberto ao testar no MIMIC-IV.
    eps_por_subject = {sid: list(zip(g["inicio"], g["fim"]))
                       for sid, g in episodios.groupby("subject_id_str")}

    contagens = np.zeros(len(espinha), dtype=int)
    subj_arr = espinha["subject_id_str"].values
    ini_arr = espinha["abs_start"].values
    fim_arr = espinha["abs_end"].values

    for subj, eps in eps_por_subject.items():
        idxs = np.where(subj_arr == subj)[0]
        if len(idxs) == 0:
            continue
        for idx in idxs:
            ini_j, fim_j = ini_arr[idx], fim_arr[idx]
            n = sum(1 for (ei, ef) in eps if ini_j < ef and fim_j > ei)
            contagens[idx] = n

    espinha["n_vasoativos_ativos"] = contagens
    espinha["vasopressor_ativo"] = (contagens > 0).astype(int)
    return espinha


def main():
    # NÃO usar parse_dates aqui: timestamps com microssegundos exatamente a
    # zero são gravados no CSV sem parte decimal (ex: "05:25:52" vs
    # "00:14:12.600"), criando um formato misto que o parser vetorizado do
    # parse_dates falha silenciosamente (~24% das linhas viravam NaT sem
    # aviso). format="mixed" trata cada valor individualmente -- correto.
    espinha = pd.read_csv(ESPINHA_CSV)
    espinha["abs_start"] = pd.to_datetime(espinha["abs_start"], format="mixed")
    espinha["abs_end"] = pd.to_datetime(espinha["abs_end"], format="mixed")
    print(f"{len(espinha)} janelas na espinha dorsal, {espinha['subject_id_str'].nunique()} pacientes.")

    subs_iii = espinha.loc[espinha["dataset"] == "MIMIC-III", "subject_id_str"]
    subs_iv = espinha.loc[espinha["dataset"] == "MIMIC-IV", "subject_id_str"]
    ids_iii = set(subs_iii.apply(lambda s: int(s[1:])))
    ids_iv = set(subs_iv.apply(lambda s: int(s[1:])))

    # ---- vitais (CHARTEVENTS) ----
    chart_iii = carregar_chartevents_mimic3(ids_iii) if ids_iii else pd.DataFrame()
    chart_iv = carregar_chartevents_mimic4(ids_iv) if ids_iv else pd.DataFrame()
    if len(chart_iii):
        chart_iii["subject_id"] = chart_iii["subject_id"].apply(lambda i: f"p{int(i):06d}")
    if len(chart_iv):
        chart_iv["subject_id"] = chart_iv["subject_id"].apply(lambda i: f"p{int(i):08d}")
    partes_chart = [d for d in (chart_iii, chart_iv) if len(d)]
    chart_todos = pd.concat(partes_chart, ignore_index=True) if partes_chart \
        else pd.DataFrame(columns=["subject_id", "itemid", "charttime", "valuenum"])
    # a partir daqui, chart_todos['subject_id'] já é a string com zero-padding
    # (mesmo formato de espinha['subject_id_str']) -- consistente com o resto do script.

    serie_temp_c = construir_serie_vital(chart_todos, ITEMIDS_TEMP_C, "temperatura_c", converter_f_para_c=False)
    serie_temp_f = construir_serie_vital(chart_todos, ITEMIDS_TEMP_F, "temperatura_c", converter_f_para_c=True)
    serie_temp = pd.concat([serie_temp_c, serie_temp_f], ignore_index=True).sort_values(["subject_id", "charttime"])

    serie_vent = construir_serie_vital(chart_todos, ITEMIDS_VENT_MODE, "_dummy")
    serie_vent["ventilacao_mecanica"] = 1
    serie_vent = serie_vent[["subject_id", "charttime", "ventilacao_mecanica"]]

    serie_rass = construir_serie_vital(chart_todos, ITEMIDS_RASS, "rass")

    espinha = asof_merge_vital(espinha, serie_temp, "temperatura_c", LOOKBACK_HORAS_VITAIS)
    espinha = asof_merge_vital(espinha, serie_vent, "ventilacao_mecanica", LOOKBACK_HORAS_VITAIS)
    espinha["ventilacao_mecanica"] = espinha["ventilacao_mecanica"].fillna(0).astype(int)
    espinha = asof_merge_vital(espinha, serie_rass, "rass", LOOKBACK_HORAS_VITAIS)

    # ---- vasoativos (reutiliza episódios já construídos) ----
    print("A reconstruir episódios de fármaco vasoativo (MV/CV/IV)...")
    eps_mv3 = v3.construir_episodios_mv(ids_iii) if ids_iii else pd.DataFrame()
    eps_cv3 = v3.construir_episodios_cv(ids_iii) if ids_iii else pd.DataFrame()
    eps_iv = v4.construir_episodios_iv(ids_iv) if ids_iv else pd.DataFrame()
    episodios = pd.concat([eps_mv3, eps_cv3, eps_iv], ignore_index=True)
    if len(episodios):
        episodios["subject_id_str"] = episodios.apply(
            lambda r: f"p{int(r['subject_id']):06d}" if r["subject_id"] in ids_iii else f"p{int(r['subject_id']):08d}",
            axis=1)
    espinha = construir_contagem_vasoativos(espinha, episodios)

    cols_saida = ["subject_id_str", "dataset", "record_id", "segment_order", "window_idx",
                  "temperatura_c", "ventilacao_mecanica", "rass",
                  "vasopressor_ativo", "n_vasoativos_ativos"]
    espinha[cols_saida].to_csv(SAIDA_CSV, index=False)

    print("\nCobertura:")
    for col in ["temperatura_c", "rass"]:
        pct = 100 * espinha[col].notna().mean()
        print(f"  {col}: {pct:.1f}% das janelas com valor")
    print(f"  ventilacao_mecanica=1: {100*espinha['ventilacao_mecanica'].mean():.1f}% das janelas")
    print(f"  vasopressor_ativo=1: {100*espinha['vasopressor_ativo'].mean():.1f}% das janelas")
    print(f"\nGuardado em: {SAIDA_CSV}")


if __name__ == "__main__":
    main()
