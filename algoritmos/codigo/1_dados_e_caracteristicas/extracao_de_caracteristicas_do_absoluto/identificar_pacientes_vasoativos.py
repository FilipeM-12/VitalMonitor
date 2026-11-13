"""
Identifica, entre os pacientes com dados reais em MIMIC_DATASET_CONSOLIDADO, quais
receberam fármacos vasoativos (vasopressores ou vasodilatadores) durante o
internamento -- candidatos a descartar, porque estes fármacos alteram diretamente
o tónus arterial e podem quebrar a relação PAT/PWV <-> pressão arterial que o
modelo assume.

Fontes: INPUTEVENTS_CV.csv / INPUTEVENTS_MV.csv do MIMIC-III (ITEMID confirmados
a partir de D_ITEMS.csv nesta sessão).
"""
import os
import re
import pandas as pd

PASTA_DATASET = "/media/filipe/Dados/MIMIC_DATASET_CONSOLIDADO"
PASTA_MIMIC = "/media/filipe/Filipe/Datasets/mimic-iii-1.4"
SAIDA_CSV = "../../06_FEATURES_DATASETS/pacientes_vasoativos.csv"

# ITEMID confirmados via D_ITEMS.csv (LINKSTO = inputevents_cv / inputevents_mv)
ITEMIDS = {
    # vasopressores
    221906: ("Norepinefrina", "vasopressor", "mv"),
    30047:  ("Norepinefrina (Levophed)", "vasopressor", "cv"),
    30120:  ("Norepinefrina (Levophed-k)", "vasopressor", "cv"),
    221289: ("Epinefrina/Adrenalina", "vasopressor", "mv"),
    30044:  ("Epinefrina/Adrenalina", "vasopressor", "cv"),
    30119:  ("Epinefrina/Adrenalina-k", "vasopressor", "cv"),
    30309:  ("Epinefrina/Adrenalina (drip)", "vasopressor", "cv"),
    221662: ("Dopamina", "vasopressor", "mv"),
    30043:  ("Dopamina", "vasopressor", "cv"),
    30307:  ("Dopamina (drip)", "vasopressor", "cv"),
    221653: ("Dobutamina", "vasopressor", "mv"),
    30042:  ("Dobutamina", "vasopressor", "cv"),
    30306:  ("Dobutamina (drip)", "vasopressor", "cv"),
    221749: ("Fenilefrina", "vasopressor", "mv"),
    222315: ("Vasopressina", "vasopressor", "mv"),
    30051:  ("Vasopressina", "vasopressor", "cv"),
    # vasodilatadores (mencionados pelo médico)
    222056: ("Nitroglicerina", "vasodilatador", "mv"),
    30049:  ("Nitroglicerina", "vasodilatador", "cv"),
    30121:  ("Nitroglicerina-k", "vasodilatador", "cv"),
    222051: ("Nitroprussiato de sódio", "vasodilatador", "mv"),
    30050:  ("Nitroprussiato de sódio", "vasodilatador", "cv"),
}


def obter_pacientes_com_dados():
    """Devolve o set de subject_id (int) cujas pastas em MIMIC_DATASET_CONSOLIDADO
    não estão vazias (i.e. têm ficheiros .hea/.dat reais)."""
    subject_ids = set()
    for nome in os.listdir(PASTA_DATASET):
        caminho = os.path.join(PASTA_DATASET, nome)
        if not os.path.isdir(caminho):
            continue
        m = re.match(r"^p0*(\d+)$", nome)
        if not m:
            continue
        if os.listdir(caminho):  # não vazia
            subject_ids.add(int(m.group(1)))
    return subject_ids


def varrer_inputevents(nome_ficheiro, col_subject, subject_ids):
    achados = []
    usecols = ["SUBJECT_ID", "ITEMID"]
    caminho = os.path.join(PASTA_MIMIC, nome_ficheiro)
    for chunk in pd.read_csv(caminho, usecols=usecols, chunksize=2_000_000, low_memory=False):
        chunk = chunk[chunk["SUBJECT_ID"].isin(subject_ids) & chunk["ITEMID"].isin(ITEMIDS)]
        if len(chunk):
            achados.append(chunk)
    if not achados:
        return pd.DataFrame(columns=["SUBJECT_ID", "ITEMID"])
    return pd.concat(achados, ignore_index=True)


def main():
    pacientes = obter_pacientes_com_dados()
    print(f"Pacientes com dados reais em MIMIC_DATASET_CONSOLIDADO: {len(pacientes)}")

    print("A varrer INPUTEVENTS_CV.csv (pode demorar alguns minutos)...")
    cv = varrer_inputevents("INPUTEVENTS_CV.csv", "SUBJECT_ID", pacientes)
    print(f"  {len(cv)} administrações relevantes encontradas (CV).")

    print("A varrer INPUTEVENTS_MV.csv...")
    mv = varrer_inputevents("INPUTEVENTS_MV.csv", "SUBJECT_ID", pacientes)
    print(f"  {len(mv)} administrações relevantes encontradas (MV).")

    todos = pd.concat([cv, mv], ignore_index=True)
    todos["farmaco"] = todos["ITEMID"].map(lambda i: ITEMIDS[i][0])
    todos["tipo"] = todos["ITEMID"].map(lambda i: ITEMIDS[i][1])

    resumo = (
        todos.groupby("SUBJECT_ID")
        .agg(farmacos=("farmaco", lambda s: sorted(set(s))),
             tipos=("tipo", lambda s: sorted(set(s))),
             n_administracoes=("ITEMID", "count"))
        .reset_index()
        .rename(columns={"SUBJECT_ID": "subject_id"})
        .sort_values("subject_id")
    )
    resumo.to_csv(SAIDA_CSV, index=False)

    print(f"\n{len(resumo)} de {len(pacientes)} pacientes ({100*len(resumo)/len(pacientes):.1f}%) "
          f"receberam pelo menos um fármaco vasoativo -- candidatos a descarte/flag.")
    print(f"Resultado detalhado guardado em: {SAIDA_CSV}")


if __name__ == "__main__":
    main()
