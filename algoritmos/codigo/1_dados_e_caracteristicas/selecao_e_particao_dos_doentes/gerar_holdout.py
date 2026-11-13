"""
Gera o conjunto de holdout (pacientes_holdout.txt) por amostragem aleatória
estratificada (faixa etária x género), com seed fixa para reprodutibilidade.

Substitui o antigo `pacientes_teste.txt`, que não foi criado com este fim —
era uma lista pequena usada para testar um script de download de segmentos
adjacentes no PhysioNet (ver Seleção_subjects.ipynb), reaproveitada mais
tarde por coincidência de nome quando o mecanismo de holdout foi introduzido.

Corre este script sempre que precisares de regenerar o holdout (ex.: se a
coorte válida mudar). O ficheiro gerado é determinístico dado o mesmo
SEED e a mesma coorte de entrada.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

PASTA_FEATURES = "/media/filipe/Dados/Features"
FICHEIROS_BASELINE = [
    f"{PASTA_FEATURES}/features_homens_baseline.csv",
    f"{PASTA_FEATURES}/features_mulheres_baseline.csv",
]
FRACAO_HOLDOUT = 0.14   # ~14 % dos doentes
SEED = 42
FICHEIRO_SAIDA = "../../03_ABSOLUTO_codigo_e_resultados/RESULTADOS/pacientes_holdout.txt"

dfs = [pd.read_csv(f, usecols=["subject_id", "gender", "age"]) for f in FICHEIROS_BASELINE]
df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="subject_id").reset_index(drop=True)

df["faixa_etaria"] = pd.cut(df["age"], bins=[0, 40, 60, 80, 100])
df["estrato"] = df["faixa_etaria"].astype(str) + "_" + df["gender"].astype(str)

print(f"Coorte total (baseline, únicos): {len(df)} pacientes")
print(df.groupby(["faixa_etaria", "gender"], observed=True).size())

dev_ids, holdout_ids = train_test_split(
    df["subject_id"],
    test_size=FRACAO_HOLDOUT,
    stratify=df["estrato"],
    random_state=SEED,
)

holdout_ids = sorted(holdout_ids.tolist())
print(f"\nHoldout gerado: {len(holdout_ids)} pacientes ({100 * len(holdout_ids) / len(df):.1f}%)")

df_holdout = df[df["subject_id"].isin(holdout_ids)]
print("\nDistribuição do holdout (para conferir representatividade):")
print(df_holdout.groupby(["faixa_etaria", "gender"], observed=True).size())
print("\nDistribuição proporcional coorte total vs. holdout:")
prop_total = df.groupby(["faixa_etaria", "gender"], observed=True).size() / len(df)
prop_holdout = df_holdout.groupby(["faixa_etaria", "gender"], observed=True).size() / len(df_holdout)
print(pd.DataFrame({"total_%": (prop_total * 100).round(1), "holdout_%": (prop_holdout * 100).round(1)}))

with open(FICHEIRO_SAIDA, "w") as f:
    f.write("\n".join(holdout_ids))

print(f"\nGuardado em: {FICHEIRO_SAIDA}")
