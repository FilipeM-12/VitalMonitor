"""w25_truncagem.py — o modelo comporta-se pior nos blocos em que a W25 está truncada?
Usa as previsões em cache de resultados_deteccao.py (previsoes.npz): fora da dobra no desenvolvimento e no conjunto de teste.
Grupos por bloco (15 s): w25_trunc = fração de batimentos do bloco com a W25 truncada; 'sem' = 0, 'com' = > 0, e terços dos que têm.
Métricas: prevalência de eventos, AUC, erro absoluto médio e correlação de Δ previsto com Δ real, FC média do grupo.
IC a 95 % por bootstrap de doentes (200 repetições), só para 'todos', 'sem truncagem', 'com truncagem' e a diferença; os terços só têm valor pontual."""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

AQUI = os.path.dirname(os.path.abspath(__file__))
A = "/home/filipe/Desktop/Tese/algoritmo-ml"
TF = f"{A}/Tentativa_22_10/tendencia_tempo_fixo"
sys.path.insert(0, TF)
import tempo_fixo as tf
DIRS = ("descida", "subida"); SINAL = {"descida": -1.0, "subida": 1.0}
z = np.load(f"{AQUI}/previsoes.npz")
out = {}
rng = np.random.default_rng(0)
NB = 200
for nome in ("dev", "holdout"):
    g, V = tf.preparar_grade(f"{TF}/dados", nome)
    ok = g.delta.notna().values
    G = g[ok].reset_index(drop=True)
    P = {d: z[("oof_" if nome == "dev" else "teste_") + d] for d in DIRS}
    # a grade não traz as colunas de diagnóstico: vêm da tabela, pela chave (sessão, bloco)
    T = pd.read_csv(f"{TF}/dados/{nome}_ppg.csv", usecols=["session_id", "bloco_id", "w25_trunc", "w25", "hr_mean"]).drop_duplicates(["session_id", "bloco_id"])
    G = G.drop(columns=[c for c in ("w25_trunc", "w25", "hr_mean") if c in G.columns]).merge(T, on=["session_id", "bloco_id"], how="left", validate="1:1")
    G["trunc"] = G["w25_trunc"]
    G["grupo"] = np.where(G.trunc.isna(), "w25 em falta", np.where(G.trunc == 0, "sem truncagem", "com truncagem"))
    com = G.trunc > 0
    q = G.loc[com, "trunc"].quantile([1 / 3, 2 / 3]).values
    G.loc[com, "grupo"] = np.where(G.loc[com, "trunc"] <= q[0], "com truncagem: terço baixo",
                           np.where(G.loc[com, "trunc"] <= q[1], "com truncagem: terço médio", "com truncagem: terço alto"))
    G["tudo"] = "todos"
    res = {}
    subj = G.subject_id.values; usub = np.unique(subj)
    idx_por = {s: np.where(subj == s)[0] for s in usub}
    for grp in ["todos", "sem truncagem", "com truncagem", "com truncagem: terço baixo", "com truncagem: terço médio",
                "com truncagem: terço alto", "w25 em falta"]:
        m = ((G.tudo == "todos") if grp == "todos" else (G.grupo == grp) if "terço" not in grp and grp != "com truncagem" else
             (G.grupo.str.startswith("com truncagem") if grp == "com truncagem" else G.grupo == grp)).values
        if m.sum() < 50:
            continue
        r = dict(n_blocos=int(m.sum()), n_doentes=int(len(np.unique(subj[m]))), fc_media=float(G.hr_mean[m].mean()), w25_media_s=float(G.w25[m].mean()))
        for d in DIRS:
            y = G[f"y_{d}"].values; p = P[d]; dl = G.delta.values
            def aucf(ix):
                return roc_auc_score(y[ix], p[ix]) if 0 < y[ix].sum() < len(ix) else np.nan
            def mae(ix): return np.mean(np.abs(SINAL[d] * p[ix] - dl[ix]))
            ix = np.where(m)[0]
            r[d] = dict(prevalencia=float(y[ix].mean()), auc=float(aucf(ix)), mae=float(mae(ix)),
                        corr=float(np.corrcoef(SINAL[d] * p[ix], dl[ix])[0, 1]))
            bs = []
            for _ in range(NB if grp in ('todos', 'sem truncagem', 'com truncagem') else 0):
                ss = rng.choice(usub, len(usub))
                ib = np.concatenate([idx_por[s] for s in ss]); ib = ib[m[ib]]
                bs.append((aucf(ib), mae(ib)))
            bs = np.array(bs)
            r[d]["auc_ic"] = [float(np.nanpercentile(bs[:, 0], 2.5)), float(np.nanpercentile(bs[:, 0], 97.5))] if len(bs) else [np.nan, np.nan]
            r[d]["mae_ic"] = [float(np.nanpercentile(bs[:, 1], 2.5)), float(np.nanpercentile(bs[:, 1], 97.5))] if len(bs) else [np.nan, np.nan]
        res[grp] = r
    # diferença sem - com, por bootstrap de doentes
    dif = {}
    sem = (G.grupo == "sem truncagem").values; comm = G.grupo.str.startswith("com truncagem").values
    for d in DIRS:
        y = G[f"y_{d}"].values; p = P[d]
        def f(ix_s, ix_c):
            return roc_auc_score(y[ix_s], p[ix_s]) - roc_auc_score(y[ix_c], p[ix_c])
        bs = []
        for _ in range(NB):
            ss = rng.choice(usub, len(usub)); ib = np.concatenate([idx_por[s] for s in ss])
            a, b = ib[sem[ib]], ib[comm[ib]]
            if 0 < y[a].sum() < len(a) and 0 < y[b].sum() < len(b):
                bs.append(f(a, b))
        a, b = np.where(sem)[0], np.where(comm)[0]
        dif[d] = dict(auc_sem_menos_com=float(f(a, b)), ic=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    res["_diferenca_auc"] = dif
    res["_fracao_blocos_com_truncagem"] = float(comm.sum() / (sem.sum() + comm.sum()))
    out[nome] = res
json.dump(out, open(f"{AQUI}/w25_truncagem.json", "w"), indent=1, ensure_ascii=False)
for nome, res in out.items():
    print("==", nome, "fração com truncagem", round(res["_fracao_blocos_com_truncagem"], 3), "dif AUC (sem − com)", res["_diferenca_auc"])
    for grp, r in res.items():
        if grp.startswith("_"): continue
        print(f"{grp:30s} n={r['n_blocos']:6d} FC={r['fc_media']:.0f} W25={r['w25_media_s']:.3f} | " + " | ".join(
            f"{d[:4]} prev {r[d]['prevalencia']:.3f} AUC {r[d]['auc']:.3f} [{r[d]['auc_ic'][0]:.3f},{r[d]['auc_ic'][1]:.3f}] MAE {r[d]['mae']:.2f} r {r[d]['corr']:.2f}" for d in DIRS))
