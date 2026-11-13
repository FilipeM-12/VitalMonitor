# 2_estimacao_absoluta/primeiros_30_min_e_finnegan

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`diagnostico_30min.py`](diagnostico_30min.py) | Dois diagnósticos nas previsões do holdout (XGB baseline, 27 caract.), primeiros 30 min após a calibração, para perceber porque o modelo não bate a âncora. |
| [`erro_ate_30min_vs_finnegan.py`](erro_ate_30min_vs_finnegan.py) | Erro do modelo absoluto (XGB baseline, 27 caract., holdout) nos primeiros 30 min depois da calibração, para comparar com Finnegan et al. 2023 (Sci Rep 13:986), cujas sessões duram ~30 min. |
