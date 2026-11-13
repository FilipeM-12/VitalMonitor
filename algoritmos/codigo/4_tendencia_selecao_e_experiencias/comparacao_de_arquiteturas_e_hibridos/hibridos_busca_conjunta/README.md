# 4_tendencia_selecao_e_experiencias/comparacao_de_arquiteturas_e_hibridos/hibridos_busca_conjunta

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`bootstrap_hibridos_vs_xgboost.py`](bootstrap_hibridos_vs_xgboost.py) | Verifica a afirmacao do Cap. 7: o ganho dos hibridos sobre o XGBoost e da ordem da incerteza de amostragem. |
| [`cv3_hibridos_vs_xgboost.py`](cv3_hibridos_vs_xgboost.py) | Verificação de robustez da comparação XGBoost contra híbridos na formulação de TENDÊNCIA, com validação cruzada de 3 dobras por sujeito sobre TODOS os sujeitos (desenvolvimento + holdout), 3 sementes por modelo. |
| [`hibrido_classificador_busca_conjunta.py`](hibrido_classificador_busca_conjunta.py) | Hibridos rede temporal + XGBoost na formulacao de TENDENCIA (causal, so-PPG), com busca CONJUNTA de hiperparametros (rede e XGBoost do topo afinados juntos). |
| [`hibrido_quantilico_busca_conjunta.py`](hibrido_quantilico_busca_conjunta.py) | Hibridos rede temporal + XGBoost na formulacao de TENDENCIA com REGRESSAO QUANTILICA sobre o Delta (causal, so-PPG), busca CONJUNTA de hiperparametros, e o XGBoost quantilico tabular como referencia, tudo sobre as MESMAS linhas. |
| [`sementes_hibrido_quantilico.py`](sementes_hibrido_quantilico.py) | Sementes adicionais dos híbridos quantílicos, para saber se o ganho sobre o XGBoost resiste à aleatoriedade do treino. |
| [`sensibilidade_custo_igualado_quantilica.py`](sensibilidade_custo_igualado_quantilica.py) | Sensibilidade por episódio a custo igualado, XGBoost contra híbridos, na regressão quantílica (holdout, 44 sujeitos, 3 sementes). |
| [`sistema_completo_quantilica.py`](sistema_completo_quantilica.py) | Comparação do SISTEMA COMPLETO (score + portão de estabilidade + limiar), XGBoost contra híbridos, na regressão quantílica. Por direção, para os três modelos. |
