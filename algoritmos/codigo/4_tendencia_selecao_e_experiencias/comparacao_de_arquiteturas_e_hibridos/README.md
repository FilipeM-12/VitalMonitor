# 4_tendencia_selecao_e_experiencias/comparacao_de_arquiteturas_e_hibridos

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`planalto_arquiteturas.py`](planalto_arquiteturas.py) | Planalto da afinação na tendência (sobre o orçamento de 20 tentativas): para cada estudo do Optuna, o melhor AUC de validação cruzada até à tentativa 5, 10, 15 e 20, e quanto melhora entre a 5.ª e a 20.ª. Uso. |
| [`comparacao_arquiteturas_classificador.py`](comparacao_arquiteturas_classificador.py) | Comparacao de arquiteturas para deteccao de tendencia (classificador binario): XGBoost tabular, BiLSTM, BiTCN e hibridos, com 20 trials de Optuna por arquitetura (GroupKFold(3) por sujeito). |
| [`tabela_sementes_hibridos.py`](tabela_sementes_hibridos.py) | Tabela do AUC no teste, por semente, do XGBoost e dos híbridos com regressão quantílica (a formulação adotada), nas duas direções (o que varia entre sementes e a maior variabilidade dos híbridos). |

Subpastas: [`hibridos_busca_conjunta/`](hibridos_busca_conjunta/).
