# 4_tendencia_selecao_e_experiencias/submodelos_do_alarme_e_do_estavel

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`alarmes_e_estavel.py`](alarmes_e_estavel.py) | Dois submodelos, um do que é ESTÁVEL e outro do que é ALARME, ajudam o modelo principal? Submodelos (treinados com previsões FORA DA AMOSTRA no desenvolvimento, 3 dobras por doente. |
| [`dois_submodelos.py`](dois_submodelos.py) | Dois submodelos independentes, um do ALARME e outro do ESTÁVEL, decidem pela comparação. Versão fiel da ideia (alarmes_e_estavel.py usava as pontuações como características extra de UM modelo supervisionado). |
