# 6_linhas_de_base_e_sensibilidade_da_tendencia/previsoes_e_sensibilidade_do_detetor

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`09_previsoes_causais.py`](09_previsoes_causais.py) | Gera as previsoes do modelo CAUSAL sobre o holdout CAUSAL. |
| [`17_numeros_capitulo5.py`](17_numeros_capitulo5.py) | Recalcula, na variante CAUSAL, os números do Capítulo 5 que faltavam. |
| [`22_sensibilidade_gap_credito.py`](22_sensibilidade_gap_credito.py) | Sensibilidade das duas convenções do nível de episódio: o intervalo de fusão (gap) e o crédito de antecipação. |
| [`23_nulo_por_credito.py`](23_nulo_por_credito.py) | Modelo contra alarme aleatório de igual estrutura, COM portão, ao ponto de operação, na grelha gap x crédito de 0,5 / 1 / 2 / 3 min. Reutiliza o 17. |
