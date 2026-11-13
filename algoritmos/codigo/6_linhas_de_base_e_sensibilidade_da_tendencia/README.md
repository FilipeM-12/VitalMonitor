# 6_linhas_de_base_e_sensibilidade_da_tendencia

O que ainda sustenta números da dissertação da cadeia anterior à de 43 características, que ficou substituída. O modelo
instalado vem de [`../3_tendencia_modelo_final/`](../3_tendencia_modelo_final/).

| Pasta ou ficheiro | O que faz |
|---|---|
| [`previsoes_e_sensibilidade_do_detetor/`](previsoes_e_sensibilidade_do_detetor/) | As previsões do regressor causal no conjunto de teste, a função que mede a sensibilidade por episódio, a sensibilidade às convenções do episódio (intervalo de fusão e crédito de antecipação) e a comparação com o alarme aleatório de igual estrutura, que dá a Tabela 6.7. |
| [`uniao_isolada.py`](uniao_isolada.py) | Ciclo de serviço da união dos dois detetores quando cada um é afinado isoladamente, citado no Cap. 6. |
