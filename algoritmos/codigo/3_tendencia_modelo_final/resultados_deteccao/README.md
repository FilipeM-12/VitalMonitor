# 3_tendencia_modelo_final/resultados_deteccao

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`resultados_deteccao.py`](resultados_deteccao.py) | Números e figuras dos «resultados da deteção» do Capítulo 6, para o modelo final (43 características, janela de 5 min por tempo exato). |
| [`vias_tempo.py`](vias_tempo.py) | Vias exploradas para melhorar a deteção, calculadas com o modelo final (43 características, janela de 5 min por tempo exato). |
| [`w25_truncagem.py`](w25_truncagem.py) | O modelo comporta-se pior nos blocos em que a W25 está truncada? Usa as previsões em cache de resultados_deteccao.py (previsoes.npz): fora da dobra no desenvolvimento e no conjunto de teste. |
