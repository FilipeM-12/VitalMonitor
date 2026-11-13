# 3_tendencia_modelo_final/servidor_tempo/teste_de_carga_tempo

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`correr_repeticoes.sh`](correr_repeticoes.sh) | 3 repeticoes independentes entre 400 e 700 pulseiras, os tres modelos em sequencia (nunca em paralelo). |
| [`correr_tudo.sh`](correr_tudo.sh) | Corre os tres modelos em sequencia (nunca em paralelo: partilhariam CPU e falseariam a latencia). |
| [`figura_carga.py`](figura_carga.py) | Figura e resumo do teste de carga com o worker da tendência. Níveis 10-200 e 400-500: mediana das 3 repetições limpas quando existem, senão da corrida grossa. |
| [`teste_de_carga_tempo.py`](teste_de_carga_tempo.py) | Worker de tendencia instalado (servidor/workers): extracao so-PPG com as caracteristicas morfologicas (extracao_ppg), derivacao causal por tempo exato (derivacao_ppg, 43 caracteristicas. |
