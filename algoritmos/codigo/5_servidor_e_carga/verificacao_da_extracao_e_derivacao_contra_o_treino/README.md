# 5_servidor_e_carga/verificacao_da_extracao_e_derivacao_contra_o_treino

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`01_verificar_extracao.py`](01_verificar_extracao.py) | Equivalência numérica entre a extração de TREINO e a de PRODUÇÃO (PPG puro). |
| [`04_verificar_derivacao_causal.py`](04_verificar_derivacao_causal.py) | Derivação CAUSAL, 20 características: TREINO (vetorizada) contra PRODUÇÃO (incremental). |
| [`05_verificar_servidor_teste_completo.py`](05_verificar_servidor_teste_completo.py) | Derivação da tendência: servidor contra treino no conjunto de teste inteiro. O 04 verifica a derivação em 5 registos, cada um tratado como um conjunto de dados à parte. |
| [`06_desvio_de_reserva_tendencia.py`](06_desvio_de_reserva_tendencia.py) | Desvio de reserva da normalização causal da tendência, para o servidor. normalizar_causal_calibracao (suavizacao_e_normalizacao_causal.py) divide pelo desvio da janela de calibração da sessão mais 1e-6 e, quando o desvio é nulo ou indefinido. |
