# 3_tendencia_modelo_final/servidor_tempo

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`06_desvio_reserva_morf.py`](06_desvio_reserva_morf.py) | Igual a 5_servidor_e_carga/verificacao_da_extracao_e_derivacao_contra_o_treino/06_desvio_de_reserva_tendencia.py. |
| [`comparar_dispositivo_mimic.py`](comparar_dispositivo_mimic.py) | As características novas têm no dispositivo a mesma variabilidade (dentro da sessão) que no MIMIC? O modelo vê cada característica normalizada pela calibração da sessão (z-score dos 10 primeiros blocos). |
| [`ensaio_worker_tempo.py`](ensaio_worker_tempo.py) | Ensaio ponta a ponta do worker da tendência (janela por tempo exato, 43 características, extracao_ppg + derivacao_ppg + modelo_tendencia_ppg_quant_tempo.joblib). |
| [`exportar_modelo_tempo.py`](exportar_modelo_tempo.py) | Exporta o modelo final da tendência (XGBoost quantílico, 43 características, janela de 5 min por tempo exato) para o worker. |
| [`figura_ensaio_sem_ecg_morf.py`](figura_ensaio_sem_ecg_morf.py) | Figura do ensaio sem ECG com o worker da tendência (um só detetor, 43 caracteristicas). Painel de cima: janelas aceites (sonda_extracao_ecg_morf.py, 20 min de sinal real de bancada). |
| [`sonda_extracao_ecg_morf.py`](sonda_extracao_ecg_morf.py) | Sonda da extração de características com e sem o canal de ECG, com a extração só-PPG do worker instalado (extracao_ppg, 43 características). |
| [`verificar_extracao_morf.py`](verificar_extracao_morf.py) | A extração do servidor (extracao_ppg.py) devolve EXATAMENTE o que a do treino (09b) devolve? Para N registos do MIMIC. |
| [`verificar_inferencia_tempo.py`](verificar_inferencia_tempo.py) | As previsões do modelo de tendência INSTALADO no servidor coincidem com as do treino? Entrada: as 43 características da tabela de teste do treino (tendencia_tempo_fixo. |
| [`verificar_servidor_tempo.py`](verificar_servidor_tempo.py) | Derivação do servidor (derivacao_ppg.py) contra a tabela da tendência do treino, no conjunto de teste inteiro. |

Subpastas: [`ensaio_worker_600s/`](ensaio_worker_600s/), [`teste_de_carga_tempo/`](teste_de_carga_tempo/).
