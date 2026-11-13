# 1_dados_e_caracteristicas/extracao_ppg_e_segunda_derivada_da_tendencia

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`01_extrair_features_ppg_puro.py`](01_extrair_features_ppg_puro.py) | Extração de features SÓ a partir do PPG, sem ECG em lado nenhum. |
| [`07_correlacao.py`](07_correlacao.py) | Funções de correlação no espaço de variação (Parte B), sobre o dataset PPG-PURO. Usa só dados que um dispositivo sem elétrodos consegue produzir. |
| [`09_extrair_features_ppg_apg.py`](09_extrair_features_ppg_apg.py) | Extração PPG-pura + features APG (2.ª derivada do PPG). |
