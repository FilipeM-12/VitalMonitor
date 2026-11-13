# 3_tendencia_modelo_final

A cadeia final da deteção de tendência, a do modelo instalado no servidor.

| Pasta | O que faz |
|---|---|
| [`extracao_morfologia/`](extracao_morfologia/) | Extração das características de forma do pulso (W25, W75, declive máximo da subida, índice de reflexão) e verificação contra a extração anterior. |
| [`tendencia_tempo_fixo/`](tendencia_tempo_fixo/) | Tabela com 43 características e janela de 5 min ancorada no tempo, afinação do regressor quantílico (os hiperparâmetros do modelo instalado), classificador contra regressão quantílica, ponto de operação com intervalos por bootstrap de doentes, e a subpasta `hibridos/` com os híbridos e as redes puras. |
| [`comparacao_janela/`](comparacao_janela/) e [`resultados_comparacoes/`](resultados_comparacoes/) | Comparação da janela ancorada no tempo com a de 20 blocos, com os mesmos hiperparâmetros. A comparação das características novas é o [`tendencia_tempo_fixo/comparar_caracteristicas_novas.py`](tendencia_tempo_fixo/comparar_caracteristicas_novas.py). Os registos de ambas estão em `resultados_comparacoes/`. |
| [`janela_desenvolvimento/`](janela_desenvolvimento/) | Escolhas feitas só no desenvolvimento: janela de 2, 3, 5 e 10 min, nível do quantil, custo da união dos dois detetores e permanência das variações. |
| [`cascata_tempo/`](cascata_tempo/) | Cascata de janelas de 2, 3 e 5 min com o modelo final. |
| [`ablacao_ecg_tempo/`](ablacao_ecg_tempo/) e [`ablacao_ecg_todas/`](ablacao_ecg_todas/) | Quanto vale o ECG com o modelo final, só no desenvolvimento, e com todas as suas variações. |
| [`resultados_deteccao/`](resultados_deteccao/) | Resultados da deteção com o modelo final (limiares, AUC, robustez, união, sessões sem evento, importância das características, vias) e as figuras respetivas. |
| [`servidor_tempo/`](servidor_tempo/) | A extração e a derivação instaladas no servidor, exportação do modelo, verificações contra o treino, ensaio ponta a ponta, ensaio sem ECG e, em `teste_de_carga_tempo/`, o teste de carga com o worker da tendência. |
