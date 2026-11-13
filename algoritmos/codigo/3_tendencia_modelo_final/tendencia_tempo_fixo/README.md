# 3_tendencia_modelo_final/tendencia_tempo_fixo

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`01_optuna_tempo.py`](01_optuna_tempo.py) | Reafinação do XGBoost quantílico da tendência com a janela de 5 min ANCORADA NO TEMPO. Mesmo espaço de pesquisa, mesmo objetivo (AUC de deteção sobre previsões fora da amostra, GroupKFold de 3 por sujeito, folds fixos). |
| [`02_classificador_vs_quantilica.py`](02_classificador_vs_quantilica.py) | Classificador contra regressão quantílica (XGBoost), com a janela de 5 min ANCORADA NO TEMPO. Repete a comparação da secção «classificação ou regressão quantílica» na definição final: as mesmas 43 características. |
| [`comparar_caracteristicas_novas.py`](comparar_caracteristicas_novas.py) | Com a janela de 5 min ANCORADA NO TEMPO, que características novas ajudam? (A/B, mesmos hiperparâmetros) Usa a tabela da tendência (construir_tabela_ppg_morf.py: as mesmas linhas da antiga. |
| [`bootstrap_ponto.py`](bootstrap_ponto.py) | Intervalos de confiança (bootstrap de DOENTES) da sensibilidade, do VPP e da carga nos pontos de operação já escolhidos. |
| [`construir_tabela_ppg_morf.py`](construir_tabela_ppg_morf.py) | Tabela da tendência com as características novas. |
| [`ponto_tempo.py`](ponto_tempo.py) | Sensibilidade e VPP a 10 min/h de falso alarme (união), janela de 5 min por TEMPO, com e sem as características novas. |
| [`tempo_fixo.py`](tempo_fixo.py) | Definição da tendência com a janela de 5 min ANCORADA NO TEMPO. Na definição por posição, a variação a 5 min era PA(bloco i) - PA(bloco i-20) e os Δ das características a 2, 5 e 10 min eram shift(8), shift(20) e shift(40) sobre os blocos de cada sessão. |

Subpastas: [`dados/`](dados/), [`hibridos/`](hibridos/).
