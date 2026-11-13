# 2_estimacao_absoluta/modelo_e_experiencias/nucleo

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`__init__.py`](__init__.py) | Marca a pasta como pacote Python. |
| [`avaliacao.py`](avaliacao.py) | Avaliação do absoluto: do desvio previsto à pressão, validação cruzada, holdout, bootstrap por sujeito e graus BHS. Um só pós-processamento para tudo. |
| [`busca.py`](busca.py) | Busca de hiperparâmetros (Optuna) para qualquer família e variante, com o mesmo objetivo em todas: o MAE do desvio na validação cruzada por sujeito (N_FOLDS dobras) sobre o desenvolvimento. |
| [`config.py`](config.py) | Configuração única do absoluto: caminhos, constantes do protocolo, listas de características e variantes. |
| [`dados.py`](dados.py) | Dados do absoluto: UMA só derivação, usada por todos os scripts. |
| [`modelos.py`](modelos.py) | Modelos do absoluto. Uma só função, `ajustar_e_prever`, treina e prevê qualquer família. É chamada igual pela validação cruzada (busca de hiperparâmetros), pela avaliação no holdout e pelos testes de sementes. |
| [`suavizacao.py`](suavizacao.py) | Suavização causal das previsões, por sessão: só olha para trás (nenhuma amostra usa o futuro). kalman filtro de Kalman de nível local. |
