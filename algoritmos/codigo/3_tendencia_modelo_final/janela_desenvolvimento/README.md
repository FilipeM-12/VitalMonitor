# 3_tendencia_modelo_final/janela_desenvolvimento

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`classificador_nivel_dev.py`](classificador_nivel_dev.py) | A referência do classificador para a tabela dos níveis do quantil. |
| [`janela_dev.py`](janela_dev.py) | Escolha da janela temporal W (2, 3, 5 e 10 min) SÓ COM O DESENVOLVIMENTO (a janela não pode ser escolhida com os dados de teste). |
| [`nivel_quantil_dev.py`](nivel_quantil_dev.py) | Qual o nível do quantil (tau) a usar? SÓ NO DESENVOLVIMENTO (as escolhas fazem-se no desenvolvimento). |
| [`nivel_uniao_dev.py`](nivel_uniao_dev.py) | Tabela dos níveis do quantil (e do classificador) com o custo da UNIÃO dos dois detetores (substitui a leitura da tabela de nivel_quantil_dev.py, que mede cada detetor sozinho). |
| [`persistencia_dev.py`](persistencia_dev.py) | Persistência dos episódios por janela temporal W (2, 3, 5 e 10 min), SÓ NO DESENVOLVIMENTO e com a janela por tempo exato. |
