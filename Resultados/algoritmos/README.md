# algoritmos · os resultados dos modelos

Os números por trás das tabelas dos Capítulos 6 e 7 da dissertação, para os dois modelos do servidor. Cada parte tem a sua pasta, com um `README.md` que diz o que há em cada ficheiro e que script o escreveu. Os da tendência são também cópias dos que
estão em `algoritmos/codigo/`, ao lado dos scripts.

| Parte | Onde estão | Na dissertação |
|---|---|---|
| Estimação absoluta | [`absoluto/`](absoluto/): o MAE de validação cruzada das cinco arquiteturas e das 20 tentativas de cada uma, e as tabelas dos scripts de `modelo_e_experiencias/`. | Secção 6.3 |
| Tendência | [`tendencia/`](tendencia/): o AUC de validação cruzada e no teste das arquiteturas, os resultados da deteção com o modelo adotado, as escolhas feitas no desenvolvimento, o valor do ECG e as verificações do servidor. | Secções 6.4 e 6.5, Apêndice C |
| Comparação com o estudo de Finnegan, primeiros 30 min | [`primeiros_30_min_e_finnegan/`](../../algoritmos/codigo/2_estimacao_absoluta/primeiros_30_min_e_finnegan/) | Secção 6.3 |

Cada pasta de `algoritmos/codigo/` tem um `README.md` com o que faz cada script. Os dados do MIMIC não estão no repositório.
