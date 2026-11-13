# algoritmos/notebooks · os três modelos, de ponta a ponta

Um notebook por modelo, executados e com as saídas guardadas, por isso podem ler-se sem correr nada. Cada um segue o caminho
do sinal do MIMIC ao ficheiro `.joblib` que o servidor carrega e confere cada passo com o valor usado na dissertação.

| Notebook | Modelo | Pacotes | Na dissertação |
|---|---|---|---|
| [`01_estimacao_absoluta.ipynb`](01_estimacao_absoluta.ipynb) | Pressão sistólica absoluta (XGBoost, 27 características, ECG e PPG). | [`versoes_pacotes_absoluto.txt`](../versoes_pacotes_absoluto.txt) | Secção 6.3 |
| [`02_tendencia_classificador.ipynb`](02_tendencia_classificador.ipynb) | Classificador binário com que a regressão quantílica é comparada. | [`versoes_pacotes_tendencia.txt`](../versoes_pacotes_tendencia.txt) | Cap. 6 |
| [`03_tendencia_regressao_quantilica.ipynb`](03_tendencia_regressao_quantilica.ipynb) | Deteção de tendência adotada: regressão quantílica sobre a variação da pressão em 5 min, só PPG, 43 características. É o modelo instalado no servidor. | [`versoes_pacotes_tendencia.txt`](../versoes_pacotes_tendencia.txt) | Secções 6.4 e 6.5 |

Em [`saidas/`](saidas/) estão as figuras, os modelos exportados e as previsões no conjunto de teste que os notebooks produzem.

Para os voltar a correr são precisos os dados do MIMIC (ver o [`README.md`](../../README.md) da raiz) e os caminhos da primeira
célula de código ajustados. Os interruptores dessa célula, por omissão, não repetem a extração completa nem a pesquisa de
hiperparâmetros, que demoram várias horas.
