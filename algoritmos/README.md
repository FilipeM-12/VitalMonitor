# algoritmos · modelos e resultados da dissertação

| Pasta ou ficheiro | O que é |
|---|---|
| [`notebooks/`](notebooks/) | **Ponto de entrada.** Três notebooks executados, um por modelo, que seguem o caminho completo do sinal do MIMIC ao ficheiro `.joblib` que o servidor carrega e conferem cada passo com o valor usado na dissertação. |
| [`notebooks/01_estimacao_absoluta.ipynb`](notebooks/01_estimacao_absoluta.ipynb) | Estimação da pressão sistólica absoluta (XGBoost, 27 características, ECG e PPG). Secção 6.3. |
| [`notebooks/03_tendencia_regressao_quantilica.ipynb`](notebooks/03_tendencia_regressao_quantilica.ipynb) | Deteção de tendência adotada: regressão quantílica sobre a variação da pressão em 5 min, só PPG, 43 características. Secções 6.4 e 6.5. É o modelo instalado no servidor. |
| [`notebooks/02_tendencia_classificador.ipynb`](notebooks/02_tendencia_classificador.ipynb) | O classificador binário com que a regressão quantílica é comparada no Cap. 6. |
| [`notebooks/saidas/`](notebooks/saidas/) | Figuras, modelos exportados e previsões no conjunto de teste produzidos pelos notebooks. |
| [`codigo/`](codigo/) | Todos os scripts que produzem os números, as tabelas e as figuras, com a estrutura de pastas original do desenvolvimento. O [`README.md`](codigo/README.md) dessa pasta identifica cada subpasta. |
| [`../Resultados/algoritmos/`](../Resultados/algoritmos/) | Os resultados que esses scripts escreveram, agrupados pelo que respondem, com o MAE de validação cruzada das cinco arquiteturas do absoluto e o AUC das da tendência. |
| [`versoes_pacotes_absoluto.txt`](versoes_pacotes_absoluto.txt) | Versões dos pacotes Python usadas no treino do modelo absoluto (notebook 01): XGBoost 3.2.0, scikit-learn 1.8.0. |
| [`versoes_pacotes_tendencia.txt`](versoes_pacotes_tendencia.txt) | Versões usadas na tendência (notebooks 02 e 03): XGBoost 3.3.0, scikit-learn 1.9.1. Com outra versão do XGBoost o AUC muda na quarta casa decimal. |

Os notebooks têm interruptores na primeira célula de código. Por omissão não repetem a extração completa do MIMIC nem
a pesquisa de hiperparâmetros, que demoram várias horas. Os dados do MIMIC não estão no repositório (ver o
[`README.md`](../README.md) da raiz).
