# tendencia · os resultados da deteção de tendência

Secções 6.4 e 6.5 da dissertação. São cópias dos ficheiros que os scripts de [`algoritmos/codigo/`](../../../algoritmos/codigo/) escreveram, nas pastas
[`3_tendencia_modelo_final/`](../../../algoritmos/codigo/3_tendencia_modelo_final/) e [`4_tendencia_selecao_e_experiencias/`](../../../algoritmos/codigo/4_tendencia_selecao_e_experiencias/),
agrupadas aqui pelo que respondem. O modelo adotado é o regressor quantílico do XGBoost, só com PPG, com 43 características e a janela de 5 min ancorada no
tempo. Cada pasta de `algoritmos/codigo/` tem um `README.md` com o que faz cada script.

## A escolha entre arquiteturas

As redes BiLSTM e BiTCN, sozinhas e em híbrido com o XGBoost, compararam-se com o XGBoost quantílico com 20 tentativas do Optuna por modelo. O AUC de CV é o da
melhor tentativa, calculado sobre previsões fora da amostra, com validação cruzada de 3 dobras agrupadas por doente, só no desenvolvimento. O AUC no teste é o
do conjunto de teste, que não entra na escolha.

| Modelo | Descida, AUC CV | Descida, AUC teste | Subida, AUC CV | Subida, AUC teste |
|---|---|---|---|---|
| XGBoost | 0,843 | 0,834 | 0,838 | 0,829 |
| Híbrido BiLSTM–XGBoost | 0,842 | 0,845 | 0,838 | 0,830 |
| Híbrido BiTCN–XGBoost | 0,842 | 0,842 | 0,833 | 0,824 |
| BiLSTM | 0,787 | 0,791 | 0,788 | 0,779 |
| BiTCN | 0,775 | 0,766 | 0,778 | 0,756 |

O XGBoost desta tabela é reajustado nas mesmas linhas que as redes usam, as que têm uma sequência completa de 20 blocos, por isso o seu AUC não coincide com o do
modelo adotado, que se ajustou e avalia com outras linhas (`modelo_adotado/numeros_deteccao.json`). O ganho de cada híbrido sobre o XGBoost, com a mesma semente e o intervalo
de confiança por doente, está em `arquiteturas_e_hibridos/resumo_hibrido_quantilico_sementes.csv`. Os `cv3_*.json` repetem a comparação numa validação cruzada de 3
dobras com todos os doentes do desenvolvimento e têm o AUC de cada dobra (`auc_por_dobra`).

## Pastas

| Pasta | O que tem | Origem em `algoritmos/codigo/` |
|---|---|---|
| [`modelo_adotado/`](modelo_adotado/) | Os resultados da deteção com o modelo final: limiares, AUC, robustez, a união dos dois detetores (`grelha_dev.csv` e `grelha_teste.csv` têm a sensibilidade e os falsos alarmes por hora de cada par de limiares), a importância das características, as vias exploradas para melhorar a deteção e se o modelo é pior onde a W25 está truncada. Em `hiperparametros/`, os hiperparâmetros das pesquisas do Optuna, do regressor quantílico (`quant`) e do classificador (`clf`), com o conjunto das 43 características (`tudo`, o do modelo adotado) e com 19 (`base`), e o AUC de CV da melhor tentativa (`best_cv`). | [`3_tendencia_modelo_final/resultados_deteccao/`](../../../algoritmos/codigo/3_tendencia_modelo_final/resultados_deteccao/), [`tendencia_tempo_fixo/`](../../../algoritmos/codigo/3_tendencia_modelo_final/tendencia_tempo_fixo/) |
| [`escolhas_no_desenvolvimento/`](escolhas_no_desenvolvimento/) | As escolhas feitas só no desenvolvimento: a janela de 2, 3, 5 e 10 min, o nível do quantil, o custo da união dos dois detetores, a permanência das variações e a cascata de janelas de 2, 3 e 5 min. As escolhas usam só o desenvolvimento, e as colunas do teste estão lá só para referência. | [`3_tendencia_modelo_final/janela_desenvolvimento/`](../../../algoritmos/codigo/3_tendencia_modelo_final/janela_desenvolvimento/), [`cascata_tempo/`](../../../algoritmos/codigo/3_tendencia_modelo_final/cascata_tempo/) |
| [`caracteristicas_e_janela/`](caracteristicas_e_janela/) | As comparações das características e da janela, em que se treinam duas versões do modelo que só diferem numa coisa (a A sem e a B com), com os mesmos hiperparâmetros e dobras. `comparar_caracteristicas_novas.txt`: as 19 características do conjunto base contra versões com as novas e as 43 finais, com o AUC no desenvolvimento e no teste e o intervalo de confiança da diferença no teste. `comparar_janela_por_tempo.txt`: a janela de 20 blocos contra a de 300 s exatos. | [`3_tendencia_modelo_final/resultados_comparacoes/`](../../../algoritmos/codigo/3_tendencia_modelo_final/resultados_comparacoes/), [`tendencia_tempo_fixo/comparar_caracteristicas_novas.py`](../../../algoritmos/codigo/3_tendencia_modelo_final/tendencia_tempo_fixo/comparar_caracteristicas_novas.py), [`comparacao_janela/`](../../../algoritmos/codigo/3_tendencia_modelo_final/comparacao_janela/) |
| [`arquiteturas_e_hibridos/`](arquiteturas_e_hibridos/) | A tabela acima e a sua origem. As redes sozinhas, os híbridos quantílicos e os híbridos com classificador, o ganho sobre o XGBoost por semente, a validação cruzada de 3 dobras e a sensibilidade ao mesmo custo de falsos alarmes, esta calculada com as previsões do teste. | [`3_tendencia_modelo_final/tendencia_tempo_fixo/hibridos/`](../../../algoritmos/codigo/3_tendencia_modelo_final/tendencia_tempo_fixo/hibridos/) |
| [`valor_do_ecg/`](valor_do_ecg/) | O ECG acrescenta algo ao PPG? A comparação só PPG contra PPG com ECG, só no desenvolvimento para decidir, e com todas as variações do ECG. As colunas do teste estão lá só para referência. | [`3_tendencia_modelo_final/ablacao_ecg_tempo/`](../../../algoritmos/codigo/3_tendencia_modelo_final/ablacao_ecg_tempo/), [`ablacao_ecg_todas/`](../../../algoritmos/codigo/3_tendencia_modelo_final/ablacao_ecg_todas/) |
| [`servidor/`](servidor/) | A extração, a derivação e a inferência do servidor contra o treino (a derivação e a inferência no conjunto de teste), as medianas de reserva da normalização, o ensaio ponta a ponta do worker e o ensaio sem ECG. | [`3_tendencia_modelo_final/servidor_tempo/`](../../../algoritmos/codigo/3_tendencia_modelo_final/servidor_tempo/) |
| [`outras_experiencias/`](outras_experiencias/) | Modelos treinados com um só doente, e os submodelos do alarme e do estável. | [`4_tendencia_selecao_e_experiencias/`](../../../algoritmos/codigo/4_tendencia_selecao_e_experiencias/) |

O teste de carga do servidor, com os três modelos, está em [`../../resultados_sistema/latencia/`](../../resultados_sistema/latencia/).

## Siglas das comparações

Em `caracteristicas_e_janela/comparar_caracteristicas_novas.txt`, cada versão do modelo tem uma sigla, e as linhas `X - base` dão a diferença de AUC face à base, no desenvolvimento e no teste, com o intervalo de confiança a 95 % da diferença no teste.

| Sigla | O que tem |
|---|---|
| `base` | as 19 características do conjunto base |
| `morf` | as 19 e as variações de W50, crest time e decay time a 2 e 5 min (25) |
| `apg` | as 19 e os rácios b/a e d/a da segunda derivada do PPG (25) |
| `novas` | as 19 e W25, W75, a inclinação máxima e o índice de reflexão (31) |
| `tudo` | as 43 finais |

Em `comparar_janela_por_tempo.txt`, `bloco` é a definição de 20 blocos, `tempo` o alvo e as variações por tempo exato (300 s) e `tempo_cheio` o mesmo, só nas linhas que têm todas as características, por isso o seu AUC não se compara com o das outras.


## Desenvolvimento e teste

As escolhas (arquitetura, hiperparâmetros, janela, limiares) fazem-se só com as medidas do desenvolvimento. O conjunto de teste só se usa no fim, para relatar o
resultado. Nos ficheiros, as colunas do teste têm `teste` ou `holdout` no nome (`auc_teste`, `teste_auc`, `mae_holdout`) e as do desenvolvimento costumam ter `dev` ou
`cv` (`auc_dev`, `mae_cv`). Alguns ficheiros trazem as duas lado a lado. Um ficheiro sem estas marcas não diz de que conjunto é, e a pasta e o script que o escreveu
dizem de que dados vêm.
