# latencia · teste de carga de um processo do servidor

Quantas pulseiras um só processo do servidor aguenta sem fila, para o XGBoost adotado e para os dois híbridos comparados na Secção 6.4. É a
Secção 7.3 da tese e a Figura 7.1.

## Como foi medido

As pulseiras são simuladas, com sinal real de bancada repetido em ciclo, e o teste corre só em CPU. Cada pulseira gera um trabalho por janela de
5 s (extração das características, derivação causal e, de 15 em 15 s, a inferência). Cada nível de carga (número de pulseiras por processo) corre
45 s. A **latência** de um trabalho é o instante em que acaba menos o instante em que devia ter começado. Há **fila** quando o percentil 95 da
latência passa 1 s. As redes dos híbridos têm pesos aleatórios, por isso o teste vale para tempo e memória e não para exatidão.

## Ficheiros

| Ficheiro | O que tem |
|---|---|
| [`latencia_e_cpu_por_nivel.csv`](latencia_e_cpu_por_nivel.csv) | Uma linha por modelo, corrida e nível de carga. É o que a Figura 7.1 desenha. |
| [`resumo_por_modelo.csv`](resumo_por_modelo.csv) | A capacidade de cada modelo (o maior nível sem fila em todas as repetições), a memória que o modelo acrescenta ao processo e os tempos de extração e de inferência. |
| [`json_originais/`](json_originais/) | Os 12 ficheiros que o teste escreveu (um por modelo e corrida), com todos os campos. O nome diz o modelo: `carga_xgb*` é o XGBoost adotado, e `carga_bilstm*` e `carga_bitcn*` são os dois híbridos, BiLSTM + XGBoost e BiTCN + XGBoost (uma rede que dá uma representação ao XGBoost). Sem sufixo é a corrida `grossa`, e `_rep1` a `_rep3` são as três repetições. |

Colunas de `latencia_e_cpu_por_nivel.csv`:

| Coluna | Significado |
|---|---|
| `modelo` | XGBoost, BiLSTM + XGBoost ou BiTCN + XGBoost. |
| `corrida` | `grossa` é a primeira corrida, com níveis de 10 a 800 pulseiras; `repeticao_1`, `_2` e `_3` são as três repetições independentes dos níveis de 300 a 500, perto do limite. |
| `pulseiras` | Pulseiras simuladas num só processo. |
| `trabalhos`, `trabalhos_por_s` | Trabalhos feitos no nível e ritmo a que chegam (cada pulseira gera um a cada 5 s). |
| `latencia_p50_ms`, `latencia_p95_ms`, `latencia_p99_ms`, `latencia_max_ms` | Percentis e máximo da latência dos trabalhos, em milissegundos. |
| `espera_fila_p95_ms` | Percentil 95 do tempo que o trabalho esperou na fila: o instante em que começou menos o instante em que devia ter começado. |
| `cpu_nucleos` | CPU média do processo no nível, em núcleos. |
| `memoria_rss_mb` | Memória residente do processo no fim do nível, em MB. |
| `estavel` | Verdadeiro se o p95 da latência ficou abaixo de 5 s (o intervalo entre janelas de uma pulseira) e a fila esvaziou no fim. |
| `usado_na_figura` | Falso nos níveis de 400 e 800 pulseiras da corrida `grossa`, que a Figura 7.1 substitui pelas três repetições. Nos outros níveis, a figura usa a mediana das repetições (300 a 500) ou o valor da corrida `grossa` (10 a 200). |

## Resultado

O XGBoost aguentou 400 pulseiras por processo sem fila e os dois híbridos 350, em três repetições independentes. Quase todo o tempo vai na extração
das características, comum aos três modelos, com cerca de 11 ms por janela, contra 0,6 ms da inferência do XGBoost e 2 a 4 ms da dos híbridos.
O que distingue os híbridos é a memória, cerca de 500 MB por processo contra 17 MB do XGBoost.

As corridas que coincidiram com outra tarefa na máquina ficaram de fora.

O código que redesenha a Figura 7.1, com uma cópia destes JSON, está em [`../../figuras/7.1_teste_carga/`](../../figuras/7.1_teste_carga/).
