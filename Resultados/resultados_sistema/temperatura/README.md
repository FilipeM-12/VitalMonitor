# temperatura · o sensor MLX90614 contra um termómetro

É a Secção 7.4 da tese, com a Figura 7.2 e a Tabela 7.1.

## Como foi medido

O sensor MLX90614 ficou debaixo de uma botija de água quente e a temperatura foi baixando, o que dá uma descida lenta e contínua. O termómetro foi lido
no início e no fim de cada ensaio e esses valores compararam-se com a leitura do sensor nos mesmos instantes. Foram dois ensaios, de 15,7 e 11,5
minutos, com uma leitura do sensor a cada 0,5 s, ambos de 8 de abril de 2026. O termómetro dá os valores com uma casa decimal. Só há quatro
comparações, e o sensor mediu uma botija e não um dedo, por isso isto não substitui um ensaio em pessoas.

## Ficheiros

| Ficheiro | O que tem |
|---|---|
| [`ensaio_1_sensor.csv`](ensaio_1_sensor.csv), [`ensaio_2_sensor.csv`](ensaio_2_sensor.csv) | As leituras do sensor de cada ensaio. `time_ms` é o relógio da pulseira em milissegundos (`millis()`) e `Temp` é a temperatura do objeto, em °C. |
| [`comparacao_termometro_sensor.csv`](comparacao_termometro_sensor.csv) | A Tabela 7.1: por ensaio, a leitura do termómetro e a do sensor no início e no fim, e o erro (sensor menos termómetro). |
| [`ensaio_1_grafico.html`](ensaio_1_grafico.html), [`ensaio_2_grafico.html`](ensaio_2_grafico.html) | O gráfico interativo de cada ensaio (abrem no navegador). |

Colunas de `comparacao_termometro_sensor.csv`: `ensaio` (1 ou 2), `momento` (`inicio` ou `fim`), `tempo_s` (segundos desde o início do ensaio),
`termometro_C`, `sensor_C` e `erro_C`.

## Resultado

O sensor ficou a menos de 0,3 °C do termómetro em todas as leituras, com um erro médio de +0,10 °C e um erro absoluto médio de 0,15 °C. O segundo
ensaio cobre de 36,4 a 38,3 °C, a gama da temperatura corporal.

O código que redesenha a Figura 7.2, com uma cópia destes dados, está em [`../../figuras/7.2_temperatura/`](../../figuras/7.2_temperatura/).
