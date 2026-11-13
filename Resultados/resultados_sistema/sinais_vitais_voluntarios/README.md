# sinais_vitais_voluntarios · FC, SpO₂ e FR nos nove ensaios com a pulseira, em três voluntários

É a Secção 7.5 da tese, com a Figura 7.3 e a Tabela 7.2.

## Como foi medido

A frequência cardíaca (FC), a SpO₂ e a frequência respiratória (FR) que o servidor calcula compararam-se com medições de referência em nove ensaios de
7 a 15 minutos, três em cada uma de três pessoas em repouso. A referência da FC e da SpO₂ foi a leitura de uma pulseira de atividade (Xiaomi Mi
Band 7 Pro), e a da FR a contagem manual das respirações durante um minuto. Em cada ensaio tiraram-se algumas leituras ao longo do tempo, anotadas
com a hora. Cada leitura compara-se com a mediana dos valores do servidor nos 10 s de cada lado dela; na FR, o minuto contado compara-se com a
mediana do servidor nos 5 s de cada lado do fim desse minuto. O sinal foi gravado à saída do broker e, em dois dos ensaios mais antigos (o do
participante B de 12/10 e o do participante C de 14/10), repetido no servidor com a versão atual do código. Nestes dois, `gravacao.csv.gz` é a gravação original, e os
valores do servidor de `pares.csv` e de `resultados.json` são os do replay, com a hora do replay.

As pessoas aparecem como **participante A, B e C**. Cada ensaio tem também a gravação em bruto do dispositivo, sem o nome de ninguém.

## Ficheiros

| Ficheiro | O que tem |
|---|---|
| [`resumo_nove_ensaios.csv`](resumo_nove_ensaios.csv) | A Tabela 7.2 e o seu detalhe: por grandeza, o erro do servidor no conjunto (`grupo` = `todos`), por pessoa e por ensaio. |
| `participante_A/`, `participante_B/`, `participante_C/` | Uma pasta por pessoa, com uma subpasta por ensaio (a data). |
| `participante_A/2026-10-30/serie_do_servidor.csv` | A série de tempo que a Figura 7.3 desenha: os valores de FC, SpO₂ e FR do servidor ao longo deste ensaio. A SpO₂ não tem os primeiros e os últimos 30 s, tal como na figura. |

Cada subpasta de ensaio tem:

| Ficheiro | O que tem |
|---|---|
| `pares.csv` | As leituras de referência emparelhadas com os valores do servidor. |
| `registo_referencia.csv` | As leituras de referência tal como foram anotadas durante o ensaio. |
| `resultados.json` | A duração, o número de amostras, a taxa de amostragem e o viés, o erro absoluto médio, o desvio padrão e o erro máximo de cada grandeza. |
| `resumo.md` | O mesmo resumo em tabela. |
| `gravacao.csv.gz` | A gravação em bruto do dispositivo, comprimida, tal como o servidor a recebeu (a original, nos dois ensaios repetidos). |

Colunas de `gravacao.csv.gz`: `rx_utc` (hora UTC a que o servidor recebeu cada lote de 25 amostras), `time_ms` (relógio da pulseira em milissegundos), `ecg`, `ir` e `red`
(contagens do ADC, a cerca de 100 amostras por segundo) e `temp` (temperatura da pele, em °C).

Para repetir um ensaio no servidor, descomprime-se a gravação (`gunzip -k gravacao.csv.gz`) e reproduz-se com
[`arduino/simulador/reproduzir_gravacao.py`](../../../arduino/simulador/reproduzir_gravacao.py) (`python reproduzir_gravacao.py gravacao.csv --pulseira PULSEIRA_REPLAY_X`), com o
servidor ligado e a pulseira atribuída a um doente na base de dados. Os valores ficam na tabela `medicao`, com a hora do replay.

Colunas de `pares.csv`: `hora` (hora local da leitura), `respiracoes` (respirações contadas no minuto que **começa** nessa hora), `fc_smartband`,
`spo2_smartband`, `notas`, `hora_utc`, `fc_servidor`, `spo2_servidor` (mediana do servidor nos 10 s de cada lado da leitura) e `fr_servidor` (mediana nos
5 s de cada lado do fim do minuto contado). Uma linha de referência só tem preenchida a grandeza que se leu nessa hora.

Colunas de `resumo_nove_ensaios.csv`: `grandeza` (FC, SpO2 ou FR), `grupo`, `n` (pares), `vies` (média do erro, servidor menos referência), `mae` (erro
absoluto médio), `desvio_padrao_do_erro`, `erro_maximo`, `referencia_min` e `referencia_max` (a gama dos valores de referência) e
`fracao_dentro_do_limite` (a fração de comparações a menos de 2 bpm ou 2 respirações por minuto da referência, ou a menos de 3 pontos, na SpO₂).

## Resultado

| | Pares | Gama da referência | Viés | MAE | Desvio padrão | Erro máximo |
|---|---|---|---|---|---|---|
| FC (bpm) | 36 | 52 a 81 | +0,2 | 2,3 | 3,2 | 10,5 |
| SpO₂ (%) | 33 | 95 a 99 | +1,3 | 1,6 | 1,7 | 4 |
| FR (respirações/min) | 23 | 9 a 19 | −0,4 | 0,8 | 1,1 | 3 |

Cerca de 79 % das janelas passaram os critérios de qualidade do sinal, entre 37 % e 100 % consoante o ensaio. São três pessoas em repouso e poucas
leituras por ensaio, por isso o resultado não permite conclusões sobre a exatidão em esforço, com FC alta ou SpO₂ baixa, e não tem valor clínico por si
só: mostra que as três grandezas seguem a referência em condições normais.

O código que redesenha a Figura 7.3, com a série do servidor e as referências do ensaio anterior, está em [`../../figuras/7.3_sinais_vitais/`](../../figuras/7.3_sinais_vitais/).
