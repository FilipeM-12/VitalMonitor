# resultados_sistema · os ensaios do sistema

Os dados por trás das três figuras de ensaio do Capítulo 7 da dissertação, cada parte numa subpasta, com um `README.md` que diz como foi medido,
o que tem cada ficheiro e o que significa cada coluna.

| Pasta | O que tem | Na dissertação |
|---|---|---|
| [`latencia/`](latencia/) | A latência e o CPU de um processo do servidor com 10 a 800 pulseiras simuladas, para o XGBoost e para os dois híbridos. | Secção 7.3, Figura 7.1 |
| [`temperatura/`](temperatura/) | As leituras do sensor de temperatura nos dois ensaios, a comparação com o termómetro e os gráficos interativos. | Secção 7.4, Figura 7.2, Tabela 7.1 |
| [`sinais_vitais_voluntarios/`](sinais_vitais_voluntarios/) | As medições do servidor e as de referência (FC, SpO₂ e FR) nos nove ensaios, três em cada uma de três pessoas. | Secção 7.5, Figura 7.3, Tabela 7.2 |

As três pessoas dos ensaios com a pulseira aparecem como participante A, B e C. Cada ensaio com a pulseira tem as medições calculadas, as de referência e a gravação
em bruto do dispositivo.
