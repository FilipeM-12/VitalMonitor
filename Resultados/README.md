# Resultados · os resultados e as figuras da dissertação

Os resultados da dissertação em três partes, cada uma com um `README.md` que diz o que há em cada ficheiro e como foi obtido.

| Pasta | O que tem | Na dissertação |
|---|---|---|
| [`algoritmos/`](algoritmos/) | Os resultados dos modelos. Na estimação absoluta, o MAE de validação cruzada das cinco arquiteturas e as tabelas da Secção 6.3. Para a tendência, o AUC das arquiteturas, os resultados da deteção e as escolhas no desenvolvimento. | Capítulo 6 |
| [`figuras/`](figuras/) | O código e os dados para refazer 22 gráficos da dissertação, cada um numa subpasta com o número da figura e um script que lê os dados da própria pasta. | Figuras 6.1 a 7.3 e F.1 |
| [`resultados_sistema/`](resultados_sistema/) | Os ensaios do sistema: o teste de carga do servidor (`latencia/`), o sensor de temperatura (`temperatura/`) e as medições nos três voluntários (`sinais_vitais_voluntarios/`). | Capítulo 7 |

As três pessoas dos ensaios com a pulseira aparecem como participante A, B e C. Cada ensaio com a pulseira tem as medições calculadas, as de referência e a gravação
em bruto do dispositivo.

## Origem dos dados e licença

Contém informação da **MIMIC-III Waveform Database Matched Subset** (<https://physionet.org/content/mimic3wdb-matched/1.0/>) e da **MIMIC-IV Waveform Database**
(<https://physionet.org/content/mimic4wdb/0.1.0/>), disponibilizadas pelo PhysioNet sob a Open Database License (ODbL) v1.0
(<https://opendatacommons.org/licenses/odbl/1-0/>). Os ficheiros de `figuras/` e de `algoritmos/` que derivam dessas bases ficam disponíveis sob a mesma licença.
Não há aqui dados das tabelas clínicas do MIMIC, que são de acesso credenciado.
