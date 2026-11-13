# VitalMonitor · código da dissertação

Código do VitalMonitor, o sistema da dissertação de mestrado em Engenharia Eletrotécnica e de Computadores de
Filipe Mendes (Instituto Superior Técnico, outubro de 2026; orientação de Prof.ª Teresa Vazão e Dr. João Gouveia).
Um dispositivo vestível (ESP32-C3 com PPG, ECG e temperatura) envia o sinal em bruto para um servidor, que estima os
sinais vitais em tempo real e deteta a tendência da pressão arterial sem braçadeira, com modelos treinados no
MIMIC-III e MIMIC-IV.

## Organização

O repositório segue as três partes do sistema, e a pasta `Resultados/` guarda os dados da tese e o código para refazer os gráficos. Cada pasta tem o seu `README.md` a identificar o que lá está, ou é descrita no da pasta de cima.

| Pasta | O que contém | Na dissertação |
|---|---|---|
| [`arduino/`](arduino/) | Firmware da pulseira (ESP32-C3, MAX30102, AD8232, MLX90614), firmware do ensaio de autonomia, simulador de pulseiras e as ferramentas de gravação, anotação e análise dos ensaios de bancada | Cap. 4 e 5 (Arquitetura e Implementação), ensaios do Cap. 7 |
| [`servidor/`](servidor/) | Servidor Flask com o painel de monitorização, workers que recebem o sinal por MQTT e aplicam os modelos, extração e derivação das características em tempo real, frequência respiratória, modelos instalados e serviço systemd | Cap. 4, 5 e 7 |
| [`algoritmos/`](algoritmos/) | Notebooks de ponta a ponta (do sinal do MIMIC ao modelo que o servidor carrega) e o código que produz os números e as tabelas da dissertação | Cap. 6 (Algoritmos) e apêndices |
| [`Resultados/`](Resultados/) | Os resultados da dissertação, em três pastas: `algoritmos/` (os resultados dos modelos, com o MAE e o AUC de validação cruzada das arquiteturas e as tabelas do Capítulo 6), `figuras/` (o código e os dados para refazer 22 gráficos) e `resultados_sistema/` (os ensaios do Capítulo 7: o teste de carga, o sensor de temperatura e as medições nos três voluntários) | Capítulos 6 e 7 e Figuras 6.1 a 7.3 e F.1 |

Para ler o código pela primeira vez, a ordem natural é o firmware (`arduino/firmware/ECG_e_PPG/`), o núcleo do
servidor (`servidor/workers/comum.py` e `servidor/workers/worker_tendencia_quantilica.py`) e os notebooks
(`algoritmos/notebooks/`).

## O que não está incluído

Os dados do MIMIC-III e do MIMIC-IV, que se descarregam da PhysioNet (os scripts regeneram tudo a partir deles), e o software de
terceiros do leitor de Cartão de Cidadão (Autenticação.gov), que se instala à parte.

| Dados | Versão | Acesso | Licença |
|---|---|---|---|
| [MIMIC-III Waveform Database, Matched Subset](https://physionet.org/content/mimic3wdb-matched/1.0/) (os sinais) | 1.0 | Aberto | Open Data Commons Open Database License v1.0 |
| [MIMIC-IV Waveform Database](https://physionet.org/content/mimic4wdb/0.1.0/) (os sinais) | 0.1.0 | Aberto | Open Data Commons Open Database License v1.0 |
| [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/) (idade, sexo, admissões) | 1.4 | Só com conta credenciada | PhysioNet Credentialed Health Data License 1.5.0 |
| [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) (idade, sexo) | 3.1 | Só com conta credenciada | PhysioNet Credentialed Health Data License 1.5.0 |

Os sinais são de acesso aberto, para quem cumpra a licença. As tabelas clínicas exigem uma conta credenciada na PhysioNet (curso CITI e acordo de
utilização), e essa licença não permite partilhar o acesso aos dados com outras pessoas; por isso as tabelas por doente que saem delas (idade e sexo) não
estão neste repositório.

Contém informação da MIMIC-III Waveform Database Matched Subset e da MIMIC-IV Waveform Database, disponibilizadas pelo PhysioNet sob a Open Database License
(ODbL) v1.0 (<https://opendatacommons.org/licenses/odbl/1-0/>). Os ficheiros de `Resultados/figuras/` que derivam dessas bases ficam disponíveis sob a mesma licença.

## Como correr

* Python 3.12. Os pacotes do treino estão em `algoritmos/versoes_pacotes_absoluto.txt` (notebook 01) e
  `algoritmos/versoes_pacotes_tendencia.txt` (notebooks 02 e 03), e os do servidor em `servidor/instalacao/requirements.txt`. A instalação do servidor de raiz (PostgreSQL, Mosquitto, serviços, HTTPS) está em
  `servidor/instalacao/INSTALACAO.md`.
* Os scripts usam os caminhos da máquina de desenvolvimento (`/home/filipe/Desktop/Tese/...`), definidos no topo de cada
  ficheiro. Para correr noutra máquina, basta pôr estas pastas na mesma estrutura ou mudar essa raiz.
  Esses caminhos usam os nomes das pastas do desenvolvimento: `Tentativa_22_10/` é hoje `algoritmos/codigo/3_tendencia_modelo_final/`
  e `RESULTADOS_RELATORIO/Absoluto_27/` é `algoritmos/codigo/2_estimacao_absoluta/modelo_e_experiencias/`. As pastas de datas
  (`2026-09-...`), que guardam a cadeia de desenvolvimento anterior, não estão neste repositório.
  Os estudos do Optuna e os resultados da comparação de arquiteturas (`optuna_arquiteturas/`, `resumo_arquiteturas.csv` e
  `holdout_preds_arquiteturas_*`) chamam-se no desenvolvimento `optuna_retune50causal/`, `resumo_retune50causal.csv` e
  `holdout_preds_retune50causal_*`. Esse «50» é um resto do desenvolvimento: as pesquisas do Optuna usadas na dissertação têm 20 tentativas cada.
* Os notebooks têm interruptores na primeira célula de código. Por omissão não repetem a extração completa do MIMIC nem
  a pesquisa de hiperparâmetros, que demoram várias horas.

## Credenciais

As credenciais que aparecem no código (base de dados, MQTT e o token das pulseiras) são as de desenvolvimento e devem ser
mudadas numa instalação real. No firmware, o nome e a password da rede Wi-Fi foram substituídos por marcadores
(`A_TUA_REDE_WIFI`, `A_PASSWORD_DA_REDE`).

## Como este repositório foi montado

É o conteúdo da pasta de entrega da dissertação, montada por scripts a partir do código de desenvolvimento: só os
ficheiros que produzem resultados da dissertação ou que correm no sistema, com a estrutura de pastas original, as
credenciais do Wi-Fi substituídas e as participantes dos ensaios de bancada anonimizadas. Os `README.md` de cada pasta
foram escritos para este repositório.
