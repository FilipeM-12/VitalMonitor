# servidor · receção do sinal, painel e modelos em tempo real

O servidor corre numa máquina da rede local: o Mosquitto recebe os lotes das pulseiras por MQTT, quatro instâncias do
worker processam o sinal (cada pulseira cai sempre na mesma instância) e o Flask serve o painel por HTTPS, atrás do
Caddy. Os resultados ficam no PostgreSQL.

| Pasta | O que tem |
|---|---|
| [`painel/`](painel/) | O servidor Flask e tudo o que ele serve: [`server.py`](painel/server.py), as páginas ([`templates/`](painel/templates/)) e o leitor de Cartão de Cidadão ([`ler_cartao.cpp`](painel/ler_cartao.cpp) e [`makefile`](painel/makefile)). Corre-se a partir desta pasta. |
| [`workers/`](workers/) | Os workers que recebem o sinal e aplicam os modelos, com os módulos que usam. Estão todos juntos porque se importam uns aos outros pelo nome. |
| [`modelos/`](modelos/) | Os modelos treinados que o painel e os workers carregam. |
| [`instalacao/`](instalacao/) | Como pôr o servidor a correr numa máquina nova, a unidade systemd dos workers e os pacotes Python. |

## painel/

| Ficheiro | O que faz |
|---|---|
| [`server.py`](painel/server.py) | Servidor Flask: painel de monitorização (`templates/painel.html`), associação das pulseiras aos doentes, configuração dos alertas e histórico. |
| [`ler_cartao.cpp`](painel/ler_cartao.cpp) e [`makefile`](painel/makefile) | Leitura do Cartão de Cidadão para a associação. O middleware Autenticação.gov instala-se à parte. |

## workers/

| Ficheiro | O que faz |
|---|---|
| [`worker_tendencia_quantilica.py`](workers/worker_tendencia_quantilica.py) | **O worker instalado.** Deteção de tendência por regressão quantílica sobre a variação da pressão em 5 min, só com o PPG (43 características). Corre em quatro instâncias (`pulseiras-worker@0` a `@3`). |
| [`comum.py`](workers/comum.py) | Núcleo partilhado pelos workers: receção dos lotes MQTT, buffers por pulseira, via rápida (frequência cardíaca, SpO₂, frequência respiratória e temperatura, cerca de uma vez por segundo), via lenta por blocos de 15 s, filtros e deteção de picos, escrita na base de dados. |
| [`extracao_ppg.py`](workers/extracao_ppg.py) e [`derivacao_ppg.py`](workers/derivacao_ppg.py) | Extração das características de uma janela e derivação por bloco com a janela de 5 min ancorada no tempo e as 43 características. São as versões do worker instalado, verificadas contra o treino no conjunto de teste inteiro. |
| [`frequencia_respiratoria.py`](workers/frequencia_respiratoria.py) | Estimação da frequência respiratória a partir do PPG, com a exclusão da zona cardíaca. |
| [`extracao_ppg_classificador.py`](workers/extracao_ppg_classificador.py) e [`derivacao_ppg_classificador.py`](workers/derivacao_ppg_classificador.py) | A extração e a derivação com 20 características e janela por blocos, anteriores às do worker instalado. Ficam porque as ferramentas de análise e as verificações da extração as usam. |
| [`worker_absoluto.py`](workers/worker_absoluto.py) | Variante que estima o valor absoluto da pressão sistólica com ECG e PPG. |
## modelos/

| Ficheiro | O que é |
|---|---|
| `modelo_tendencia_ppg_quant.joblib` | O modelo instalado: regressor quantílico da descida e da subida, exportado pelo notebook 03. Lido pelo worker instalado e pelo painel. |
| `modelo_sbp.joblib` | O modelo absoluto, exportado pelo notebook 01. |

## instalacao/

| Ficheiro | O que é |
|---|---|
| [`INSTALACAO.md`](instalacao/INSTALACAO.md) | O guia passo a passo: PostgreSQL, Mosquitto, Python, serviços systemd, rede, variáveis de ambiente, HTTPS com o Caddy e o leitor de Cartão de Cidadão. |
| [`mosquitto_hospital.conf`](instalacao/mosquitto_hospital.conf) | Configuração do Mosquitto: porta 1883, sem acesso anónimo, até 500 clientes. |
| [`pulseiras.service`](instalacao/pulseiras.service) | Unidade systemd do painel (Gunicorn, só em `127.0.0.1:5000`). |
| [`pulseiras-worker@.service`](instalacao/pulseiras-worker@.service) | Unidade systemd dos workers, uma instância por número. |
| [`Caddyfile`](instalacao/Caddyfile) | HTTPS, a única entrada do painel: redireciona a porta 80 e passa a 443 ao painel. |
| [`requirements.txt`](instalacao/requirements.txt) | Pacotes Python do servidor. |

As credenciais no código (base de dados, MQTT e o token das pulseiras) são as de desenvolvimento e devem ser mudadas
numa instalação real.
