# arduino · firmware da pulseira, simulador e ensaios de bancada

A pulseira é um ESP32-C3 com o MAX30102 (PPG), o AD8232 (ECG) e o MLX90614 (temperatura). O firmware lê os sensores a
100 Hz e publica as amostras em bruto por MQTT, em lotes de 25 amostras (QoS 0); todo o cálculo é feito no servidor.
Antes de gravar um `.ino`, pôr no topo o nome e a password da rede Wi-Fi (estão os marcadores `A_TUA_REDE_WIFI` e
`A_PASSWORD_DA_REDE`) e o IP do broker em `MQTT_HOST`.

| Pasta | O que é |
|---|---|
| [`firmware/`](firmware/) | Os três firmwares, cada um na sua pasta. |
| [`firmware/ECG_e_PPG/`](firmware/ECG_e_PPG/) | **Firmware da pulseira completa**, com ECG, PPG e temperatura. |
| [`firmware/PPG/`](firmware/PPG/) | Variante só com PPG e temperatura, sem o AD8232 nem os elétrodos, usada no funcionamento sem ECG. |
| [`firmware/Ensaio_Bateria_so_PPG/`](firmware/Ensaio_Bateria_so_PPG/) | Variante só com PPG para o ensaio de autonomia da bateria. |
| [`simulador/`](simulador/) | Substitui o ESP32-C3 para testar o servidor: [`simulador_pulseira.py`](simulador/simulador_pulseira.py) publica no broker exatamente o mesmo payload que o firmware, e [`reproduzir_gravacao.py`](simulador/reproduzir_gravacao.py) volta a publicar uma gravação real. |
| [`simulador/testes/`](simulador/testes/) | Ensaios com o simulador: ponta a ponta do worker do absoluto com broker real, e atraso entre a associar a pulseira a um doente e aparecerem os primeiros valores (`ensaio_atraso_associacao.py`, com os resultados de antes e depois da correção da cache, e `limpar_ensaio_atraso.py`, que apaga da base as linhas desse ensaio). |
| `simulador/ensaio_worker_absoluto/` | Registo do ensaio ponta a ponta do worker do absoluto, de 15/10. |
| [`ensaios/`](ensaios/) | Ferramentas dos ensaios de bancada: [`gravar_mqtt.py`](ensaios/gravar_mqtt.py) grava em CSV o sinal em bruto que as pulseiras publicam, [`anotar_ensaio.py`](ensaios/anotar_ensaio.py) é a folha de registo com a hora automática, e [`analisar_ensaio_ppg.py`](ensaios/analisar_ensaio_ppg.py) compara a frequência cardíaca, a SpO₂ e a frequência respiratória do servidor com as medições de referência e gera a figura e os números. |

As gravações dos ensaios com pessoas identificadas não estão no repositório.
