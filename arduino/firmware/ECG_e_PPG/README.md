# arduino/firmware/ECG_e_PPG · pulseira completa

Firmware da pulseira completa: ESP32-C3 com o MAX30102 (PPG), o AD8232 (ECG) e o MLX90614 (temperatura). Um núcleo lê os
sensores a 100 Hz e o outro mantém a ligação MQTT e publica. Não calcula nada: a frequência cardíaca, a SpO₂, a frequência
respiratória e os modelos são do servidor.

* Tópico `pulseiras/raw/<PULSEIRA_ID>`, QoS 0, um lote de 25 amostras (250 ms) por mensagem.
* Payload: `{"pulseira_id","token","amostras":[{t,ecg,ir,red,temp}, ...]}`.

Antes de gravar, pôr no topo do `.ino` o nome e a password da rede Wi-Fi (estão os marcadores `A_TUA_REDE_WIFI` e
`A_PASSWORD_DA_REDE`), o IP do broker em `MQTT_HOST` e o `AUTH_TOKEN` do servidor. Bibliotecas: `MAX30105` (SparkFun),
`PubSubClient` e as do núcleo ESP32 (`WiFi`, `Wire`).

| Ficheiro | O que é |
|---|---|
| [`ECG_e_PPG.ino`](ECG_e_PPG.ino) | O firmware. |
