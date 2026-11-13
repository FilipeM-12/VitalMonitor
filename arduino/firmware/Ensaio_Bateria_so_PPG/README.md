# arduino/firmware/Ensaio_Bateria_so_PPG · ensaio de autonomia

Variante só com PPG e temperatura para medir a autonomia da bateria. **Não gravar numa pulseira de produção**: o payload já não
tem o campo `ecg` e inclui `v_frio`, `v_estab` e `v_bat_est`, leituras do divisor de tensão da bateria feitas a cada 5 s, que os
workers não esperam. Para voltar ao normal, gravar [`ECG_e_PPG.ino`](../ECG_e_PPG/ECG_e_PPG.ino).

Antes de gravar, pôr no topo do `.ino` o nome e a password da rede Wi-Fi (estão os marcadores `A_TUA_REDE_WIFI` e
`A_PASSWORD_DA_REDE`), o IP do broker em `MQTT_HOST` e o `AUTH_TOKEN` do servidor. Bibliotecas: `MAX30105` (SparkFun),
`PubSubClient` e as do núcleo ESP32 (`WiFi`, `Wire`).

| Ficheiro | O que é |
|---|---|
| [`Ensaio_Bateria_so_PPG.ino`](Ensaio_Bateria_so_PPG.ino) | O firmware. |
