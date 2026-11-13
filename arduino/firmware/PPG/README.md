# arduino/firmware/PPG · pulseira só com PPG

Variante de [`ECG_e_PPG`](../ECG_e_PPG/) sem o AD8232 nem os elétrodos, a usada no funcionamento sem ECG: o modelo final de
deteção de tendência usa só PPG. O payload mantém o campo `ecg`, sempre a 0, porque os workers o leem, e por isso a via do
absoluto, que depende do ECG, deixa de produzir valores.

As correntes dos LED são fixas e não há controlo automático de ganho, de propósito: um ganho automático normalizaria o sinal e
destruiria a modulação da respiração de que depende a frequência respiratória. Não as alterar.

Antes de gravar, pôr no topo do `.ino` o nome e a password da rede Wi-Fi (estão os marcadores `A_TUA_REDE_WIFI` e
`A_PASSWORD_DA_REDE`), o IP do broker em `MQTT_HOST` e o `AUTH_TOKEN` do servidor. Bibliotecas: `MAX30105` (SparkFun),
`PubSubClient` e as do núcleo ESP32 (`WiFi`, `Wire`).

| Ficheiro | O que é |
|---|---|
| [`PPG.ino`](PPG.ino) | O firmware. |
