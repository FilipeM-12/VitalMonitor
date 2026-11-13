// =============================================================================
// PPG.ino
// Pulseira ESP32-C3, variante SÓ PPG — envia dados brutos (ir, red, temp) por
// MQTT. Deriva de ECG_e_PPG.ino, sem o AD8232 nem os elétrodos.
//
// PORQUÊ ESTA VARIANTE
//   1. O modelo final de deteção de tendência usa exclusivamente PPG. No sistema
//      completo o ECG não acrescenta AUC (ablacao_ecg_todas.py); só nos blocos que
//      têm as duas cadeias acrescenta cerca de 0,02, e são poucos. Não justifica o
//      atrito de colocar e manter elétrodos torácicos durante horas de espera.
//   2. Não lê o AD8232 nem os pinos de lead-off. (O firmware com ECG também já
//      não descarta amostras quando o elétrodo se solta: envia
//      ecg = 0 e mantém o PPG. A diferença aqui é só não depender do AD8232.)
//   3. Sem AD8232 e sem analogRead() por amostra, o consumo desce.
//
// FREQUÊNCIA RESPIRATÓRIA
//   Este firmware não a calcula — como todo o resto do processamento, isso
//   pertence ao servidor. O que ele garante é a matéria-prima de que a extração
//   depende, e que é frágil:
//     · sampleAverage = 1        sem média, sem suavização a bordo
//     · correntes de LED FIXAS   (setPulseAmplitude*), SEM controlo automático
//                                de ganho
//   A respiração manifesta-se no PPG como modulação lenta da linha de base e da
//   amplitude do pulso. Qualquer ganho automático normaliza o sinal e destrói
//   exatamente essas duas modulações — é a razão pela qual a extração falha
//   sobre o MIMIC, cujo canal PLETH vem em unidades normalizadas. NÃO ATIVAR
//   AGC nem alterar as correntes em execução.
//
// COMPATIBILIDADE COM O SERVIDOR
//   O payload mantém o campo "ecg" com o valor 0, porque o worker acede a
//   s["ecg"] sem valor por omissão (comum.py). Requer, portanto, a variante
//   só-PPG do worker: a cadeia de características que depende da deteção de
//   picos R rejeita as janelas por falta de sinal, que é o comportamento
//   correto, mas a via de PA absoluta baseada em PAT deixa de produzir valores.
//
//   tópico   pulseiras/raw/<PULSEIRA_ID>
//   payload  {"pulseira_id","token","amostras":[{t,ecg,ir,red,temp}, ...]}
//   QoS 0
// =============================================================================

#include <Arduino.h>
#include <Wire.h>
#include "MAX30105.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include "esp_timer.h"

// ─────────────────────────────────────────────
// CONFIGURAÇÕES
// ─────────────────────────────────────────────

#define DEBUG       true
// Mudar se esta pulseira correr ao mesmo tempo que uma com o firmware de ECG,
// senão as duas publicam no mesmo tópico e o worker mistura as amostras.
#define PULSEIRA_ID "PULSEIRA_01"
#define AUTH_TOKEN  "segredo123"

const char* WIFI_SSID = "A_TUA_REDE_WIFI";
const char* WIFI_PASS = "A_PASSWORD_DA_REDE";

// ── MQTT ──
const char* MQTT_HOST = "192.168.1.244";  // servidor, Wi-Fi (wlp3s0), IP por DHCP: confirmar com `ip -4 addr` antes de gravar
const int   MQTT_PORT = 1883;
const char* MQTT_USER = "pulseira";
const char* MQTT_PASS = "pulseira_pass";

#define MQTT_TOPIC "pulseiras/raw/" PULSEIRA_ID

// A 100 Hz, 25 amostras = 250 ms por lote
#define BATCH_SIZE  25

#define JSON_BUF_SIZE 2048
#define MQTT_BUF_SIZE (JSON_BUF_SIZE + 128)
#define MQTT_KEEPALIVE_S 30

// ─────────────────────────────────────────────
// MAX30102 (PPG)
// ─────────────────────────────────────────────

MAX30105 particleSensor;

// Ver a nota em ECG_e_PPG.ino: o protótipo gerado pelo Arduino para
// buildJson() referiria Amostra antes da definição, o que não compila.
struct Amostra;

// ─────────────────────────────────────────────
// MLX90614 (temperatura)
// ─────────────────────────────────────────────

#define MLX_ADDR     0x5A
#define MLX_REG_TOBJ 0x07

float mlx_read() {
    Wire.beginTransmission(MLX_ADDR);
    Wire.write(MLX_REG_TOBJ);
    if (Wire.endTransmission(false) != 0) return -1000.0f;
    delayMicroseconds(200);
    Wire.requestFrom(MLX_ADDR, (uint8_t)3);
    if (Wire.available() < 3) return -1000.0f;
    uint8_t lo = Wire.read(), hi = Wire.read();
    Wire.read(); // PEC (ignorar)
    return ((hi << 8) | lo) * 0.02f - 273.15f;
}

// ─────────────────────────────────────────────
// DOUBLE-BUFFER + TASK MQTT
// ─────────────────────────────────────────────

// Sem campo ecg: o valor 0 é escrito diretamente no JSON, para que a ausência
// de ECG seja explícita no código em vez de ser um zero que se arrasta.
struct Amostra {
    uint32_t t;
    long     ir;
    long     red;
    float    temp;
};

static Amostra collectBuf[BATCH_SIZE];
static Amostra sendBuf[BATCH_SIZE];
static int     collectCount = 0;

static char    jsonBuf[JSON_BUF_SIZE];

volatile bool sendPending = false;
volatile uint32_t nEnviados = 0, nFalhados = 0, nDescartados = 0;

TaskHandle_t mqttTask_h = NULL;

WiFiClient   espClient;
PubSubClient mqttClient(espClient);

// ─── Constrói o payload JSON no buffer estático ───
void buildJson(const Amostra* buf, int n) {
    int pos = snprintf(jsonBuf, sizeof(jsonBuf),
        "{\"pulseira_id\":\"%s\",\"token\":\"%s\",\"amostras\":[",
        PULSEIRA_ID, AUTH_TOKEN);

    for (int i = 0; i < n && pos < (int)sizeof(jsonBuf) - 80; i++) {
        if (i > 0) jsonBuf[pos++] = ',';
        // "ecg":0 — campo mantido por compatibilidade com o worker
        pos += snprintf(jsonBuf + pos, sizeof(jsonBuf) - pos,
            "{\"t\":%lu,\"ecg\":0,\"ir\":%ld,\"red\":%ld,\"temp\":%.2f}",
            (unsigned long)buf[i].t,
            buf[i].ir,
            buf[i].red,
            buf[i].temp);
    }
    snprintf(jsonBuf + pos, sizeof(jsonBuf) - pos, "]}");
}

// ─── Garante ligação ao broker ───
bool ensureMqtt() {
    if (mqttClient.connected()) return true;
    if (!WiFi.isConnected())    return false;

    if (mqttClient.connect(PULSEIRA_ID, MQTT_USER, MQTT_PASS)) {
        if (DEBUG) Serial.println("[MQTT] ligado");
        return true;
    }
    if (DEBUG) {
        Serial.print("[MQTT] falhou, state=");
        Serial.println(mqttClient.state());
    }
    return false;
}

// ─── Task MQTT ───
void mqttSendTask(void* param) {
    for (;;) {
        uint32_t notificado = ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));

        if (!ensureMqtt()) {
            if (notificado) { nDescartados++; sendPending = false; }
            continue;
        }

        mqttClient.loop();

        if (notificado) {
            bool ok = mqttClient.publish(MQTT_TOPIC, jsonBuf);
            if (ok) nEnviados++; else nFalhados++;

            if (DEBUG && !ok) {
                Serial.print("[MQTT] publish falhou (");
                Serial.print(strlen(jsonBuf));
                Serial.println(" bytes) — verificar setBufferSize");
            }
            sendPending = false;
        }
    }
}

// ─────────────────────────────────────────────
// WiFi
// ─────────────────────────────────────────────

void connectWiFi() {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.print("WiFi");
    unsigned long t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 10000) {
        delay(200);
        Serial.print(".");
    }
    if (WiFi.isConnected())
        Serial.println(" OK → " + WiFi.localIP().toString());
    else
        Serial.println(" FALHOU (a continuar sem rede)");
}

// ─────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────

void setup() {
    Serial.begin(115200);

    // I²C + MAX30102
    Wire.begin(6, 7);
    if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
        Serial.println("ERRO: MAX30102 não encontrado");
        while (1);
    }

    // Mesmos parâmetros do firmware com ECG, para que o sinal seja comparável
    // com as aquisições de bancada já realizadas. Ver nota sobre AGC no topo:
    // sampleAverage = 1 e correntes fixas são o que preserva as modulações
    // respiratórias da linha de base e da amplitude.
    particleSensor.setup(
        31,      // ledBrightness
        1,       // sampleAverage = 1 → sem média
        2,       // ledMode = 2 (IR + RED)
        100,     // sampleRate = 100 Hz
        215,     // pulseWidth µs
        16384    // adcRange
    );
    particleSensor.setPulseAmplitudeIR(0x22);
    particleSensor.setPulseAmplitudeRed(0x3D);

    connectWiFi();

    mqttClient.setServer(MQTT_HOST, MQTT_PORT);
    mqttClient.setKeepAlive(MQTT_KEEPALIVE_S);
    if (!mqttClient.setBufferSize(MQTT_BUF_SIZE)) {
        Serial.println("ERRO: sem memória para o buffer MQTT");
    }
    ensureMqtt();

    xTaskCreatePinnedToCore(
        mqttSendTask, "MQTT_SEND",
        8192, NULL, 1, &mqttTask_h, 0
    );

    if (DEBUG) {
        // Cabeçalho CSV. A coluna ecg fica a 0 para que os scripts de análise
        // já existentes continuem a ler o ficheiro sem alterações.
        Serial.println("time_ms,ecg,ir,red,temp");
    }
}

// ─────────────────────────────────────────────
// VARIÁVEIS AUXILIARES
// ─────────────────────────────────────────────

static unsigned long lastTempMs  = 0;
static float         tempObj     = 0.0f;
static unsigned long lastWifiChk = 0;
static unsigned long lastStatsMs = 0;

// ─────────────────────────────────────────────
// LOOP (amostragem contínua)
// ─────────────────────────────────────────────

void loop() {

    // Temperatura a cada 250 ms (MLX90614 a ~4 Hz)
    if (millis() - lastTempMs > 250) {
        float t = mlx_read();
        if (t > -100.0f) tempObj = t;
        lastTempMs = millis();
    }

    // Reconnect WiFi periódico (cada 10 s)
    if (millis() - lastWifiChk > 10000) {
        if (!WiFi.isConnected()) {
            WiFi.reconnect();
        }
        lastWifiChk = millis();
    }

    if (DEBUG && millis() - lastStatsMs > 10000) {
        Serial.printf("[MQTT] enviados=%lu falhados=%lu descartados=%lu\n",
                      (unsigned long)nEnviados,
                      (unsigned long)nFalhados,
                      (unsigned long)nDescartados);
        lastStatsMs = millis();
    }

    // Drenar o FIFO do MAX30102
    particleSensor.check();

    while (particleSensor.available()) {

        uint32_t t   = millis();
        long     ir  = particleSensor.getFIFOIR();
        long     red = particleSensor.getFIFORed();

        if (DEBUG) {
            Serial.printf("%lu,0,%ld,%ld,%.2f\n", (unsigned long)t, ir, red, tempObj);
        }

        // Sem condição de lead-off: toda a amostra de PPG é guardada. É esta a
        // diferença que garante uma série sem buracos, de que a extração da
        // frequência respiratória depende.
        if (collectCount < BATCH_SIZE) {
            collectBuf[collectCount].t    = t;
            collectBuf[collectCount].ir   = ir;
            collectBuf[collectCount].red  = red;
            collectBuf[collectCount].temp = tempObj;
            collectCount++;
        }

        particleSensor.nextSample();
    }

    // ─── Lote cheio → copiar, serializar e notificar a task MQTT ───
    if (collectCount >= BATCH_SIZE && !sendPending) {
        memcpy(sendBuf, collectBuf, sizeof(Amostra) * BATCH_SIZE);
        buildJson(sendBuf, BATCH_SIZE);
        collectCount  = 0;
        sendPending   = true;
        xTaskNotifyGive(mqttTask_h);
    }
}
