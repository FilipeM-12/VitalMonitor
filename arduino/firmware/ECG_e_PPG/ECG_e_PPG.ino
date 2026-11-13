// =============================================================================
// ECG_e_PPG.ino
// Pulseira ESP32-C3 — envia dados brutos (ecg, ir, red, temp) por MQTT.
// BPM e SpO2 são calculados no servidor (servidor/workers/comum.py, processamento zero-phase).
//
// Arquitectura de cores:
//   Core 1 (loop) — leitura dos sensores a 100 Hz
//   Core 0        — cliente MQTT: keep-alive, reconexão e publicação
//
// Compatibilidade com os workers do servidor:
//   tópico   pulseiras/raw/<PULSEIRA_ID>   → o worker extrai o ID do 3.º segmento
//   payload  {"pulseira_id","token","amostras":[{t,ecg,ir,red,temp}, ...]}
//   QoS 0    (igual ao subscribe do worker)
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

#define DEBUG       false
#define PULSEIRA_ID "PULSEIRA_01"
#define AUTH_TOKEN  "segredo123"

const char* WIFI_SSID = "A_TUA_REDE_WIFI";
const char* WIFI_PASS = "A_PASSWORD_DA_REDE";

// ── MQTT ──
// O broker tem allow_anonymous false (ver mosquitto.conf), por isso é
// obrigatório autenticar. O utilizador "pulseira" é criado com:
//   sudo mosquitto_passwd /etc/mosquitto/passwd pulseira
const char* MQTT_HOST = "192.168.1.244";  // servidor, Ethernet (eno1)
const int   MQTT_PORT = 1883;
const char* MQTT_USER = "pulseira";
const char* MQTT_PASS = "pulseira_pass";     // <-- pôr a password real

// O worker subscreve pulseiras/raw/# e lê o ID do 3.º segmento do tópico
#define MQTT_TOPIC "pulseiras/raw/" PULSEIRA_ID

// A 100 Hz, 25 amostras = 250 ms por lote
#define BATCH_SIZE  25

// JSON estático: 25 amostras × ~65 chars + 150 overhead ≈ 1800 chars
#define JSON_BUF_SIZE 2048

// ATENÇÃO: o PubSubClient traz MQTT_MAX_PACKET_SIZE = 256 bytes por omissão.
// Com um payload de ~1800 bytes, publish() devolve false e a mensagem é
// descartada SEM ERRO VISÍVEL. É preciso setBufferSize() com margem para o
// payload + tópico + cabeçalho.
#define MQTT_BUF_SIZE (JSON_BUF_SIZE + 128)

// Keep-alive de 30 s (o mosquitto.conf assume 1.5 × 30 = 45 s de tolerância)
#define MQTT_KEEPALIVE_S 30

// ─────────────────────────────────────────────
// AD8232 ECG
// ─────────────────────────────────────────────

const int ecgPin  = 2;
const int loPlus  = 4;
const int loMinus = 5;

// ─────────────────────────────────────────────
// MAX30102 (PPG)
// ─────────────────────────────────────────────

MAX30105 particleSensor;

// Declaração antecipada. O Arduino gera sozinho os protótipos das funções e
// insere-os logo antes da PRIMEIRA função do ficheiro (aqui, mlx_read). Como
// o buildJson() recebe um Amostra*, o protótipo gerado referiria um tipo
// ainda desconhecido e a compilação falharia com "'Amostra' does not name a
// type". Declarar o tipo aqui em cima resolve: um ponteiro para tipo
// incompleto é válido numa declaração, e a definição completa vem mais
// abaixo.
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
// DOUBLE-BUFFER + TASK MQTT (core 0)
// ─────────────────────────────────────────────

struct Amostra {
    uint32_t t;
    int      ecg;
    long     ir;
    long     red;
    float    temp;
};

// Buffer activo (core 1 escreve) e buffer de envio (core 0 lê)
static Amostra collectBuf[BATCH_SIZE];
static Amostra sendBuf[BATCH_SIZE];
static int     collectCount = 0;

// Nunca há alocação dinâmica (evita heap fragmentation no ESP32)
static char    jsonBuf[JSON_BUF_SIZE];

// Flag de sincronização entre cores: enquanto for true, o core 1 não mexe
// no jsonBuf porque o core 0 pode estar a publicá-lo.
volatile bool sendPending = false;

// Contadores de diagnóstico
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
        pos += snprintf(jsonBuf + pos, sizeof(jsonBuf) - pos,
            "{\"t\":%lu,\"ecg\":%d,\"ir\":%ld,\"red\":%ld,\"temp\":%.2f}",
            (unsigned long)buf[i].t,
            buf[i].ecg,
            buf[i].ir,
            buf[i].red,
            buf[i].temp);
    }
    snprintf(jsonBuf + pos, sizeof(jsonBuf) - pos, "]}");
}

// ─── Garante ligação ao broker (não bloqueia mais do que uma tentativa) ───
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

// ─── Task MQTT no core 0 ───
// Acorda de 100 em 100 ms mesmo sem lote, para manter o keep-alive vivo e
// reatar a ligação quando cai.
void mqttSendTask(void* param) {
    for (;;) {
        uint32_t notificado = ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));

        if (!ensureMqtt()) {
            // Sem broker: descartar o lote para não bloquear a amostragem
            if (notificado) { nDescartados++; sendPending = false; }
            continue;
        }

        mqttClient.loop();          // serviço do protocolo (PINGREQ/PINGRESP)

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

    // AD8232
    pinMode(ecgPin,  INPUT);
    pinMode(loPlus,  INPUT);
    pinMode(loMinus, INPUT);

    // I²C + MAX30102
    Wire.begin(6, 7);
    if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
        Serial.println("ERRO: MAX30102 não encontrado");
        while (1);
    }

    // Configuração de baixa latência
    particleSensor.setup(
        31,      // ledBrightness
        1,       // sampleAverage = 1 → sem média, latência mínima
        2,       // ledMode = 2 (IR + RED)
        100,     // sampleRate = 100 Hz
        215,     // pulseWidth µs
        16384    // adcRange
    );
    particleSensor.setPulseAmplitudeIR(0x22);
    particleSensor.setPulseAmplitudeRed(0x3D);

    // WiFi
    connectWiFi();

    // MQTT — o setBufferSize tem de vir ANTES do primeiro connect
    mqttClient.setServer(MQTT_HOST, MQTT_PORT);
    mqttClient.setKeepAlive(MQTT_KEEPALIVE_S);
    if (!mqttClient.setBufferSize(MQTT_BUF_SIZE)) {
        Serial.println("ERRO: sem memória para o buffer MQTT");
    }
    ensureMqtt();

    xTaskCreatePinnedToCore(
        mqttSendTask, "MQTT_SEND",
        8192,            // stack (bytes)
        NULL,            // param
        1,               // prioridade
        &mqttTask_h,
        0                // core 0
    );

    if (DEBUG) {
        // Cabeçalho CSV para monitorização via Serial
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
// LOOP (core 1 — amostragem contínua)
// ─────────────────────────────────────────────

void loop() {

    // Ler temperatura a cada 250 ms (MLX90614 a ~4 Hz)
    if (millis() - lastTempMs > 250) {
        float t = mlx_read();
        if (t > -100.0f) tempObj = t;   // ignora leituras inválidas
        lastTempMs = millis();
    }

    // Reconnect WiFi periódico (cada 10 s)
    if (millis() - lastWifiChk > 10000) {
        if (!WiFi.isConnected()) {
            WiFi.reconnect();
        }
        lastWifiChk = millis();
    }

    // Estatísticas de envio a cada 10 s
    if (DEBUG && millis() - lastStatsMs > 10000) {
        Serial.printf("[MQTT] enviados=%lu falhados=%lu descartados=%lu\n",
                      (unsigned long)nEnviados,
                      (unsigned long)nFalhados,
                      (unsigned long)nDescartados);
        lastStatsMs = millis();
    }

    // Drenar o FIFO do MAX30102 (pode ter várias amostras acumuladas)
    particleSensor.check();

    while (particleSensor.available()) {

        uint32_t t   = millis();
        long     ir  = particleSensor.getFIFOIR();
        long     red = particleSensor.getFIFORed();
        int      ecg = analogRead(ecgPin);
        bool leadOff = (digitalRead(loPlus) == HIGH || digitalRead(loMinus) == HIGH);

        // ─── Saída CSV no Serial (para debug / script Python) ───
        if (DEBUG) {
            Serial.printf("%lu,%d,%ld,%ld,%.2f\n", (unsigned long)t, ecg, ir, red, tempObj);
        }

        // ─── Acumular no lote. O PPG e a temperatura são enviados sempre;
        // sem eléctrodos de ECG em contacto (leadOff) o campo ecg vai a 0,
        // que o servidor trata como janela sem ECG válido (sem crash: ver
        // _detectar_r em comum.py) em vez de descartar o lote inteiro. ───
        if (collectCount < BATCH_SIZE) {
            collectBuf[collectCount].t    = t;
            collectBuf[collectCount].ecg  = leadOff ? 0 : ecg;
            collectBuf[collectCount].ir   = ir;
            collectBuf[collectCount].red  = red;
            collectBuf[collectCount].temp = tempObj;
            collectCount++;
        }

        particleSensor.nextSample();
    }

    // ─── Lote cheio → copiar, serializar e notificar core 0 ───
    // sendPending == false garante que não se sobrescreve um envio em curso.
    if (collectCount >= BATCH_SIZE && !sendPending) {
        memcpy(sendBuf, collectBuf, sizeof(Amostra) * BATCH_SIZE);
        buildJson(sendBuf, BATCH_SIZE);   // escreve em jsonBuf
        collectCount  = 0;
        sendPending   = true;
        xTaskNotifyGive(mqttTask_h);      // acorda task MQTT no core 0
    }
}
