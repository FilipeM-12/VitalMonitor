// =============================================================================
// Ensaio_Bateria_so_PPG.ino
// Variante do ECG_e_PPG.ino SÓ PARA O ENSAIO DE AUTONOMIA,
// SEM ECG — só PPG (MAX30102) e temperatura (MLX90614).
//
// PORQUÊ ESTA VERSÃO
//   O modelo final de deteção de tendência usa exclusivamente PPG — o ECG
//   deixou de fazer parte da cadeia (ver Capítulo 7, "simplificação do
//   dispositivo": uma versão sem AD8232 e sem eléctrodos reduz consumo).
//   Manter o AD8232 ligado e a ser lido no ensaio mediria a autonomia de um
//   hardware que já não corresponde ao que o sistema final usa. Esta versão
//   remove o AD8232 por completo: não configura os pinos, não faz
//   analogRead(), não lê loPlus/loMinus, e o payload MQTT deixa de ter o
//   campo "ecg".
//
// O QUE FICA IGUAL
//   MAX30102 (IR, RED) e MLX90614 (temperatura), lidos e enviados por MQTT
//   exactamente como nas outras variantes, ao mesmo ritmo (lotes de 25
//   amostras, ~250 ms). O script do ensaio de bateria não precisa de nenhuma alteração
//   — só conta o número de amostras em cada lote, não olha a campos
//   individuais.
//
// ATENÇÃO: o payload já não tem "ecg". Se este ficheiro for usado com o
// worker de produção (que espera esse campo), vai dar erro — isto é só
// para o ensaio de bateria, não para produção.
//
// NÃO GRAVAR EM PRODUÇÃO. Para voltar ao firmware normal depois do ensaio,
// grava de novo o ECG_e_PPG.ino.
//
// DIAGNÓSTICO EXTRA: o payload inclui "v_frio", "v_estab" e "v_bat_est",
// leituras do divisor de tensão da bateria no pino D0/A0, lidas a cada 5 s
// (não por amostra, para não atrasar o PPG). Servem para ver se a leitura do
// ADC muda quando o cabo USB é desligado: com o USB ligado, o ADC lia abaixo
// do que o multímetro media no mesmo pino. Os valores não são fiáveis em
// absoluto; o que interessa é só ver se mudam ao tirar o USB, não o valor em si.
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
#define PULSEIRA_ID "PULSEIRA_01"
#define AUTH_TOKEN  "segredo123"

const char* WIFI_SSID = "A_TUA_REDE_WIFI";
const char* WIFI_PASS = "A_PASSWORD_DA_REDE";

// ── MQTT ──
const char* MQTT_HOST = "192.168.1.239";  // servidor, Ethernet (eno1)
const int   MQTT_PORT = 1883;
const char* MQTT_USER = "pulseira";
const char* MQTT_PASS = "pulseira_pass";     // <-- pôr a password real

#define MQTT_TOPIC "pulseiras/raw/" PULSEIRA_ID

// A 100 Hz, 25 amostras = 250 ms por lote
#define BATCH_SIZE  25

// Payload sem o campo "ecg", mais os três campos de diagnóstico da tensão:
// 25 amostras × ~45 chars + overhead + v_frio/v_estab/v_bat_est
#define JSON_BUF_SIZE 1650
#define MQTT_BUF_SIZE (JSON_BUF_SIZE + 128)
#define MQTT_KEEPALIVE_S 30

// Declaração antecipada. O Arduino gera sozinho os protótipos das funções e
// insere-os logo antes da PRIMEIRA função do ficheiro (aqui, mlx_read). Como
// o buildJson() recebe um Amostra*, o protótipo gerado referiria um tipo
// ainda desconhecido e a compilação falharia com "'Amostra' does not name a
// type". Declarar o tipo aqui em cima resolve: um ponteiro para tipo
// incompleto é válido numa declaração, e a definição completa vem mais
// abaixo.
struct Amostra;

// ─────────────────────────────────────────────
// MAX30102 (PPG)
// ─────────────────────────────────────────────

MAX30105 particleSensor;

// ─────────────────────────────────────────────
// MLX90614 (temperatura)
// ─────────────────────────────────────────────

#define MLX_ADDR     0x5A
#define MLX_REG_TOBJ 0x07

float mlx_read() {
    Wire.beginTransmission(MLX_ADDR);
    Wire.write(MLX_REG_TOBJ);

    // O "false" pede um repeated START: nao liberta o barramento, porque a
    // leitura vem logo a seguir. Se o MLX nao responder e sairmos aqui sem
    // mais nada, o I2C fica preso a meio de uma transacao --- e como isto
    // corre a cada 250 ms, o MAX30102 deixa de conseguir entregar o FIFO,
    // mesmo estando ligado e com os LEDs acesos. Por isso, no caminho de
    // erro, fecha-se a transacao com um STOP para libertar o barramento.
    if (Wire.endTransmission(false) != 0) {
        Wire.endTransmission(true);
        return -1000.0f;
    }

    delayMicroseconds(200);
    Wire.requestFrom(MLX_ADDR, (uint8_t)3);
    if (Wire.available() < 3) return -1000.0f;
    uint8_t lo = Wire.read(), hi = Wire.read();
    Wire.read(); // PEC (ignorar)
    return ((hi << 8) | lo) * 0.02f - 273.15f;
}

// ─────────────────────────────────────────────
// Divisor de tensão da bateria (D0/A0, GPIO2) — diagnóstico
// ─────────────────────────────────────────────

static float vFrio = 0.0f, vEstab = 0.0f, vBatEst = 0.0f;

void lerTensaoBateria() {
    vFrio = analogReadMilliVolts(A0) / 1000.0f;

    for (int i = 0; i < 50; i++) {
        analogReadMilliVolts(A0);
        delayMicroseconds(200);
    }

    uint32_t soma = 0;
    for (int i = 0; i < 16; i++) {
        soma += analogReadMilliVolts(A0);
        delayMicroseconds(200);
    }
    vEstab  = soma / 16.0f / 1000.0f;
    vBatEst = 2.0f * vEstab;
}

// ─────────────────────────────────────────────
// DOUBLE-BUFFER + TASK MQTT (core 0)
// ─────────────────────────────────────────────

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
volatile uint32_t nAmostras = 0;   // amostras lidas do FIFO do MAX30102
volatile uint32_t nTempFalha = 0;  // leituras de temperatura falhadas

#define MAX_FALHAS_TEMP 8          // falhas seguidas ate desistir do MLX90614
static bool mlxAtivo = true;
static int  falhasSeguidas = 0;

TaskHandle_t mqttTask_h = NULL;

WiFiClient   espClient;
PubSubClient mqttClient(espClient);

void buildJson(const Amostra* buf, int n) {
    int pos = snprintf(jsonBuf, sizeof(jsonBuf),
        "{\"pulseira_id\":\"%s\",\"token\":\"%s\",\"amostras\":[",
        PULSEIRA_ID, AUTH_TOKEN);

    for (int i = 0; i < n && pos < (int)sizeof(jsonBuf) - 100; i++) {
        if (i > 0) jsonBuf[pos++] = ',';
        pos += snprintf(jsonBuf + pos, sizeof(jsonBuf) - pos,
            "{\"t\":%lu,\"ir\":%ld,\"red\":%ld,\"temp\":%.2f}",
            (unsigned long)buf[i].t,
            buf[i].ir,
            buf[i].red,
            buf[i].temp);
    }
    pos += snprintf(jsonBuf + pos, sizeof(jsonBuf) - pos,
        "],\"v_frio\":%.3f,\"v_estab\":%.3f,\"v_bat_est\":%.3f}",
        vFrio, vEstab, vBatEst);
}

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
    Serial.println("### ENSAIO DE BATERIA — só PPG e temperatura, sem AD8232 ###");

    // Sem AD8232: nenhum pino de ECG é configurado.

    // Divisor de tensão da bateria (diagnóstico) — D0/A0, GPIO2.
    pinMode(A0, INPUT);
    analogSetAttenuation(ADC_11db);

    Wire.begin(6, 7);
    if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
        Serial.println("ERRO: MAX30102 não encontrado");
        while (1);
    }

    particleSensor.setup(
        31,      // ledBrightness
        1,       // sampleAverage
        2,       // ledMode = 2 (IR + RED)
        100,     // sampleRate Hz
        215,     // pulseWidth µs
        16384    // adcRange
    );
    particleSensor.setPulseAmplitudeIR(0x22);
    particleSensor.setPulseAmplitudeRed(0x3D);

    // Varrimento do barramento I2C: diz logo o que esta ligado, em vez de se
    // descobrir pela ausencia de dados. 0x57 = MAX30102, 0x5A = MLX90614.
    Serial.print("I2C encontrados:");
    for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) {
            Serial.printf(" 0x%02X", addr);
            if (addr == 0x57) Serial.print("(MAX3010x)");
            if (addr == 0x5A) Serial.print("(MLX90614)");
        }
    }
    Serial.println();

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
        Serial.println("time_ms,ir,red,temp");
    }
}

// ─────────────────────────────────────────────
// VARIÁVEIS AUXILIARES
// ─────────────────────────────────────────────

static unsigned long lastTempMs  = 0;
static float         tempObj     = 0.0f;
static unsigned long lastWifiChk = 0;
static unsigned long lastStatsMs = 0;
static unsigned long lastVoltMs  = 0;

// ─────────────────────────────────────────────
// LOOP (core 1 — amostragem contínua)
// ─────────────────────────────────────────────

void loop() {

    // O MLX90614 so e consultado enquanto for dado como vivo. Se falhar
    // MAX_FALHAS_TEMP vezes seguidas, desiste-se dele: no ESP32 uma
    // transacao I2C falhada pode deixar o barramento num estado que impede
    // as leituras seguintes, e nao vale a pena um sensor avariado arrastar
    // consigo o PPG, que e o que interessa ao ensaio. A temperatura passa a
    // ir a zero no payload, e o contador continua visivel nas estatisticas.
    if (mlxAtivo && millis() - lastTempMs > 250) {
        float t = mlx_read();
        if (t > -100.0f) {
            tempObj = t;
            falhasSeguidas = 0;
        } else {
            nTempFalha++;
            if (++falhasSeguidas >= MAX_FALHAS_TEMP) {
                mlxAtivo = false;
                Wire.end();            // reiniciar o barramento I2C...
                delay(10);
                Wire.begin(6, 7);      // ...para o MAX30102 recuperar
                Serial.printf("[TEMP] MLX90614 sem resposta apos %d tentativas "
                              "--- desativado, barramento I2C reiniciado\n",
                              MAX_FALHAS_TEMP);
            }
        }
        lastTempMs = millis();
    }

    if (millis() - lastWifiChk > 10000) {
        if (!WiFi.isConnected()) {
            WiFi.reconnect();
        }
        lastWifiChk = millis();
    }

    if (millis() - lastVoltMs > 5000) {
        lerTensaoBateria();   // ~13 ms, bloqueante — cabe folgado num FIFO de 100 Hz
        if (DEBUG) {
            Serial.printf("[BAT] frio=%.3f  estab=%.3f  bat_est=%.3f\n",
                          vFrio, vEstab, vBatEst);
        }
        lastVoltMs = millis();
    }

    if (DEBUG && millis() - lastStatsMs > 10000) {
        Serial.printf("[MQTT] enviados=%lu falhados=%lu descartados=%lu | "
                      "amostras_PPG=%lu lote=%d | temp_falhadas=%lu\n",
                      (unsigned long)nEnviados,
                      (unsigned long)nFalhados,
                      (unsigned long)nDescartados,
                      (unsigned long)nAmostras,
                      collectCount,
                      (unsigned long)nTempFalha);
        lastStatsMs = millis();
    }

    particleSensor.check();

    while (particleSensor.available()) {

        uint32_t t   = millis();
        long     ir  = particleSensor.getFIFOIR();
        long     red = particleSensor.getFIFORed();
        nAmostras++;

        if (DEBUG) {
            Serial.printf("%lu,%ld,%ld,%.2f\n", (unsigned long)t, ir, red, tempObj);
        }

        // ─── Acumular no lote (sem qualquer condição de bloqueio) ───
        if (collectCount < BATCH_SIZE) {
            collectBuf[collectCount].t    = t;
            collectBuf[collectCount].ir   = ir;
            collectBuf[collectCount].red  = red;
            collectBuf[collectCount].temp = tempObj;
            collectCount++;
        }

        particleSensor.nextSample();
    }

    if (collectCount >= BATCH_SIZE && !sendPending) {
        memcpy(sendBuf, collectBuf, sizeof(Amostra) * BATCH_SIZE);
        buildJson(sendBuf, BATCH_SIZE);
        collectCount  = 0;
        sendPending   = true;
        xTaskNotifyGive(mqttTask_h);
    }
}
