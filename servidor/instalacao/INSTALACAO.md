# Instalação do servidor numa máquina nova

Como pôr a base de dados, o broker MQTT, o painel e os workers a correr do zero, num Debian ou Ubuntu. Os comandos correm todos
no mesmo terminal, a partir da raiz deste repositório. As credenciais são as de desenvolvimento (ver o
[`README.md`](../../README.md) da raiz) e devem ser mudadas numa instalação real.

```bash
cd <raiz do repositório>
SERVIDOR="$PWD/servidor"
```

| Ficheiro desta pasta | Para quê |
|---|---|
| [`mosquitto_hospital.conf`](mosquitto_hospital.conf) | Configuração do Mosquitto (passo 2). |
| [`pulseiras.service`](pulseiras.service) | Unidade systemd do painel (passo 5). |
| [`pulseiras-worker@.service`](pulseiras-worker@.service) | Unidade systemd dos workers, uma instância por número (passo 5). |
| [`Caddyfile`](Caddyfile) | Configuração do HTTPS com o Caddy (passo 6). |
| [`requirements.txt`](requirements.txt) | Pacotes Python do servidor (passo 3). |

## 1. Base de dados (PostgreSQL)

```bash
sudo apt install postgresql
sudo -u postgres psql -c "CREATE USER hospital WITH PASSWORD 'hospital_pass';"
sudo -u postgres psql -c "CREATE DATABASE hospital OWNER hospital;"
```

A aplicação lê `DATABASE_URL`; por omissão é `postgresql://hospital:hospital_pass@localhost:5432/hospital`. Só é preciso
definir a variável se se quiser outro valor.

## 2. Broker MQTT (Mosquitto)

```bash
sudo apt install mosquitto mosquitto-clients
sudo cp "$SERVIDOR/instalacao/mosquitto_hospital.conf" /etc/mosquitto/conf.d/hospital.conf
sudo mosquitto_passwd -c /etc/mosquitto/passwd pulseira   # password: pulseira_pass
sudo mosquitto_passwd    /etc/mosquitto/passwd worker     # password: worker_pass
sudo systemctl restart mosquitto
```

As pulseiras ligam-se com o utilizador `pulseira` e o painel e os workers com o `worker`. A configuração escuta na porta 1883,
não aceita ligações anónimas, admite até 500 clientes, aceita mensagens até 8192 bytes, não guarda mensagens em disco e só
regista erros e avisos.

## 3. Python e dependências

```bash
cd "$SERVIDOR"
python3 -m venv venv
./venv/bin/pip install -r instalacao/requirements.txt
cd painel && ../venv/bin/python3 server.py --setup       # cria as tabelas e as configurações iniciais
cd "$SERVIDOR"
```

## 4. Leitor de Cartão de Cidadão (opcional, só se o `/ler_cartao` for usado)

1. Compilar e instalar o middleware oficial (`autenticacao.gov/pteid-mw-pt`, ver o README dessa pasta) ou instalar o pacote
   `pteid-mw` oficial.
2. Compilar o binário local:

   ```bash
   cd "$SERVIDOR/painel" && make build && cd "$SERVIDOR"
   ```

3. **Regra de polkit.** Sem ela, o `pcscd` rejeita sempre o pedido do serviço systemd com
   `NOT authorized for action: access_pcsc`. A política `org.debian.pcsc-lite` só autoriza por omissão sessões gráficas
   ativas, e um serviço de sistema não tem nenhuma. Isto só se nota quando se corre como serviço: correr `python3 server.py`
   à mão num terminal funciona sempre, porque herda a sessão gráfica, e disfarça o problema até se passar para produção.

   ```bash
   sudo tee /etc/polkit-1/rules.d/50-pcsc-pulseiras.rules > /dev/null <<EOF
   polkit.addRule(function(action, subject) {
       if ((action.id == "org.debian.pcsc-lite.access_pcsc" ||
            action.id == "org.debian.pcsc-lite.access_card") &&
           subject.user == "$USER") {
           return polkit.Result.YES;
       }
   });
   EOF
   sudo systemctl restart polkit
   ```

   O utilizador da regra é o que corre o `pulseiras.service`, que é o mesmo do passo 5.

## 5. Serviços systemd (painel e workers)

As unidades trazem os caminhos e o utilizador da máquina de desenvolvimento. O `sed` troca-os pelos desta máquina ao instalar:

```bash
for u in pulseiras.service pulseiras-worker@.service; do
  sed -e "s|/home/filipe/Desktop/Tese/servidor|$SERVIDOR|g" \
      -e "s|^User=.*|User=$USER|" -e "s|^Group=.*|Group=$(id -gn)|" \
      "$SERVIDOR/instalacao/$u" | sudo tee "/etc/systemd/system/$u" > /dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable --now pulseiras.service
sudo systemctl enable --now pulseiras-worker@{0,1,2,3}.service
```

O `pulseiras.service` serve o painel com o Gunicorn (4 processos) em `127.0.0.1:5000`, atrás do Caddy (passo 6). Os `pulseiras-worker@0` a `@3` são as quatro
instâncias do worker de tendência (`NUM_WORKERS=4`); cada pulseira cai sempre na mesma instância. Para ver se está tudo a correr:

```bash
sudo systemctl status pulseiras.service
sudo systemctl status "pulseiras-worker@*"
```

## 6. HTTPS com o Caddy

Instalar o Caddy, que o Ubuntu 24.04 traz nos repositórios (é a versão 2.6.2, com que o Caddyfile desta pasta foi validado):

```bash
sudo apt install caddy
```

Configurar e arrancar:

```bash
sudo cp "$SERVIDOR/instalacao/Caddyfile" /etc/caddy/Caddyfile
sudo systemctl restart caddy
```

O painel só escuta em `127.0.0.1:5000`: o acesso é sempre pelo Caddy, que redireciona a porta 80 para HTTPS e passa o pedido da 443 ao painel. Abre-se em `https://<IP desta máquina>`. O certificado é emitido
pela autoridade interna do Caddy (`tls internal`), por isso os navegadores avisam até se instalar o certificado dessa
autoridade. Para reiniciar, ativar ou desativar o Caddy: `sudo systemctl restart|enable|disable caddy`.

## 7. Pulseiras (Arduino)

Pôr em `MQTT_HOST`, no `.ino` (por exemplo `arduino/firmware/ECG_e_PPG/ECG_e_PPG.ino`), o IP desta máquina na rede local, e
voltar a gravar o firmware no ESP32.

## 8. Rede

```bash
hostname -I                       # IP desta máquina, para pôr no firmware
sudo ufw allow 80,443/tcp         # para outros dispositivos acederem ao painel, por HTTPS
sudo ufw allow 1883               # para as pulseiras chegarem ao broker
```

Os comandos `ufw` só são precisos se a firewall estiver ativa. A porta 5000 não se abre: o painel só escuta em `127.0.0.1`.

## Variáveis de ambiente

Todas têm um valor por omissão que funciona com os passos acima. Para mudar uma, define-se com `Environment=` na unidade
systemd ou no ambiente de quem arranca o processo.

| Variável | Quem a lê | Por omissão | Para quê |
|---|---|---|---|
| `DATABASE_URL` | painel e workers | `postgresql://hospital:hospital_pass@localhost:5432/hospital` | Ligação à base de dados. |
| `MQTT_BROKER`, `MQTT_PORT` | painel e workers | `localhost`, `1883` | Onde está o Mosquitto. |
| `MQTT_USER`, `MQTT_PASS` | painel e workers | `worker`, `worker_pass` | Utilizador do passo 2. |
| `AUTH_TOKEN` | painel e workers | `segredo123` | Token que cada mensagem da pulseira leva; tem de ser igual ao `AUTH_TOKEN` do `.ino`. |
| `WORKER_ID`, `NUM_WORKERS` | workers | `0`, `1` | Número da instância e total de instâncias; a unidade põe `%i` e `4`. |
| `THREAD_POOL_SIZE` | workers | `8` | Threads de cada worker. |
| `MODELO_TENDENCIA` | painel | `servidor/modelos/modelo_tendencia_ppg_quant.joblib` | Modelo de onde o painel lê os limiares. |
| `DUMP_BRUTO_DIR` | workers | vazio | Se definida, grava o sinal em bruto, um ficheiro por pulseira, para sessões curtas de diagnóstico. |
