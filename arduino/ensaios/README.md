# arduino/ensaios · ferramentas dos ensaios de bancada

Usam-se por esta ordem: gravar o sinal, anotar as medições de referência durante o ensaio e comparar tudo no fim. As gravações
dos ensaios com pessoas identificadas não estão no repositório.

| Ficheiro | O que faz |
|---|---|
| [`gravar_mqtt.py`](gravar_mqtt.py) | Cliente do broker, ao lado dos workers: grava em CSV o sinal em bruto que as pulseiras publicam, um ficheiro por pulseira e por execução. |
| [`anotar_ensaio.py`](anotar_ensaio.py) | Folha de registo preenchida no terminal, ao mesmo tempo que a gravação: cada frequência cardíaca, SpO₂, contagem de respirações ou nota fica gravada com a hora do relógio do PC. |
| [`analisar_ensaio_ppg.py`](analisar_ensaio_ppg.py) | Compara a frequência cardíaca, a SpO₂ e a frequência respiratória do servidor com as medições de referência e gera a figura e os números. Lê a gravação e a folha de registo, esta com as colunas `hora`, `respiracoes`, `fc_smartband`, `spo2_smartband` e `notas`. |
