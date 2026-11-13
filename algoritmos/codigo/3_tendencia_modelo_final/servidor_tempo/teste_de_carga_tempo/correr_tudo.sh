#!/bin/bash
# Corre os tres modelos em sequencia (nunca em paralelo: partilhariam CPU e falseariam a latencia).
cd "$(dirname "$0")"
PY=/home/filipe/Desktop/Tese/algoritmo-ml/venv/bin/python3
for m in xgb bilstm bitcn; do
    $PY -u teste_de_carga_tempo.py --modelo $m --duracao 45 2>&1 | grep --line-buffered -v "Warning\|WARNING\|^I0\|^W0\|cuda\|E0000" | tee registo_carga_$m.log
done
touch TERMINADO
