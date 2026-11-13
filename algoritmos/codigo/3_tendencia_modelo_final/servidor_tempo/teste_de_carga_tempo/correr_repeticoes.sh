#!/bin/bash
# 3 repeticoes independentes entre 400 e 700 pulseiras, os tres modelos em sequencia (nunca em paralelo).
cd "$(dirname "$0")"
PY=/home/filipe/Desktop/Tese/algoritmo-ml/venv/bin/python3
for r in 1 2 3; do
  for m in xgb bilstm bitcn; do
    $PY -u teste_de_carga_tempo.py --modelo $m --duracao 45 --niveis 300,350,400,450,500 --sufixo _rep$r 2>&1 | grep --line-buffered -v "Warning\|WARNING\|^I0\|^W0\|cuda\|E0000" > registo_carga_${m}_rep$r.log
  done
done
touch REPETICOES_TERMINADAS
