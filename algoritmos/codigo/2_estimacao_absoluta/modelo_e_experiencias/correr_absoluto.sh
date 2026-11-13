#!/bin/bash
# Absoluto: a cadeia do absoluto com 27 características, mesmos
# hiperparâmetros, sem Optuna (nucleo/busca.py recusa tentativas novas e recalcula o MAE de validação cruzada).
# Corre em sequência a fila A (XGBoost e variantes), a fila B (redes, híbridos e sementes) e as análises da tese.
# Retomável: estado/<passo>.ok.
AQUI="$(cd "$(dirname "$0")" && pwd)"
PY=/home/filipe/Desktop/Tese/algoritmo-ml/venv/bin/python
cd "$AQUI" || exit 1
mkdir -p estado logs
export TF_CPP_MIN_LOG_LEVEL=2 NUCLEOS_MAX=10

passo () {
    local nome="$1"; shift
    [ -f "estado/$nome.ok" ] && { echo "[$(date '+%F %T')] $nome: ja feito"; return 0; }
    rm -f "estado/$nome.falhou"; echo "[$(date '+%F %T')] $nome: INICIO"
    ( "$@" ) >> "logs/$nome.log" 2>&1; local c=$?
    if [ $c -eq 0 ]; then touch "estado/$nome.ok"; echo "[$(date '+%F %T')] $nome: OK"
    else echo "codigo $c" > "estado/$nome.falhou"; echo "[$(date '+%F %T')] $nome: FALHOU ($c)"; fi
    return $c
}

# A. XGBoost e variantes
passo A1_XGB_baseline   $PY -u 01_correr_experiencia.py XGB baseline --paralelo 2 || exit 1
passo A2_XGB_nk         $PY -u 01_correr_experiencia.py XGB nk --paralelo 2
passo A3_XGB_nk2        $PY -u 01_correr_experiencia.py XGB nk2 --paralelo 2
passo A4_XGB_clinicas   $PY -u 01_correr_experiencia.py XGB clinicas --paralelo 2
passo A5_suavizacao     $PY -u 04_suavizacao.py
passo A6_versoes        $PY -u 05_versoes_features.py
passo A7_XGB_fft        $PY -u 01_correr_experiencia.py XGB fft --paralelo 2
passo A8_custo_causal   $PY -u 08_custo_normalizacao_causal.py
# B. redes, híbridos e sementes (modo determinista do TensorFlow, ligado no topo do 01 e do 02)
passo B1_BILSTM         $PY -u 01_correr_experiencia.py BILSTM baseline
passo B2_BITCN          $PY -u 01_correr_experiencia.py BITCN baseline
passo B3_HIB_BILSTM     $PY -u 01_correr_experiencia.py HIBRIDO_BILSTM_XGB baseline
passo B4_HIB_BITCN      $PY -u 01_correr_experiencia.py HIBRIDO_BITCN_XGB baseline
passo B5_sementes       $PY -u 02_sementes_bootstrap.py
# C. análises da tese
passo C03_tabelas       $PY -u 03_tabelas_modelos.py
passo C06_variantes     $PY -u 06_tabela_variantes.py
# o 07 lê ganho_contra_ancora.json do 12 (título da figura da deriva): o 12 tem de correr antes
passo C12_ganho_ancora  $PY -u 12_ganho_contra_ancora.py
passo C07_ancora_shap   $PY -u 07_ancora_bhs_shap_deriva.py
passo C09_deriva        $PY -u 09_analise_deriva.py
passo C10_emparelhadas  $PY -u 10_janelas_emparelhadas.py
passo C11_bland_altman  $PY -u 11_bland_altman.py
passo C16_direcao       $PY -u 16_direcao_variacao.py
passo C14_reprodutib    $PY -u 14_reprodutibilidade.py
passo C14_determinista  $PY -u 14_reprodutibilidade.py determinista
echo "[$(date '+%F %T')] FILA DO ABSOLUTO TERMINADA"; touch estado/FILA_TERMINADA
