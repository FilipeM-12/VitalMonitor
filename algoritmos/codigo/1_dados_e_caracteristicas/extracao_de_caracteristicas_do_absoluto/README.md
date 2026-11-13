# 1_dados_e_caracteristicas/extracao_de_caracteristicas_do_absoluto

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`extrair_features_clinicas.py`](extrair_features_clinicas.py) | Extrai, para cada janela da "espinha dorsal" (janelas_absolutas_TODOS.csv), variáveis clínicas com timestamp que plausivelmente modulam o tónus arterial e por isso a relação PAT <-> pressão arterial. |
| [`extrair_features_fft.py`](extrair_features_fft.py) | 1 thread por worker (a FFT/scipy usa BLAS multi-thread; com N_JOBS=10 os workers competiriam). Tem de ser antes de importar numpy. |
| [`extrair_features_nk2.py`](extrair_features_nk2.py) | Funções: `butter_lowpass_filter`, `butter_highpass_filter`, `remover_picos_isolados`, `selecionar_ecg`, `extrair_targets_abp_robusto` |
| [`extrair_janelas_absolutas.py`](extrair_janelas_absolutas.py) | Constrói a "espinha dorsal" temporal: para CADA janela bruta (subject_id, segment_name, segment_order, window_idx) de TODOS os pacientes com dados reais (MIMIC-III + MIMIC-IV, não só os afetados por fármaco vasoativo). |
| [`identificar_janelas_vasoativas.py`](identificar_janelas_vasoativas.py) | Identifica, ao nível da JANELA (15s, stride 5s -- mesma convenção do Extração_features.ipynb), quais janelas de sinal se sobrepõem a uma administração de fármaco vasoativo (vasopressor ou vasodilatador). |
| [`identificar_janelas_vasoativas_mimic4.py`](identificar_janelas_vasoativas_mimic4.py) | Equivalente a identificar_janelas_vasoativas.py, mas para os pacientes MIMIC-IV (pastas com 8 dígitos, ex. p10126957) em MIMIC_DATASET_CONSOLIDADO. Diferenças-chave face ao MIMIC-III. |
| [`identificar_pacientes_vasoativos.py`](identificar_pacientes_vasoativos.py) | Identifica, entre os pacientes com dados reais em MIMIC_DATASET_CONSOLIDADO, quais receberam fármacos vasoativos (vasopressores ou vasodilatadores) durante o internamento -- candidatos a descartar. |
