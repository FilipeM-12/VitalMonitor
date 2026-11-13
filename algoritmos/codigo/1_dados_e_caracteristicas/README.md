# 1_dados_e_caracteristicas

| Pasta | O que faz |
|---|---|
| [`selecao_e_particao_dos_doentes/`](selecao_e_particao_dos_doentes/) | Seleção dos doentes do MIMIC (notebook), partição em desenvolvimento e teste (`gerar_holdout.py`) e a lista dos doentes de teste do absoluto. |
| [`extracao_de_caracteristicas_do_absoluto/`](extracao_de_caracteristicas_do_absoluto/) | Extração das janelas e das características do absoluto (clínicas, FFT e NeuroKit2) e identificação dos fármacos vasoativos. |
| [`extracao_do_absoluto_em_notebooks/`](extracao_do_absoluto_em_notebooks/) | Os notebooks com que se começou a extrair as características do absoluto, anteriores aos scripts. |
| [`extracao_ppg_e_segunda_derivada_da_tendencia/`](extracao_ppg_e_segunda_derivada_da_tendencia/) | Extração das características só com PPG e com a segunda derivada do pulso (APG), de que sai a primeira versão das características da tendência (a versão final está em `3_tendencia_modelo_final/extracao_morfologia/`) e a correlação no espaço de variação. |
| [`tabelas_de_treino_e_teste_da_tendencia/`](tabelas_de_treino_e_teste_da_tendencia/) | Constrói as tabelas de treino e de teste da tendência (blocos de 15 s por sessão, normalização causal), as que todos os modelos usam e que o servidor reproduz, e mede quanto tempo cobrem os «20 blocos válidos». |
| [`caracterizacao_dos_conjuntos_de_dados/`](caracterizacao_dos_conjuntos_de_dados/) | Caracterização dos conjuntos de dados das duas formulações e concordância da frequência cardíaca do PPG com a do ECG. |
