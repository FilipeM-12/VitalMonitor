# 1_dados_e_caracteristicas/caracterizacao_dos_conjuntos_de_dados

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`caracterizar_dataset.py`](caracterizar_dataset.py) | Caracterização dos conjuntos de dados do MIMIC usados nas duas formulações (para a secção inicial do Cap. 6). |
| [`comparar_fc_ppg_ecg.py`](comparar_fc_ppg_ecg.py) | A FC que a extração só-PPG dá (picos do PPG, proeminência mínima de 15 % da amplitude da janela) contra a FC que a extração com ECG dá (intervalos RR), nas mesmas janelas do MIMIC (mesma subject_id, segment_name, window_idx). |
