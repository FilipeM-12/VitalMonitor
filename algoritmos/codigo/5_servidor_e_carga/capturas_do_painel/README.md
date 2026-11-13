# 5_servidor_e_carga/capturas_do_painel

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`capturar.py`](capturar.py) | Capturas do painel de controlo para o anexo, com o Brave sem janela (Playwright, sem browsers próprios). |
| [`demo.py`](demo.py) | Doentes de demonstração para as capturas do painel. python demo.py criar/apagar Cria 3 doentes fictícios («Paciente 1, 2, 3») com pulseira, atribuição ativa e calibração. |
| [`esperar.py`](esperar.py) | Espera até cada um dos 3 doentes DEMO ter pelo menos N previsões de tendência (omissão 12, ou seja 3 min de blocos). |
| [`mimic_para_csv.py`](mimic_para_csv.py) | Converte segmentos reais do MIMIC (PLETH e ECG derivação II) para o CSV que o simulador reproduz (formato da bancada: time_ms, ecg, ir, red a 100 Hz), para as capturas do painel. Só para ilustração; não entra em resultados. |
