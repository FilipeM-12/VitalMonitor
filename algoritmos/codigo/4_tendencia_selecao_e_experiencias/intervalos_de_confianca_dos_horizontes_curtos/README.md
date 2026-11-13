# 4_tendencia_selecao_e_experiencias/intervalos_de_confianca_dos_horizontes_curtos

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`ic_curtos_vs_5.py`](ic_curtos_vs_5.py) | Generaliza ic_so3_vs_so5.py a 2 e 3 min. Para cada horizonte curto C (2 ou 3 min), compara os episódios que só a definição de 5 min apanha (sem sobreposição com os de C) com os que só a de C apanha (sem sobreposição com os de 5). |
| [`ic_curtos_vs_5_tabela_atual.py`](ic_curtos_vs_5_tabela_atual.py) | Igual ao ic_curtos_vs_5.py mas sobre a tabela de teste atual (causal09/dados/holdout_ppg.csv), porque holdout_ppg_apg.csv já não existe. Generaliza ic_so3_vs_so5.py a 2 e 3 min. |
