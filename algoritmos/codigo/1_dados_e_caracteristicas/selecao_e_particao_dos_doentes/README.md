# 1_dados_e_caracteristicas/selecao_e_particao_dos_doentes

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`Seleção_subjects.ipynb`](Seleção_subjects.ipynb) | Seleção dos pacientes do MIMIC-III por faixa etária (`min_age`, `max_age`), só os que estiveram em UCI, onde é mais provável haver ABP, e descarga dos seus registos. |
| [`gerar_holdout.py`](gerar_holdout.py) | Gera o conjunto de holdout (pacientes_holdout.txt) por amostragem aleatória estratificada (faixa etária x género), com seed fixa para reprodutibilidade. |
