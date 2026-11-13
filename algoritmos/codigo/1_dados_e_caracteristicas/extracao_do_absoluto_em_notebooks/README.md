# 1_dados_e_caracteristicas/extracao_do_absoluto_em_notebooks

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`Extração_dataset_total.ipynb`](Extração_dataset_total.ipynb) | Copia as pastas dos pacientes válidos de cada faixa etária para um dataset consolidado e gera o CSV mestre, com um segmento por linha e a idade e o sexo do doente (MIMIC-III e MIMIC-IV). |
| [`Extração_features.ipynb`](Extração_features.ipynb) | Extração das características das janelas do dataset consolidado (ECG, PPG e ABP) em três variantes, cada uma com a sua extração e a visualização das deteções. |
| [`Extração_features_FFT.ipynb`](Extração_features_FFT.ipynb) | Variante que acrescenta às características o espetro de frequência do batimento do ECG (`ecg_fft_mag_00..23`), inspirada em Mousavi et al. (2018). |
