# absoluto · os resultados da estimação da pressão sistólica absoluta

Secção 6.3 da dissertação. Os ficheiros de `tabelas/` são os que os scripts de
[`algoritmos/codigo/2_estimacao_absoluta/modelo_e_experiencias/`](../../../algoritmos/codigo/2_estimacao_absoluta/modelo_e_experiencias/) escreveram,
sem alterações. Os ficheiros `cv_*.csv` e a pasta `optuna/` têm os resultados da afinação, que servem para escolher entre modelos.

## A escolha entre as cinco arquiteturas, no conjunto de desenvolvimento

Os cinco modelos treinam com as mesmas 27 características e a mesma divisão de doentes. Cada um tem uma pesquisa de hiperparâmetros com o Optuna, com 20
tentativas, e cada tentativa avalia-se por validação cruzada com 3 dobras agrupadas por doente, só no conjunto de desenvolvimento (244 doentes e 321 667 janelas). O conjunto de
teste não entra na escolha. O MAE de uma tentativa é a média do MAE das três dobras.

| Modelo | MAE CV, melhor tentativa | MAE CV, mediana | MAE CV, pior | MAE no teste, janelas comuns |
|---|---|---|---|---|
| XGBoost | 9,56 | 9,63 | 9,95 | 12,31 |
| BiLSTM | 10,13 | 10,50 | 10,69 | 11,27 |
| BiTCN | 10,02 | 10,13 | 10,69 | 11,83 |
| Híbrido BiLSTM–XGBoost | 10,22 | 10,57 | 10,85 | 11,71 |
| Híbrido BiTCN–XGBoost | 10,02 | 10,26 | 10,98 | 11,74 |

Valores em mmHg. O MAE no teste é o da semente 1 da Tabela 6.3, nas 48 999 janelas que os cinco modelos preveem. As redes só começam no quinto bloco de cada sessão.

O XGBoost é o melhor no desenvolvimento, com 0,46 a 0,66 mmHg de vantagem sobre os outros quatro. No conjunto de teste a ordem
inverte-se um pouco, mas o intervalo de confiança por doente da diferença inclui o zero em quatro dos seis casos dos híbridos (Secção 6.3.2). O
XGBoost foi adotado como modelo de referência por oferecer um desempenho próximo com muito menos recursos.

Dois cuidados ao ler a tabela:

* O MAE de CV do XGBoost conta todas as janelas do desenvolvimento, e o das redes e dos híbridos só as que estas preveem. O do teste usa só as janelas comuns, e o de CV não.
* O MAE de CV da «melhor tentativa» é o mínimo de 20, por isso é um pouco otimista. Só há um valor por tentativa (a média das três dobras), sem o de cada dobra.

## Ficheiros

| Ficheiro | O que tem |
|---|---|
| [`cv_resumo.csv`](cv_resumo.csv) | Uma linha por estudo (as cinco arquiteturas e as quatro variantes do XGBoost): a melhor tentativa, a mediana e a pior, a melhor até à tentativa 5, 10, 15 e 20, o tamanho do desenvolvimento e do teste e o MAE no teste. |
| [`cv_por_tentativa.csv`](cv_por_tentativa.csv) | O MAE de CV de cada uma das 20 tentativas de cada estudo, e o melhor até então. A Figura 6.7 desenha-as. |
| [`optuna/`](optuna/) | Uma pasta por estudo, com `trials.csv` (o MAE de CV de cada tentativa), `best_params.json` (os hiperparâmetros escolhidos e o MAE de CV com eles) e `metricas.json` (a avaliação única no teste e a coorte). |
| [`tabelas/`](tabelas/) | As tabelas dos scripts de `modelo_e_experiencias/`, que a lista abaixo identifica. |

As variantes retêm doentes diferentes (`sujeitos_dev` em `cv_resumo.csv`), e as NeuroKit (`XGB_nk` e `XGB_nk2`) bastantes menos, por isso o MAE delas
não é comparável com o do `XGB_baseline`. `tabelas/variantes_absoluto.csv` dá também o MAE só nas janelas que cada variante tem em comum com a referência
(Tabela 6.4).

## O que cada tabela responde

| Ficheiro em `tabelas/` | Script | O que mostra |
|---|---|---|
| `modelos_absoluto.csv`, `planalto_incerteza.json` | `03_tabelas_modelos.py` | O MAE de CV e no teste das cinco arquiteturas (Tabela 6.3) e a incerteza das tentativas face à largura do intervalo (Figura 6.7). |
| `sementes_bootstrap.json`, `reprodutibilidade*.csv`, `reprodutibilidade*.json` | `02_sementes_bootstrap.py`, `14_reprodutibilidade.py` | O MAE no teste de cada arquitetura com três sementes, o ganho dos híbridos sobre o XGBoost e o seu intervalo (bootstrap por doente, no teste), e se a mesma semente dá o mesmo resultado. |
| `efeito_hiperparametros.json` | `17_efeito_hiperparametros.py` | Que hiperparâmetros acompanham o MAE de validação cruzada nas 20 tentativas do XGBoost, no desenvolvimento (Secção 6.3.6). |
| `suavizacao.csv`, `suavizacao.json` | `04_suavizacao.py` | O mesmo modelo com Kalman, EWMA e sem suavização. |
| `versoes_features.csv`, `versoes_features.json` | `05_versoes_features.py` | As quatro versões incrementais das características, com os mesmos hiperparâmetros (Tabela B.3). |
| `variantes_absoluto.csv` | `06_tabela_variantes.py` | As variantes de extração e de covariáveis (Tabela 6.4). |
| `janelas_emparelhadas.csv` | `10_janelas_emparelhadas.py` | A extração própria contra NeuroKit e NeuroKit2, só nas janelas que ambas têm, nas previsões do teste. |
| `ancora_bhs.json`, `ancora_horizonte.csv`, `ancora_variantes.csv`, `efeito_janela_calibracao.csv`, `ganho_contra_ancora.json`, `shap_importancia.csv` | `07_ancora_bhs_shap_deriva.py`, `12_ganho_contra_ancora.py` | Sobre as previsões do teste, a âncora (repetir a última calibração), os graus BHS, a importância das características (SHAP, numa amostra do teste) e a diferença de erro entre a âncora e o modelo. |
| `bland_altman_final.json` | `11_bland_altman.py` | No teste, o enviesamento, os limites de concordância e o declive da diferença contra a média (Figura 6.5). |
| `deriva_absoluto.json`, `direcao_variacao.json` | `09_analise_deriva.py`, `16_direcao_variacao.py` | Sobre as previsões do teste, a análise da deriva do erro, incluindo a autocorrelação, e o acerto na direção das variações desde a calibração (Figuras 6.8 e 6.10). |
| `custo_normalizacao_causal.csv`, `custo_normalizacao_causal.json` | `08_custo_normalizacao_causal.py` | O que custa normalizar as âncoras morfológicas só com a janela de calibração, em vez do doente inteiro (medido no teste). |
| `sd_reserva_calibracao.json`, `verificacao_servidor.json` | `18_desvio_de_reserva.py`, `19_verificar_servidor.py` | As medianas de reserva da normalização e a verificação de que o servidor calcula o mesmo que o treino. |

## Desenvolvimento e teste

As escolhas (arquitetura, hiperparâmetros, janela, limiares) fazem-se só com as medidas do desenvolvimento. O conjunto de teste só se usa no fim, para relatar o
resultado. Nos ficheiros, as colunas do teste têm `teste` ou `holdout` no nome (`auc_teste`, `teste_auc`, `mae_holdout`) e as do desenvolvimento costumam ter `dev` ou
`cv` (`auc_dev`, `mae_cv`). Alguns ficheiros trazem as duas lado a lado. Um ficheiro sem estas marcas não diz de que conjunto é, e a pasta e o script que o escreveu
dizem de que dados vêm.
