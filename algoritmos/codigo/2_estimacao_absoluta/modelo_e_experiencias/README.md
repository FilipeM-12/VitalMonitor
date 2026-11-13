# 2_estimacao_absoluta/modelo_e_experiencias

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`01_correr_experiencia.py`](01_correr_experiencia.py) | Uma experiência do absoluto: busca de hiperparâmetros (3 dobras, dados causais) e avaliação única no holdout. python 01_correr_experiencia.py FAMILIA VARIANTE [--trials 20] [--paralelo 3] [--ancoras causal] FAMILIA XGB / BILSTM /… |
| [`02_sementes_bootstrap.py`](02_sementes_bootstrap.py) | Ganho dos híbridos sobre o XGBoost no absoluto, com a incerteza de sementes e de amostragem. As redes sozinhas também se treinam com as 3 sementes (para a tabela dos modelos, que mostra o MAE de cada arquitetura por semente). |
| [`03_tabelas_modelos.py`](03_tabelas_modelos.py) | tab:modelos_absoluto e figura do planalto (dispersão das tentativas face à incerteza da métrica). As cinco famílias vêm de 01_correr_experiencia.py (variante baseline, 20 tentativas cada, 3 dobras). |
| [`04_suavizacao.py`](04_suavizacao.py) | tab:suavizacao_absoluto. A mesma configuração de referência (XGBoost da variante baseline, mesmos hiperparâmetros, mesma divisão de sujeitos) com três suavizações das previsões: Kalman, EWMA e nenhuma. Só muda o pós-processamento. |
| [`05_versoes_features.py`](05_versoes_features.py) | tab:versoes_absoluto. As quatro versões incrementais das características, todas com os MESMOS hiperparâmetros (os do XGBoost de referência) e a mesma divisão de sujeitos: v1 PAT. |
| [`06_tabela_variantes.py`](06_tabela_variantes.py) | tab:variantes_absoluto. As variantes de extração e de covariáveis, cada uma com a sua busca (01_correr_experiencia.py). |
| [`07_ancora_bhs_shap_deriva.py`](07_ancora_bhs_shap_deriva.py) | Âncora de calibração, graus BHS, Shapley e figura da deriva (Capítulo 6, "Síntese: um patamar de erro reprodutível"). Lê as previsões do holdout de 01_correr_experiencia.py. |
| [`08_custo_normalizacao_causal.py`](08_custo_normalizacao_causal.py) | Quanto custa normalizar as âncoras morfológicas com a janela de calibração do doente (causal) em vez do doente inteiro (que olha para o futuro e não se pode calcular à cabeceira)? Para cada variante, mesma coorte. |
| [`09_analise_deriva.py`](09_analise_deriva.py) | Análise da deriva do absoluto (Cap. 6, "O Erro é Deriva e não Ruído" e "Onde Está a Informação Utilizável"), sobre as previsões causais do holdout da variante de referência (nada se treina). |
| [`10_janelas_emparelhadas.py`](10_janelas_emparelhadas.py) | Extração própria contra NeuroKit/NeuroKit2 SÓ nas janelas que ambas as variantes têm (Cap. 6, "Cadeias de extração"). O MAE da variante NeuroKit é muito menor do que o da extração própria, mas retém poucos sujeitos. |
| [`11_bland_altman.py`](11_bland_altman.py) | Erro quadrático médio e análise de Bland-Altman do modelo de referência (XGBoost, variante baseline), no holdout. Referência = PA sistólica invasiva (sbp_target); estimativa = preds_final. |
| [`12_ganho_contra_ancora.py`](12_ganho_contra_ancora.py) | O modelo bate a âncora (repetir a última calibração)? Diferença de erro absoluto (âncora - modelo; positivo = o modelo ganha), no conjunto todo e por faixa de tempo desde a calibração. |
| [`14_reprodutibilidade.py`](14_reprodutibilidade.py) | A mesma semente dá o mesmo resultado? Treina cada modelo várias vezes com os MESMOS hiperparâmetros, os mesmos dados e a MESMA semente (a semente 42 da corrida principal) e mede o MAE de holdout de cada corrida. |
| [`16_direcao_variacao.py`](16_direcao_variacao.py) | Acerto na direção da variação desde a calibração (janelas com /variação/>5 mmHg) do modelo de referência, no conjunto de teste. |
| [`17_efeito_hiperparametros.py`](17_efeito_hiperparametros.py) | Planalto (Cap. 6): que hiperparâmetros acompanham o erro nas 20 tentativas do Optuna do XGBoost de referência. |
| [`18_desvio_de_reserva.py`](18_desvio_de_reserva.py) | Desvio de reserva da normalização causal, para o servidor. nucleo.dados.normalizar_por_calibracao divide pelo desvio da janela de calibração da sessão mais 1e-6. |
| [`19_verificar_servidor.py`](19_verificar_servidor.py) | O servidor calcula o mesmo que o treino, com as 27 características. Corre depois do 18, que dá as medianas de reserva da normalização (comum.SD_RESERVA tem de ser igual). |
| [`20_extracao_servidor_vs_treino.py`](20_extracao_servidor_vs_treino.py) | Mede, sobre sinal do MIMIC, em que a extração das características do absoluto no servidor (`comum.extrair_features_janela` / `comum.processar_janela_absoluto`) difere da do treino (`processar_janela_teste1` do… |
| [`correr_absoluto.sh`](correr_absoluto.sh) | Absoluto: a cadeia do absoluto com 27 características, mesmos hiperparâmetros, sem Optuna (nucleo/busca.py recusa tentativas novas e recalcula o MAE de validação cruzada). |

Subpastas: [`nucleo/`](nucleo/).
