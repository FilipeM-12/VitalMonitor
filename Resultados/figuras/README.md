# Figuras · dados e código para refazer vinte e dois gráficos da dissertação

Cada subpasta tem o número da figura e o seu nome, os **dados** que a figura desenha e o script **`gerar_figura.py`**, que lê esses dados da própria
pasta (caminho relativo) e escreve a figura ao lado. A imagem que lá está já foi gerada assim, e é idêntica byte a byte à da dissertação (o PDF da
Figura 6.10, sem as datas dos metadados).

Para refazer uma figura: `python3 gerar_figura.py`, dentro da subpasta. São precisos o Python 3, o `matplotlib`, o `numpy` e o `pandas`.
Para mudar o aspeto, é nesse script que se mexe (tamanho, tipos de letra, cores, marcas); os dados não mudam.

| Figura | Subpasta | Dados | Imagem |
|---|---|---|---|
| 6.1 · Pulso do PPG e batimento do ECG, com as grandezas extraídas | `6.1_pulso_ecg` | `sinais.csv`, `pontos.json` | `pulso_ecg_caracteristicas_morf_final.png` |
| 6.3 · O PPG e o ECG antes e depois do filtro | `6.3_sinal_filtrado` | `ppg.csv`, `ecg.csv` | `sinal_filtrado_final.png` |
| 6.4 · Os portões de qualidade: uma janela aceite e uma rejeitada | `6.4_portoes` | `janela_aceite.csv`, `janela_rejeitada.csv`, `criterios.json` | `portoes_aceite.png` e `portoes_rejeitada.png` |
| 6.5 · Bland-Altman da estimação absoluta, no teste | `6.5_bland_altman` | `bland_altman_celulas.csv`, `bland_altman.json` | `bland_altman_final.png` |
| 6.6 · Pressão de referência e estimativa absoluta numa sessão do teste | `6.6_serie_sessao` | `serie_sessao.csv`, `calibracao.json` | `bp_serie_sessao_final.png` |
| 6.7 · Dispersão das 20 tentativas do Optuna contra a incerteza do MAE | `6.7_dispersao_trials` | `trials.csv`, `planalto_incerteza.json` | `dispersao_trials_baixa.png` |
| 6.8 · Autocorrelação do resíduo em função do desfasamento | `6.8_acf_residuo` | `acf_residuo.csv` | `acf_residuo_baixa.png` |
| 6.9 · O modelo absoluto em três situações do teste | `6.9_serie_absoluto` | `series.csv` | `serie_absoluto_pequena.png` |
| 6.10 · Erro do modelo e da âncora contra o tempo desde a calibração | `6.10_deriva_calibracao` | `ancora_horizonte.csv`, `ganho_contra_ancora.json`, `variantes.csv` | `figura_deriva_calibracao_pequena.pdf` e `.png` |
| 6.11 · As regras da avaliação por episódio, num troço do teste | `6.11_regras_episodio` | `blocos.csv` | `regras_episodio_tempo_baixa2_final.png` |
| 6.12 · Sensibilidade do classificador e do regressor quantílico contra os minutos de falso alarme | `6.12_classificador_vs_quantilica` | `comparacao_niveis_carga.csv`, `comparacao_niveis_carga_dev.csv` | `classificador_vs_quantilica_tempo_larga.png` |
| 6.13 · AUC no desenvolvimento em função da janela W | `6.13_auc_janela` | `resultados_janela_dev.csv` | `auc_janela_dev_larga.png` |
| 6.14 · Curvas ROC por bloco, no teste | `6.14_roc` | `roc_curvas.csv` | `roc_tempo_larga.png` |
| 6.15 · Escolha conjunta dos dois limiares, no teste | `6.15_conjunta` | `grelha_teste.csv` | `conjunta_tempo_final.png` |
| 6.16 · Avaliação da configuração final, no teste | `6.16_avaliacao_final` | `avaliacao_tendencia.json` | `avaliacao_tendencia_tempo_final.png` |
| 6.17 · Uma sessão do teste: pressão, características e alarmes | `6.17_serie_tendencia` | `blocos.csv` | `serie_tendencia_tempo_final.png` |
| 6.18 · VPP em função da sensibilidade, no teste | `6.18_vpp_sensibilidade` | `vpp_curva.json` | `vpp_sensibilidade_tempo_final.png` |
| 6.19 · As nove características mais importantes | `6.19_importancia` | `importancia_quant_descida.csv`, `importancia_quant_subida.csv` | `importancia_compacta_tempo_final.png` |
| 7.1 · Latência e CPU contra o número de pulseiras por processo | `7.1_teste_carga` | `carga_*.json` (12 ficheiros) | `teste_carga_tempo_final.png` |
| 7.2 · Temperatura do sensor e leituras do termómetro | `7.2_temperatura` | `ensaio_1_sensor.csv`, `ensaio_2_sensor.csv`, `termometro.csv` | `temperatura_ensaio_final.png` |
| 7.3 · FC, SpO₂ e FR do servidor e referências, num ensaio | `7.3_sinais_vitais` | `serie_servidor.csv`, `referencias.csv` | `ensaio_ppg.png` |
| F.1 · Sensibilidade com e sem o portão de estabilidade | `F.1_portao_estabilidade` | `portao_estabilidade.csv` | `portao_tempo_final.png` |

**Não está aqui:** a Figura C.1 (valores SHAP), que traz a idade de cada janela, um dado das tabelas clínicas do MIMIC, que são de acesso credenciado.

## O que tem cada ficheiro de dados

Quase tudo são resultados agregados. As exceções são os sinais e as séries por sessão, que vêm de três origens:
- **Bases de sinais do MIMIC** (abertas, ver a licença no fim): a janela de PPG da Figura 6.4 e as séries de uma sessão do conjunto de teste das Figuras 6.6, 6.9, 6.11 e 6.17. Não levam identificadores de doentes, registos ou sessões, e o tempo é sempre o de dentro do troço.
- **Gravações do dispositivo em voluntários:** as Figuras 6.1 e 6.3, com troços de 1,8 s e de 8 s de sinais já filtrados (e em bruto, na 6.3), e a Figura 7.3, o ensaio de uma pessoa. Sem nomes nem datas.

| Ficheiro | Colunas ou conteúdo |
|---|---|
| `bland_altman_celulas.csv` | O mapa de densidade da 6.5: `x` e `y` são o centro de cada célula (a média da estimativa e da referência, e a diferença, em mmHg) e `contagem` o número de janelas nela. |
| `bland_altman.json` | O viés e os limites de concordância (mmHg), o declive e os extremos da reta de regressão, a forma da célula e os limites dos eixos. |
| `trials.csv` | Uma linha por tentativa do Optuna: `number` e `value` (o MAE em mmHg, em validação cruzada de 3 dobras por doente). |
| `planalto_incerteza.json` | `ic95_largura`: a largura do intervalo de confiança a 95 % do MAE, por doente. |
| `acf_residuo.csv` | `lag_15s` (o desfasamento em blocos de 15 s), `minutos` e `autocorrelacao` do resíduo. |
| `ancora_horizonte.csv` | Por faixa de tempo desde a calibração (`faixa_min`): o número de janelas `n`, o MAE da âncora e o do modelo, e o `ganho`. |
| `ganho_contra_ancora.json` | O ganho global do modelo sobre a âncora e o seu intervalo de confiança a 95 %. |
| `variantes.csv` | Uma linha por variante do modelo: `n_suj` (doentes), `n` (janelas), `idade` (a mediana do tempo desde a calibração, em min), `modelo` e `ancora` (os MAE). |
| `comparacao_niveis_carga.csv` | Teste. Uma linha por modelo (`quant` é o regressor quantílico, `clf` o classificador binário) e por alvo de falso alarme: `alvo_min_h` (4, 6, 8 ou 10 min por hora da união), os limiares `tau_d` e `tau_s`, as sensibilidades `sens_d` (descida) e `sens_s` (subida), `min_falso_h`, `ciclo` e `alarmes_h`. |
| `comparacao_niveis_carga_dev.csv` | O mesmo, no desenvolvimento, com o prefixo `dev_` nas colunas. |
| `resultados_janela_dev.csv` | Uma linha por janela `W` (2, 3, 5 e 10 min) e por direção (`descida` ou `subida`): o `auc_dev`, as sensibilidades e os VPP a 4, 6, 8 e 10 min/h, e os mesmos valores no teste (colunas `teste_`). A figura usa `W`, `direcao` e `auc_dev`. |
| `roc_curvas.csv` | Os pontos de cada curva ROC no teste: `direcao`, `fpr` (1 − especificidade) e `tpr` (sensibilidade). O AUC calcula-se desses pontos. |
| `grelha_teste.csv` | Um par de limiares do teste por linha (`tau_d`, `tau_s`), com as sensibilidades `sens_d` e `sens_s`, o tempo de falso alarme da união `u_FA_min_h` (min por hora) e o ciclo de serviço `u_duty`. A figura usa os pares com `u_duty` até 0,30 e o ponto adotado (`tau_d` = 11,34, `tau_s` = 10,92). |
| `avaliacao_tendencia.json` | A avaliação final no teste: o `ponto` (sensibilidades em fração, falso alarme em min/h, VPP em %), os intervalos de confiança a 95 % (`ic95`, por bootstrap sobre os doentes) e a sensibilidade do alarme aleatório com a mesma estrutura (`alarme_aleatorio`). |
| `vpp_curva.json` | `curvas`: por direção, os pares [sensibilidade, VPP] (em fração) da curva; `adotado`: o ponto adotado de cada direção. |
| `importancia_quant_descida.csv`, `importancia_quant_subida.csv` | A importância (ganho total normalizado) de cada uma das 43 características, uma linha por característica. |
| `carga_<modelo>.json` | O teste de carga: `xgb` é o XGBoost adotado, `bilstm` e `bitcn` são os dois híbridos (BiLSTM + XGBoost e BiTCN + XGBoost). Sem sufixo é a corrida grossa (10 a 800 pulseiras), e `_rep1` a `_rep3` são as três repetições (300 a 500). Cada nível tem a latência, o CPU e a memória. |
| `ensaio_1_sensor.csv`, `ensaio_2_sensor.csv` | `time_ms` (o relógio da pulseira, em ms) e `Temp` (a temperatura do sensor MLX90614, em °C, uma leitura a cada 0,5 s). |
| `termometro.csv` | As duas leituras do termómetro de cada ensaio: `ensaio` (1 ou 2), `momento` (`inicio` ou `fim`), `tempo_s` e `termometro_C`. |
| `serie_servidor.csv` | A Figura 7.3: os valores de FC, SpO₂ e FR do servidor ao longo do ensaio (`grandeza`, `minutos` desde o início da gravação, `valor`). A SpO₂ não tem os primeiros e os últimos 30 s. |
| `referencias.csv` | As leituras de referência da Figura 7.3: a FC e a SpO₂ da pulseira de atividade, e a contagem manual das respirações (`minutos` é o início do minuto contado). |
| `sinais.csv` (6.1) | O PPG (`ppg`) e o ECG (`ecg`) filtrados, uma linha por amostra (`amostra`, a 100 Hz), e `ppg_savgol`, a suavização do PPG de onde sai a segunda derivada. |
| `pontos.json` (6.1) | A frequência de amostragem e o índice de cada ponto característico: R do batimento e do seguinte, Q, S, T, fim do QT (`qn`), picos sistólicos (`pk`, `pkn`), pé do pulso (`v`), fim da descida (`pe`), pé seguinte (`v2`), pico diastólico (`dmax`) e os pontos a, b, c e d da segunda derivada. |
| `ppg.csv`, `ecg.csv` (6.3) | Um troço de 8 s de cada sinal: `tempo_s`, o sinal em bruto (`ppg_bruto`, `ecg_bruto`, contagens do ADC) e o filtrado (`ppg_filtrado`, `ecg_filtrado`). |
| `janela_aceite.csv`, `janela_rejeitada.csv` (6.4) | O PPG filtrado de uma janela de 15 s (125 Hz): `tempo_s`, `ppg`, `pico_sistolico` (1 nos picos detetados) e `pico_retido` (1 nos que passam os critérios de batimento). |
| `criterios.json` (6.4) | Os `limites` dos portões e, por janela, o `veredito`, o valor de cada critério, a `mediana`, a `amplitude_do_pulso` e a amostra do desvio máximo. |
| `serie_sessao.csv`, `calibracao.json` (6.6) | Um bloco de 15 s por linha: `tempo_min` desde a calibração, `sbp_referencia` e `sbp_estimada` (mmHg), e o valor da calibração. |
| `series.csv` (6.9) | Os três painéis: `painel` (a, b ou c), `sessao`, `tempo_min`, `sbp_referencia`, `sbp_estimada` e `sbp_calibracao` (mmHg). |
| `blocos.csv` (6.11) | 120 blocos de 15 s: `sbp_referencia`, `evento_descida`, `evento_subida` (variação de pelo menos 15 mmHg em 5 min) e `alarme_descida`, `alarme_subida` (pontuação acima do limiar). |
| `blocos.csv` (6.17) | Os blocos de uma sessão: `tempo_min`, `sbp_referencia`, a variação a 5 min das três características mais importantes, `score_descida`, `score_subida` (mmHg) e os eventos de cada direção. |
| `portao_estabilidade.csv` | `direcao`, `serie` (`sem_portao` ou `com_portao`), `falso_alarme_min_por_h` (4 a 10) e `sensibilidade` (por episódio). |

## Origem dos dados e licença

Contém informação da **MIMIC-III Waveform Database Matched Subset** (<https://physionet.org/content/mimic3wdb-matched/1.0/>) e da **MIMIC-IV Waveform Database**
(<https://physionet.org/content/mimic4wdb/0.1.0/>), disponibilizadas pelo PhysioNet sob a Open Database License (ODbL) v1.0
(<https://opendatacommons.org/licenses/odbl/1-0/>). Os ficheiros desta pasta que derivam dessas bases ficam disponíveis sob a mesma licença.
Não há aqui dados das tabelas clínicas do MIMIC, que são de acesso credenciado.
