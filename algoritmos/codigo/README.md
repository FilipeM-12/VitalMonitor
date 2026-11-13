# algoritmos/codigo · os scripts que produzem os resultados da dissertação

As pastas estão agrupadas por tema, pela ordem do caminho que vai dos dados do MIMIC ao modelo instalado no servidor. Cada
pasta tem um `README.md` com o que faz cada script. Os dados do MIMIC não estão no repositório: os scripts leem-nos dos
caminhos da máquina de desenvolvimento.

| Grupo | O que é | Na dissertação |
|---|---|---|
| [`1_dados_e_caracteristicas/`](1_dados_e_caracteristicas/) | Do MIMIC às tabelas de treino: seleção e partição dos doentes, extração das características do absoluto e da tendência, tabela única da tendência e caracterização dos conjuntos de dados. | Secção 6.2, Apêndice B |
| [`2_estimacao_absoluta/`](2_estimacao_absoluta/) | O modelo da pressão absoluta (XGBoost, 27 características): experiências, tabelas, âncora, SHAP, deriva, Bland-Altman, comparação com a literatura e com o estudo de Finnegan. | Secção 6.3 |
| [`3_tendencia_modelo_final/`](3_tendencia_modelo_final/) | **A cadeia final da deteção de tendência**, a do modelo instalado no servidor: 43 características, janela de 5 min ancorada no tempo, afinação, resultados, valor do ECG, cascata, verificação do servidor e teste de carga. | Secções 6.4 e 6.5, Cap. 7, Apêndices C.5 a C.8, D, F e G |
| [`4_tendencia_selecao_e_experiencias/`](4_tendencia_selecao_e_experiencias/) | Como se escolheu o modelo (híbridos, redes puras, arquiteturas) e as experiências à volta. | Cap. 6, Apêndices C e F.2 |
| [`5_servidor_e_carga/`](5_servidor_e_carga/) | Verificação da extração e da derivação do servidor contra o treino, capturas do painel e o modelo relacional da base de dados. | Cap. 7 |
| [`6_linhas_de_base_e_sensibilidade_da_tendencia/`](6_linhas_de_base_e_sensibilidade_da_tendencia/) | O alarme aleatório com que se compara o detetor, a sensibilidade às convenções do episódio, o ciclo de serviço da união isolada e a verificação de constantes do texto. | Secção 6.4 |

* Não estão aqui os scripts que só escrevem números e texto nos ficheiros da dissertação.
* Alguns cabeçalhos citam, como origem de uma tabela ou de previsões, scripts da cadeia de desenvolvimento anterior que não estão neste repositório. Os resultados guardados que esses scripts produziram, e de que os de agora dependem, estão nas respetivas pastas.
