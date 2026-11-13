# 3_tendencia_modelo_final/tendencia_tempo_fixo/hibridos

Scripts desta pasta (o resumo é o início do cabeçalho de cada um).

| Ficheiro | O que faz |
|---|---|
| [`base_hib.py`](base_hib.py) | Base comum dos híbridos com a janela de 5 min ancorada no TEMPO: constantes, camadas e construtores iguais aos de 4_tendencia_selecao_e_experiencias/comparacao_de_arquiteturas_e_hibridos/comparacao_arquiteturas_classificador.py. |
| [`cv3_tempo.py`](cv3_tempo.py) | Janela de 5 min ancorada no tempo e 43 características. |
| [`hibrido_classificador_tempo.py`](hibrido_classificador_tempo.py) | Janela de 5 min ancorada no tempo e as características novas. Dados e sequências de hibrido_quantilico_tempo.preparar (CONJUNTO=tudo: 43 tabulares e 11 canais por instante). |
| [`hibrido_quantilico_tempo.py`](hibrido_quantilico_tempo.py) | Janela de 5 min ancorada no tempo (ver base_hib.py e tempo_fixo.py). Só se usam as sequências com os 20 blocos presentes. |
| [`perda_sequencia.py`](perda_sequencia.py) | Quantas linhas com alvo ficam sem previsão nas redes por lhes faltar um bloco na sequência de 20 (5 min). |
| [`rede_pura_quantilica_tempo.py`](rede_pura_quantilica_tempo.py) | As redes SOZINHAS (BiLSTM e BiTCN) como regressores quantílicos, com a janela de 5 min por tempo. |
| [`sementes_hibrido_quantilico_tempo.py`](sementes_hibrido_quantilico_tempo.py) | Versão com a janela por tempo. Sementes adicionais dos híbridos quantílicos, para saber se o ganho sobre o XGBoost resiste à aleatoriedade do treino. |
| [`sensibilidade_tempo.py`](sensibilidade_tempo.py) | Janela por tempo: lê as previsões do teste desta pasta (uma semente, colunas score_<modelo>); cada sessão é posta na grade de 15 s (blocos sem sequência = sem score e sem evento). |

Subpastas: [`resultados_tudo/`](resultados_tudo/).
