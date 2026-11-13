"""Janela por tempo: lê as previsões do teste desta pasta (uma semente, colunas score_<modelo>); cada sessão é posta na grade de 15 s (blocos sem
sequência = sem score e sem evento), para que o intervalo entre alarmes e o crédito de aviso corram em tempo real. Uso:
  python3 sensibilidade_tempo.py classificador|quantilica

Sensibilidade por episódio a custo igualado, XGBoost contra híbridos, na
regressão quantílica (holdout, 44 sujeitos, 3 sementes).

Responde à pergunta que o AUC não responde: o ganho de 0,01 a 0,015 de AUC dos
híbridos quantílicos traduz-se em sensibilidade a custo igualado, que é a
métrica com que a tese escolhe o ponto de operação?

Método (funções `runs`, `avaliar`, `fronteira` e `interp_sens`, reproduzidas aqui sem
alterações de lógica):
  - um alarme é um troço de blocos com score >= limiar, fundido até 1 min
  - um episódio conta como detetado se algum alarme lhe sobrepuser, com crédito
    de aviso prévio de 2 min
  - falso alarme em min/h = tempo de alarmes sem episódio / horas monitorizadas
  - para cada modelo varre-se o limiar, mantém-se a fronteira (sensibilidade
    crescente com o custo, ciclo de serviço <= 30%) e interpola-se a 4 a 10 min/h

Diferenças face ao ponto de operação da tese, ditas no texto:
  - SEM portão de estabilidade (o portão afina-se com previsões fora de partição
    no desenvolvimento, que não existem para os híbridos), por isso compara-se o
    score e não o sistema completo
  - os três modelos são avaliados nas MESMAS linhas, que são as das sequências de
    20 blocos, e não nas do regressor adotado, daí o valor do XGBoost ser um pouco
    diferente do da Tabela de comparação entre formulações

Cada semente é UM modelo. O híbrido da semente s compara-se com o XGBoost da
semente s. O intervalo de confiança vem de 1000 reamostragens de SUJEITOS.

Entrada: holdout_preds_hibrido_quantilico_sementes_{direcao}.csv.gz
Saída:   sensibilidade_custo_igualado_quantilica.{csv,json} (nesta pasta)
"""
import json

import numpy as np
import pandas as pd

import sys
import base_hib
FORM = sys.argv[1] if len(sys.argv) > 1 else "classificador"
assert FORM in ("classificador", "quantilica", "redes_quantilica")
SAIDA = base_hib.OUT

BLOCO_SEG = 15
GAP_MERGE = int(1 * 60 / BLOCO_SEG)
PRE_CREDITO = int(2 * 60 / BLOCO_SEG)
DUTY_MAX = 0.30
ALVOS_FA = np.array([4, 5, 6, 7, 8, 9, 10])
SEMENTES, MODELOS = (0,), ('XGB', 'BiLSTM', 'BiTCN')
N_LIMIARES, B = 150, 1000


def runs(mask, gap=GAP_MERGE):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    cortes = np.where(np.diff(idx) > gap + 1)[0]
    return [(g[0], g[-1]) for g in np.split(idx, cortes + 1)]


def estatisticas_sessao(y, alarme, eventos):
    """Contagens de uma sessão: detetados, eventos, min de falso alarme, blocos em alarme, blocos."""
    al = runs(alarme)
    det = sum(any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (a0, a1) in al) for (e0, e1) in eventos)
    fa_min = sum((a1 - a0 + 1) * BLOCO_SEG / 60.0 for (a0, a1) in al
                 if not any(a1 >= (e0 - PRE_CREDITO) and a0 <= e1 for (e0, e1) in eventos))
    em_alarme = sum(a1 - a0 + 1 for (a0, a1) in al)
    return det, len(eventos), fa_min, em_alarme, len(y)


def fronteira_interp(soma):
    """soma: (T, 5) = det, n_ev, fa_min, em_alarme, blocos. Devolve a sensibilidade a ALVOS_FA."""
    sens = soma[:, 0] / np.maximum(soma[:, 1], 1)
    fa = soma[:, 2] / (soma[:, 4] * BLOCO_SEG / 3600.0)
    duty = soma[:, 3] / soma[:, 4]
    ok = duty <= DUTY_MAX
    if ok.sum() < 2:
        return np.full(len(ALVOS_FA), np.nan)
    s, f = sens[ok], fa[ok]
    ordem = np.argsort(f, kind='stable')
    s, f = s[ordem], f[ordem]
    melhor = np.maximum.accumulate(s)
    mantem = np.concatenate([[True], s[1:] > melhor[:-1]])
    s, f = s[mantem], f[mantem]
    if len(s) < 2:
        return np.full(len(ALVOS_FA), np.nan)
    return np.interp(ALVOS_FA, f, s, left=np.nan, right=s[-1])


def principal():
    rng = np.random.default_rng(42)
    linhas, resumo = [], {}
    for direcao in ('descida', 'subida'):
        p = pd.read_csv(f"{SAIDA}/holdout_preds_" + ("rede_quantilica" if FORM == "redes_quantilica" else "hibrido_quantilico" if FORM == "quantilica" else f"hibrido_{FORM}") + f"_{direcao}.csv.gz")
        p = p.sort_values(['session_id', 'bloco_id']).reset_index(drop=True)
        partes = []                                    # grade de 15 s por sessão: blocos em falta = score NaN e rótulo 0
        for sid, g in p.groupby('session_id', sort=False):
            gr = g.set_index('bloco_id').reindex(np.arange(g.bloco_id.min(), g.bloco_id.max() + 1))
            gr['session_id'] = sid; gr['subject_id'] = g.subject_id.iloc[0]; gr['y'] = gr['y'].fillna(0)
            partes.append(gr.rename_axis('bloco_id').reset_index())
        p = pd.concat(partes, ignore_index=True)
        sessoes = [(g.subject_id.iloc[0], g.y.values.astype(bool), g) for _, g in p.groupby('session_id')]
        sujeitos = sorted({s for s, _, _ in sessoes})
        pos = {s: i for i, s in enumerate(sujeitos)}
        print(f"[{direcao}] {len(sessoes)} sessões, {len(sujeitos)} sujeitos, {len(p)} blocos", flush=True)

        # contagens por (sujeito, limiar) para cada modelo e semente
        agreg = {}
        for m in MODELOS:
            for s in SEMENTES:
                col = f"score_{m}" if FORM else f"score_{m}_s{s}"
                lim = np.nanpercentile(p[col].values, np.linspace(55, 99.95, N_LIMIARES))
                por_suj = np.zeros((len(sujeitos), N_LIMIARES, 5))
                for sid, y, g in sessoes:
                    sc = np.nan_to_num(g[col].values, nan=-np.inf)
                    ev = runs(y)
                    for t, tau in enumerate(lim):
                        por_suj[pos[sid], t] += estatisticas_sessao(y, sc >= tau, ev)
                agreg[(m, s)] = por_suj
            print(f"  {m} feito", flush=True)

        n = len(sujeitos)
        pesos = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)])
        for s in SEMENTES:
            ref_pt = fronteira_interp(agreg[('XGB', s)].sum(0))
            ref_bt = np.stack([fronteira_interp(np.tensordot(w, agreg[('XGB', s)], axes=(0, 0))) for w in pesos])
            for m in ('BiLSTM', 'BiTCN'):
                pt = fronteira_interp(agreg[(m, s)].sum(0))
                bt = np.stack([fronteira_interp(np.tensordot(w, agreg[(m, s)], axes=(0, 0))) for w in pesos])
                ganho_pt = pt - ref_pt
                ganho_bt = bt - ref_bt
                media_pt = float(np.nanmean(ganho_pt))
                media_bt = np.nanmean(ganho_bt, axis=1)
                ic = [float(np.nanpercentile(media_bt, 2.5)), float(np.nanpercentile(media_bt, 97.5))]
                linhas.append(dict(direcao=direcao, semente=s, modelo=f'{m}+XGB',
                                   sens_xgb_media=round(float(np.nanmean(ref_pt)), 4),
                                   sens_hibrido_media=round(float(np.nanmean(pt)), 4),
                                   ganho_medio=round(media_pt, 4), ic95_inf=round(ic[0], 4), ic95_sup=round(ic[1], 4),
                                   exclui_zero=bool(ic[0] > 0 or ic[1] < 0),
                                   **{f'ganho_{a}minh': round(float(g), 4) for a, g in zip(ALVOS_FA, ganho_pt)}))
                print(linhas[-1], flush=True)
        resumo[direcao] = dict(n_sessoes=len(sessoes), n_sujeitos=n, n_blocos=int(len(p)))
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(f"{SAIDA}/sensibilidade_custo_igualado_{FORM}.csv", index=False)
    json.dump(dict(resumo=resumo, alvos_min_h=ALVOS_FA.tolist(), bootstrap=B, limiares=N_LIMIARES),
              open(f"{SAIDA}/sensibilidade_custo_igualado_{FORM}.json", "w"), indent=2)
    print(tabela[['direcao', 'semente', 'modelo', 'sens_xgb_media', 'sens_hibrido_media', 'ganho_medio',
                  'ic95_inf', 'ic95_sup', 'exclui_zero']].to_string(index=False))


if __name__ == "__main__":
    principal()
