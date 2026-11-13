"""Avaliação do absoluto: do desvio previsto à pressão, validação cruzada, holdout, bootstrap por sujeito e graus BHS.

Um só pós-processamento para tudo: previsão -> subtrai a média das 3 primeiras previsões da sessão (o modelo prevê
desvios, o primeiro minuto fixa o zero) -> suavização causal -> PA = sbp_calib + desvio.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from . import config as C
from .modelos import ajustar_e_prever
from .suavizacao import suavizar_por_sessao


def reconstruir(teste, previsoes, metodo=C.SUAVIZACAO):
    """Cópia de `teste` só com as linhas previstas e as colunas preds_delta e preds_final.

    As linhas SEM previsão (as primeiras de cada sessão nas redes) ficam de fora, porque o Kalman propagaria o estado por cima delas e devolveria o primeiro valor seguinte, e assim nenhuma é avaliada com informação do futuro."""
    d = teste.copy()
    d['preds_delta'] = np.nan
    for _, g in d.groupby('session_id'):
        idx = g.index.intersection(previsoes.index)
        if len(idx):
            d.loc[idx, 'preds_delta'] = previsoes.loc[idx] - previsoes.loc[idx[:3]].mean()
    com_previsao = d['preds_delta'].notna()
    d['preds_delta'] = suavizar_por_sessao(d, 'preds_delta', metodo)
    d = d[com_previsao].copy()
    d['preds_final'] = d['sbp_calib'] + d['preds_delta']
    return d


def metricas(d):
    if len(d) == 0:
        return dict(mae_delta=np.nan, mae_sbp=np.nan, n=0, n_sujeitos=0)
    return dict(mae_delta=float(mean_absolute_error(d[C.ALVO], d['preds_delta'])),
                mae_sbp=float(mean_absolute_error(d['sbp_target'], d['preds_final'])),
                n=int(len(d)), n_sujeitos=int(d['subject_id'].nunique()))


def avaliar_holdout(familia, params, dev, teste, features, metodo=C.SUAVIZACAO, semente=C.SEMENTE):
    """Treina em todo o desenvolvimento e mede UMA vez no holdout. Devolve (linhas previstas, métricas)."""
    d = reconstruir(teste, ajustar_e_prever(familia, params, dev, teste, features, semente), metodo)
    return d, metricas(d)


def validacao_cruzada(familia, params, dev, features, metodo=C.SUAVIZACAO, semente=C.SEMENTE):
    """MAE do desvio na validação cruzada por sujeito (N_FOLDS dobras) só no desenvolvimento.
    Devolve (média das dobras, lista por dobra)."""
    maes = []
    for tr, va in GroupKFold(n_splits=C.N_FOLDS).split(dev, dev[C.ALVO], dev['subject_id']):
        treino, validacao = dev.iloc[tr].copy(), dev.iloc[va].copy()
        d = reconstruir(validacao, ajustar_e_prever(familia, params, treino, validacao, features, semente, parar_com_teste=True), metodo)
        maes.append(metricas(d)['mae_delta'])
    return float(np.mean(maes)), maes


# ── bootstrap ao nível do sujeito ────────────────────────────────────────────────────────────────────────────────
def bootstrap_sujeitos(erros, sujeito, pares=None, n_boot=1000, semente=C.SEMENTE):
    """`erros`: {modelo: Series de |erro| por linha}; `sujeito`: Series linha -> sujeito. Reamostra SUJEITOS.
    Devolve o MAE e o IC95% de cada modelo e, para `pares` = {modelo: referência}, a diferença emparelhada
    (modelo - referência; negativo = o modelo é melhor). Só usa as linhas que todos os modelos têm."""
    comum = None
    for e in erros.values():
        comum = e.index if comum is None else comum.intersection(e.index)
    tab = pd.DataFrame({k: v.loc[comum] for k, v in erros.items()})
    tab['sujeito'] = sujeito.loc[comum].values
    g = tab.groupby('sujeito')
    soma, contagem = g.sum(), g.size().values
    rng = np.random.default_rng(semente)
    S = len(soma)
    mae = {k: [] for k in erros}
    dif = {k: [] for k in (pares or {})}
    for _ in range(n_boot):
        i = rng.integers(0, S, S)
        total = contagem[i].sum()
        m = {k: soma[k].values[i].sum() / total for k in erros}
        for k in erros:
            mae[k].append(m[k])
        for k, ref in (pares or {}).items():
            dif[k].append(m[k] - m[ref])
    ic = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(
        n_linhas=int(len(tab)), n_sujeitos=int(S),
        mae={k: dict(mae=float(tab[k].mean()), ic95=ic(mae[k])) for k in erros},
        diferenca={k: dict(contra=pares[k], diferenca_mae=float(tab[k].mean() - tab[pares[k]].mean()), ic95=ic(dif[k]),
                           p_melhor=float((np.array(dif[k]) < 0).mean())) for k in dif})


# ── graus da British Hypertension Society ────────────────────────────────────────────────────────────────────────
def grau_bhs(p5, p10, p15):
    for grau, (a, b, c) in (("A", (60, 85, 95)), ("B", (50, 75, 90)), ("C", (40, 65, 85))):
        if p5 >= a and p10 >= b and p15 >= c:
            return grau
    return "D"


def bhs(d):
    """Percentagens de |erro| <= 5, 10 e 15 mmHg, por janela e média por sujeito, e o grau de cada uma."""
    e = (d['preds_final'] - d['sbp_target']).abs()
    por_janela = [100 * float((e <= t).mean()) for t in (5, 10, 15)]
    por_sujeito = e.groupby(d['subject_id']).apply(lambda s: pd.Series([100 * (s <= t).mean() for t in (5, 10, 15)])).unstack().mean()
    return dict(n_janelas=int(len(d)), n_sujeitos=int(d['subject_id'].nunique()), mae=float(e.mean()),
                p5_janela=por_janela[0], p10_janela=por_janela[1], p15_janela=por_janela[2], grau_janela=grau_bhs(*por_janela),
                p5_sujeito=float(por_sujeito.iloc[0]), p10_sujeito=float(por_sujeito.iloc[1]), p15_sujeito=float(por_sujeito.iloc[2]),
                grau_sujeito=grau_bhs(*por_sujeito.tolist()))
