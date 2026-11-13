"""Busca de hiperparâmetros (Optuna) para qualquer família e variante, com o mesmo objetivo em todas: o MAE do desvio
na validação cruzada por sujeito (N_FOLDS dobras) sobre o desenvolvimento. Cada trial corre num subprocesso, que lê os
dados da cache (assim a memória do TensorFlow e do XGBoost é libertada no fim). O estudo é guardado em sqlite e retoma-se
de onde ficou: correr outra vez com mais tentativas continua o mesmo estudo.
"""
import json
import multiprocessing as mp
import os
import queue
import time

import optuna

from . import config as C


# ── espaços de procura ───────────────────────────────────────────────────────────────────────────────────────────
def _xgb(trial, prefixo=""):
    s = lambda nome: prefixo + nome
    return dict(
        n_estimators=trial.suggest_int(s('n_estimators'), 300, 1200, step=100),
        learning_rate=trial.suggest_float(s('learning_rate'), 0.005, 0.08, log=True),
        max_depth=trial.suggest_int(s('max_depth'), 4, 10),
        subsample=trial.suggest_float(s('subsample'), 0.6, 1.0),
        colsample_bytree=trial.suggest_float(s('colsample_bytree'), 0.6, 1.0),
        reg_alpha=trial.suggest_float(s('reg_alpha'), 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float(s('reg_lambda'), 1e-3, 10.0, log=True))


def _rede(trial, rede, prefixo=""):
    s = lambda nome: prefixo + nome
    if rede == "BILSTM":
        return dict(learning_rate=trial.suggest_float(s('learning_rate'), 1e-4, 5e-3, log=True),
                    lstm_units_l1=trial.suggest_int(s('lstm_units_l1'), 32, 128, step=32),
                    lstm_units_l2=trial.suggest_int(s('lstm_units_l2'), 16, 64, step=16),
                    dropout_rate=trial.suggest_float(s('dropout_rate'), 0.1, 0.45))
    return dict(learning_rate=trial.suggest_float(s('learning_rate'), 1e-4, 3e-3, log=True),
                nb_filters=trial.suggest_int(s('nb_filters'), 16, 128, step=16),
                kernel_size=trial.suggest_int(s('kernel_size'), 3, 7),
                dropout_rate=trial.suggest_float(s('dropout_rate'), 0.1, 0.45))


def propor(trial, familia):
    """Hiperparâmetros de um trial. Nos híbridos a rede e o XGBoost levam os prefixos rede_ e xgb_."""
    if familia == "XGB":
        return _xgb(trial)
    if familia in ("BILSTM", "BITCN"):
        return _rede(trial, familia)
    rede = familia.split("_")[1]
    # Os NOMES dos parâmetros no estudo também levam o prefixo (rede_, xgb_), para o melhor trial sair do Optuna com os prefixos que
    # separar_params_hibrido procura, e para a taxa de aprendizagem da rede e a do XGBoost terem nomes diferentes.
    return {**{f"rede_{k}": v for k, v in _rede(trial, rede, "rede_").items()}, **{f"xgb_{k}": v for k, v in _xgb(trial, "xgb_").items()}}


# ── trial num subprocesso ────────────────────────────────────────────────────────────────────────────────────────
def _trabalhador(q, familia, params, variante, ancoras, features, metodo):
    try:
        from .avaliacao import validacao_cruzada
        from .dados import carregar
        dev, _, _ = carregar(variante, ancoras, features=features)
        media, _ = validacao_cruzada(familia, params, dev, features, metodo)
        q.put(media)
    except Exception:
        import traceback
        traceback.print_exc()
        q.put(C.VALOR_FALHA)


def avaliar_trial(familia, params, variante, ancoras, features, metodo=C.SUAVIZACAO):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=_trabalhador, args=(q, familia, params, variante, ancoras, tuple(features), metodo))
    p.start()
    inicio, valor = time.time(), C.VALOR_FALHA
    while True:
        try:
            valor = q.get(timeout=5)
            break
        except queue.Empty:
            if not p.is_alive():                                   # morreu sem resultado (memória, erro): falha já
                try:
                    valor = q.get(timeout=2)
                except queue.Empty:
                    pass
                break
            if time.time() - inicio > C.TIMEOUT_TRIAL_S:
                break
    if p.is_alive():
        p.terminate()
    p.join()
    return valor


def primeiras_completas(estudo, n):
    """As primeiras `n` tentativas completas, por número. Assim o 'melhor de 20' é mesmo o melhor de 20, mesmo que o
    estudo tenha ficado com mais (trials em paralelo) ou que se estenda depois a mais tentativas."""
    completas = sorted((t for t in estudo.trials if t.state == optuna.trial.TrialState.COMPLETE), key=lambda t: t.number)
    return completas[:n]


def procurar(familia, variante="baseline", ancoras="causal", features=None, n_trials=C.N_TRIALS,
             metodo=C.SUAVIZACAO, trials_em_paralelo=1, nome=None):
    """Corre (ou retoma) o estudo e devolve (melhores hiperparâmetros, tentativas consideradas). Grava best_params.json.
    O estudo é retomável: chamar outra vez com mais tentativas continua o mesmo estudo."""
    var = C.VARIANTES[variante]
    features = tuple(features if features is not None else var.features)
    nome = nome or f"{familia}_{variante}" + ("" if ancoras == "causal" else f"_{ancoras}")
    os.makedirs(C.DIR_OPTUNA, exist_ok=True)
    estudo = optuna.create_study(direction="minimize", storage=f"sqlite:///{C.DIR_OPTUNA}/{nome}.db",
                                 study_name=nome, load_if_exists=True)
    total = int(os.environ.get('NUCLEOS_MAX', os.cpu_count() or 1))                    # núcleos desta fila
    os.environ['HILOS_XGB'] = str(max(1, total // trials_em_paralelo))                    # núcleos por trial

    def objetivo(trial):
        params = propor(trial, familia)
        valor = avaliar_trial(familia, params, variante, ancoras, features, metodo)
        print(f"[{nome}] trial {trial.number}: MAE CV = {valor:.4f}", flush=True)
        return valor

    feitos = len(primeiras_completas(estudo, n_trials))
    if n_trials > feitos:
        print(f"[{nome}] {feitos}/{n_trials} trials feitos, a correr mais {n_trials - feitos}", flush=True)
        estudo.optimize(objetivo, n_trials=n_trials - feitos, n_jobs=trials_em_paralelo)
    consideradas = primeiras_completas(estudo, n_trials)
    melhor = min(consideradas, key=lambda t: t.value)
    falhas = sum(t.value >= C.VALOR_FALHA for t in consideradas)
    # a pesquisa de todas as famílias é feita com estas características (as redes foram reafinadas com as 27)
    mae_cv = melhor.value
    print(f"[{nome}] melhor MAE CV = {mae_cv:.4f} (entre {len(consideradas)} tentativas, {len(features)} características), "
          f"falhadas = {falhas}", flush=True)
    with open(f"{C.DIR_OPTUNA}/{nome}.best_params.json", "w") as f:
        json.dump(dict(best_params=melhor.params, mae_cv=mae_cv,
                       mae_cv_pesquisa=melhor.value, pesquisa_com=len(features),
                       n_trials=len(consideradas), falhas=falhas, n_features=len(features)), f, indent=1)
    return melhor.params, consideradas


def melhores_parametros(familia, variante="baseline", ancoras="causal", nome=None):
    """Lê os hiperparâmetros de um estudo já feito (nunca inicia uma busca)."""
    nome = nome or f"{familia}_{variante}" + ("" if ancoras == "causal" else f"_{ancoras}")
    with open(f"{C.DIR_OPTUNA}/{nome}.best_params.json") as f:
        return json.load(f)["best_params"]
