"""Verifica quantos eventos do holdout (e da intersecao PPG&ECG) vem de cada sujeito.
Reutiliza o carregamento de 15_ablacao_ecg_causal.py, correndo a partir da pasta original.
Saida: concentracao_eventos_por_sujeito.json / .txt"""
import importlib.util, json, os, sys
RAIZ = "/home/filipe/Desktop/Tese/algoritmo-ml/2026-09-09_normalizacao_causal_tendencia"
os.chdir(RAIZ)
spec = importlib.util.spec_from_file_location("abl15", f"{RAIZ}/15_ablacao_ecg_causal.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
hold_p, fp = m.carregar_lado(m.HOLD_PPG, m.FEAT_PPG, m.SLOPE_PPG, 'hr_mean_z', 'hr_mean_z', 'hold PPG')
hold_e, fe = m.carregar_lado(m.HOLD_ECG, m.FEAT_ECG, m.SLOPE_ECG, 'pat_mean_z', 'pat_mean_z', 'hold ECG')
W = int(m.W_MIN * 60 / m.BLOCO_SEG)
d = hold_p['sbp_target'] - hold_p.groupby('session_id')['sbp_target'].shift(W)
hold_p['y_descida'] = (d <= -m.D).astype(float); hold_p['y_subida'] = (d >= m.D).astype(float)
out = {}
for dr in ('descida', 'subida'):
    al = f'y_{dr}'
    P = hold_p[m.CHAVE + fp + [al]].dropna(subset=fp + [al]).reset_index(drop=True)
    I = P.merge(hold_e[m.CHAVE + fe], on=m.CHAVE, how='inner').dropna(subset=fe).reset_index(drop=True)
    r = {}
    for nome, df in (('holdout_completo', P), ('intersecao_ecg', I)):
        ev = df.groupby('subject_id')[al].sum().sort_values(ascending=False)
        tot = float(ev.sum())
        r[nome] = dict(n_blocos=len(df), n_eventos=int(tot), n_sujeitos=int(df.subject_id.nunique()),
                       n_sujeitos_com_evento=int((ev > 0).sum()),
                       top3_eventos=[int(x) for x in ev.head(3)], top3_fracao=float(ev.head(3).sum() / tot) if tot else None,
                       top1_fracao=float(ev.iloc[0] / tot) if tot else None)
    out[dr] = r
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "concentracao_eventos_por_sujeito.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
