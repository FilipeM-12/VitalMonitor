#!/usr/bin/env python3
"""Diagrama entidade-relação (modelo relacional) da base de dados do servidor, em TikZ, para o Capítulo 4 (figura abaixo do «Modelo de dados»).

Lê o esquema real do PostgreSQL do servidor (information_schema, só leitura): tabelas, colunas e chave primária. As colunas *_id que
apontam para outra tabela são desenhadas como chave estrangeira (FK). Na base só `pulseira.paciente_id` tem a restrição declarada,
as restantes ligações são garantidas pela aplicação (ver texto). As linhas são ortogonais; os troncos verticais e as portas nas
tabelas-pai são escolhidos por pesquisa exaustiva para minimizar os cruzamentos.

Uso: python gerar_modelo_relacional.py            escreve modelo_relacional.tikz ao lado do script e imprime o resumo
     python gerar_modelo_relacional.py --aplicar  insere a figura no Capítulo 4 do PT rascunho (logo a seguir à «Modelo de dados»)
"""
import itertools
import os
import sys

from sqlalchemy import create_engine, inspect

AQUI = os.path.dirname(os.path.abspath(__file__))
NOTACAO = "pe-de-galinha" if "--pe-de-galinha" in sys.argv else "numeros"      # por omissão: 1, N e 0..1 em cada ponta
URL = os.environ.get("DATABASE_URL", "postgresql://hospital:hospital_pass@localhost:5432/hospital")
LINHA, CAB, LARG = 0.43, 0.6, 4.2          # altura da linha, do cabeçalho e largura das tabelas (cm)
PAIS = ("pulseira", "paciente")
FILHOS_ESQ = ["atribuicao", "medicao", "medicao_tendencia"]
FILHOS_DIR = ["calibracao", "medicao_absoluta"]
SOLTA = "configuracoes"
GAP_ESQ, GAP_DIR, GAP_V = 2.9, 2.3, 0.75

# ── esquema real ──
insp = inspect(create_engine(URL))
T = {}
for t in insp.get_table_names():
    pk = set(insp.get_pk_constraint(t)["constrained_columns"])
    T[t] = [(c["name"], "PK" if c["name"] in pk else ("FK" if c["name"].endswith("_id") else "")) for c in insp.get_columns(t)]
assert set(T) == set(PAIS) | set(FILHOS_ESQ) | set(FILHOS_DIR) | {SOLTA}, sorted(T)
for f in FILHOS_ESQ + FILHOS_DIR:
    assert {"pulseira_id", "paciente_id"} <= {c for c, _ in T[f]}, f
fk_declaradas = {(t, c["constrained_columns"][0], c["referred_table"]) for t in T for c in insp.get_foreign_keys(t)}


MIN_LINHAS_PAI = 5          # as tabelas-pai recebem 5 ou 6 ligações: com poucas colunas (a pulseira só tem 2) ficariam sem espaço para elas


def altura(t):
    return CAB + LINHA * max(len(T[t]), MIN_LINHAS_PAI if t in PAIS else 0)


# ── posições (topo-esquerdo de cada tabela) ──
X_C = 0.0
X_E = X_C - GAP_ESQ - LARG
X_D = X_C + LARG + GAP_DIR
pos = {}


def colocar(esq, dir_, off_e, off_d):
    """Preenche `pos` para a ordem dada dos filhos de cada lado e o deslocamento vertical (cm) de cada coluna."""
    pos.clear()
    # a pulseira fica ACIMA do paciente, porque em todas as tabelas filhas a coluna pulseira_id vem antes de paciente_id
    # (assim as duas linhas da mesma tabela não se cruzam)
    pos["pulseira"] = (X_C, 0.0)
    pos["paciente"] = (X_C, -(altura("pulseira") + 1.5))
    pos[SOLTA] = (X_C, pos["paciente"][1] - altura("paciente") - 1.1)
    y = off_e
    for t in esq:
        pos[t] = (X_E, y)
        y -= altura(t) + GAP_V
    y = off_d
    for t in dir_:
        pos[t] = (X_D, y)
        y -= altura(t) + GAP_V


def y_linha(t, col):
    i = [c for c, _ in T[t]].index(col)
    return pos[t][1] - CAB - LINHA * (i + 0.5)


# ── ligações filho -> pai (lado esquerdo e direito) ──
def lado(filhos, esq):
    xs = X_E + LARG if esq else X_D                            # aresta do filho virada para o centro
    xp = X_C if esq else X_C + LARG                            # aresta do pai virada para o filho
    return [(f, p, y_linha(f, p + "_id")) for f in filhos for p in PAIS], xs, xp


def segmentos(linhas, xs, xp, esq):
    seg = []
    for yf, yp, xt in linhas:
        seg += [((xs, yf), (xt, yf)), ((xt, yf), (xt, yp)), ((xt, yp), (xp, yp))]
    return seg


def cruza(a, b):
    (x1, y1), (x2, y2) = a
    (x3, y3), (x4, y4) = b
    h1, h2 = y1 == y2, y3 == y4
    if h1 == h2:
        if h1 and y1 == y3:
            return max(min(x1, x2), min(x3, x4)) < min(max(x1, x2), max(x3, x4))
        if (not h1) and x1 == x3:
            return max(min(y1, y2), min(y3, y4)) < min(max(y1, y2), max(y3, y4))
        return False
    if not h1:
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = (x3, y3), (x4, y4), (x1, y1), (x2, y2)
    return min(x1, x2) < x3 < max(x1, x2) and min(y3, y4) < y1 < max(y3, y4)


def quase(a, b):
    """Dois segmentos de linhas diferentes, paralelos e tão juntos que se confundem no desenho."""
    (x1, y1), (x2, y2) = a
    (x3, y3), (x4, y4) = b
    h1, h2 = y1 == y2, y3 == y4
    if h1 and h2:
        return abs(y1 - y3) < 0.3 and min(max(x1, x2), max(x3, x4)) - max(min(x1, x2), min(x3, x4)) > 0.05
    if (not h1) and (not h2):
        return abs(x1 - x3) < 0.18 and min(max(y1, y2), max(y3, y4)) - max(min(y1, y2), min(y3, y4)) > 0.05
    return False


def otimiza_lado(filhos, esq):
    """Melhor atribuição de troncos para um lado, com as portas dos pais por ordem de altura dos filhos."""
    ls, xs, xp = lado(filhos, esq)
    n = len(ls)
    lo, hi = sorted((xs + 0.6, xp - 0.6)) if esq else sorted((xp + 0.6, xs - 0.6))
    slots = [lo + (hi - lo) * (k + 0.5) / n for k in range(n)]
    portas = {}
    for p in PAIS:
        k = sum(1 for _, pp, _ in ls if pp == p)
        topo, base = pos[p][1] - CAB - 0.2, pos[p][1] - altura(p) + 0.25
        portas[p] = [topo + (base - topo) * (i + 0.5) / k for i in range(k)]
    ordem = {p: iter(sorted([i for i, (_, pp, _) in enumerate(ls) if pp == p], key=lambda i: -ls[i][2])) for p in PAIS}
    yp = [None] * n
    for p in PAIS:
        for porta, i in zip(portas[p], ordem[p]):
            yp[i] = porta
    melhor = None
    for ps in itertools.permutations(range(n)):
        linhas = [(ls[i][2], yp[i], slots[ps[i]]) for i in range(n)]
        seg = [segmentos([l], xs, xp, esq) for l in linhas]
        cr = sum(cruza(a, b) for i in range(n) for j in range(i + 1, n) for a in seg[i] for b in seg[j])
        qs = sum(quase(a, b) for i in range(n) for j in range(i + 1, n) for a in seg[i] for b in seg[j])
        comp = sum(abs(l[1] - l[0]) + abs(l[2] - xs) + abs(xp - l[2]) for l in linhas)
        if melhor is None or (cr, qs, comp) < melhor[0]:
            melhor = ((cr, qs, comp), linhas, ls)
            if cr == 0:
                pass
    return melhor, xs, xp


def refina_portas(linhas, ls, xs, xp, esq):
    """Desloca ligeiramente a altura a que cada linha chega ao pai para que nenhuma fique colada a outra (< 0,3 cm) e sem criar
    cruzamentos novos. Devolve as linhas ajustadas e o número de pares quase sobrepostos antes e depois."""
    deltas = [-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45]
    n = len(linhas)
    cache = {}

    def par(i, j, di, dj):
        k = (i, j, di, dj)
        if k not in cache:
            a = segmentos([(linhas[i][0], linhas[i][1] + di, linhas[i][2])], xs, xp, esq)
            b = segmentos([(linhas[j][0], linhas[j][1] + dj, linhas[j][2])], xs, xp, esq)
            cache[k] = (sum(cruza(u, v) for u in a for v in b), sum(quase(u, v) for u in a for v in b))
        return cache[k]

    def avalia(ds):
        cr = sum(par(i, j, ds[i], ds[j])[0] for i in range(n) for j in range(i + 1, n))
        qs = sum(par(i, j, ds[i], ds[j])[1] for i in range(n) for j in range(i + 1, n))
        return cr, qs

    cr0, qs0 = avalia((0.0,) * n)
    faixa = []
    for (_, p, _), (_, yp, _) in zip(ls, linhas):
        base, topo = pos[p][1] - altura(p) + 0.2, pos[p][1] - CAB - 0.1
        faixa.append([d for d in deltas if base <= yp + d <= topo])
    melhor = None
    for ds in itertools.product(*faixa):
        cr, qs = avalia(ds)
        if cr > cr0:
            continue
        chave = (qs, cr, sum(abs(d) for d in ds))
        if melhor is None or chave < melhor[0]:
            melhor = (chave, ds)
    ds = melhor[1]
    return [(yf, yp + d, xt) for (yf, yp, xt), d in zip(linhas, ds)], qs0, melhor[0][0]


def procura():
    todos = FILHOS_ESQ + FILHOS_DIR
    melhor = None
    for esq in itertools.combinations(todos, 3):
        dir_ = [t for t in todos if t not in esq]
        for oe in itertools.permutations(esq):
            for od in itertools.permutations(dir_):
                for off_e in (0.0, -1.5, -3.0):
                    for off_d in (0.0, -1.5, -3.0, -4.5):
                        colocar(list(oe), list(od), off_e, off_d)
                        (ce, _, _), _, _ = otimiza_lado(list(oe), True)
                        (cd, _, _), _, _ = otimiza_lado(list(od), False)
                        chave = (ce[0] + cd[0], ce[1] + cd[1], ce[2] + cd[2])
                        if melhor is None or chave < melhor[0]:
                            melhor = (chave, list(oe), list(od), off_e, off_d)
    return melhor


# ── TikZ ──
def tabela(t):
    x, top = pos[t]
    h = altura(t)
    o = [f"    \\filldraw[fill=gray!25, draw=black, thick] ({x:.2f},{top:.2f}) rectangle ({x + LARG:.2f},{top - CAB:.2f});",
         f"    \\node[font=\\small\\bfseries] at ({x + LARG / 2:.2f},{top - CAB / 2:.2f}) {{\\texttt{{{t.replace('_', chr(92) + '_')}}}}};",
         f"    \\draw[thick] ({x:.2f},{top - CAB:.2f}) rectangle ({x + LARG:.2f},{top - h:.2f});",
         f"    \\draw[thick] ({x + 0.8:.2f},{top - CAB:.2f}) -- ({x + 0.8:.2f},{top - h:.2f});"]
    for i, (c, k) in enumerate(T[t]):
        yy = top - CAB - LINHA * (i + 0.5)
        if k:
            o.append(f"    \\node[font=\\footnotesize\\bfseries, anchor=west] at ({x + 0.1:.2f},{yy:.2f}) {{{k}}};")
        o.append(f"    \\node[font=\\footnotesize, anchor=west] at ({x + 0.9:.2f},{yy:.2f}) {{\\texttt{{{c.replace('_', chr(92) + '_')}}}}};")
    return o


def pe_de_galinha(x, y, para_a_direita):
    """A linha chega à tabela por x: três dentes abertos para a tabela."""
    d = 0.32 if para_a_direita else -0.32
    return [f"    \\draw[semithick] ({x + d:.2f},{y:.2f}) -- ({x:.2f},{y + 0.17:.2f});",
            f"    \\draw[semithick] ({x + d:.2f},{y:.2f}) -- ({x:.2f},{y - 0.17:.2f});",
            f"    \\draw[semithick] ({x + d:.2f},{y:.2f}) -- ({x:.2f},{y:.2f});"]


def barra(x, y, para_a_direita):
    d = 0.2 if para_a_direita else -0.2
    return [f"    \\draw[semithick] ({x + d:.2f},{y - 0.15:.2f}) -- ({x + d:.2f},{y + 0.15:.2f});"]


(_, ESQ, DIR, OFF_E, OFF_D), = [procura()]
colocar(ESQ, DIR, OFF_E, OFF_D)
tikz = ["\\begin{tikzpicture}[line cap=butt]"]
for t in T:
    tikz += tabela(t)
tot_cr = 0
sobrepostos = []
for filhos, esq in ((ESQ, True), (DIR, False)):
    (custo, linhas, ls), xs, xp = otimiza_lado(filhos, esq)
    tot_cr += custo[0]
    linhas, antes, depois = refina_portas(linhas, ls, xs, xp, esq)
    sobrepostos.append((antes, depois))
    for (yf, yp, xt), (f, p, _) in zip(linhas, ls):
        tikz.append(f"    \\draw[semithick] ({xs:.2f},{yf:.2f}) -- ({xt:.2f},{yf:.2f}) -- ({xt:.2f},{yp:.2f}) -- ({xp:.2f},{yp:.2f});")
        if NOTACAO == "numeros":
            dx = 0.27 if esq else -0.27
            tikz.append(f"    \\node[font=\\scriptsize, inner sep=0.5pt] at ({xs + dx:.2f},{yf + 0.17:.2f}) {{N}};")          # lado «muitos» (filho)
            tikz.append(f"    \\node[font=\\scriptsize, inner sep=0.5pt] at ({xp - dx:.2f},{yp + 0.17:.2f}) {{1}};")          # lado «um» (pai)
        else:
            tikz += pe_de_galinha(xs, yf, esq)                  # a linha parte do filho: lado «muitos»
            tikz += barra(xp, yp, not esq)                      # e chega ao pai: lado «um»
# paciente.id (um) — pulseira.paciente_id (muitos): pulseira em cima, paciente em baixo
xm = X_C + LARG / 2
y_pul, y_pac = pos["pulseira"][1] - altura("pulseira"), pos["paciente"][1]
tikz.append(f"    \\draw[semithick] ({xm:.2f},{y_pul:.2f}) -- ({xm:.2f},{y_pac:.2f});")
if NOTACAO == "numeros":
    # pulseira.paciente_id aceita nulo (pulseira sem doente): do lado do paciente é «0..1»
    tikz.append(f"    \\node[font=\\scriptsize, inner sep=0.5pt, anchor=west] at ({xm + 0.12:.2f},{y_pul - 0.25:.2f}) {{N}};")
    tikz.append(f"    \\node[font=\\scriptsize, inner sep=0.5pt, anchor=west] at ({xm + 0.12:.2f},{y_pac + 0.25:.2f}) {{0..1}};")
else:
    tikz += [f"    \\draw[semithick] ({xm:.2f},{y_pul - 0.32:.2f}) -- ({xm - 0.17:.2f},{y_pul:.2f});",
             f"    \\draw[semithick] ({xm:.2f},{y_pul - 0.32:.2f}) -- ({xm + 0.17:.2f},{y_pul:.2f});",
             f"    \\draw[semithick] ({xm:.2f},{y_pul - 0.32:.2f}) -- ({xm:.2f},{y_pul:.2f});"]
    tikz.append(f"    \\draw[semithick] ({xm - 0.17:.2f},{y_pac + 0.2:.2f}) -- ({xm + 0.17:.2f},{y_pac + 0.2:.2f});")
    tikz.append(f"    \\filldraw[semithick, fill=white] ({xm:.2f},{y_pac + 0.47:.2f}) circle (0.1);")
tikz.append("\\end{tikzpicture}")
open(f"{AQUI}/modelo_relacional.tikz", "w").write("\n".join(tikz) + "\n")
print("notação:", NOTACAO, "| esquerda:", ESQ, "| direita:", DIR, "| pares quase sobrepostos (antes -> depois):", sobrepostos)
print("tabelas:", len(T), "| cruzamentos no desenho:", tot_cr, "| FK declaradas na base:", sorted(fk_declaradas))
print("colunas:", {t: len(c) for t, c in T.items()})
