"""Gera as matrizes de confusão do conjunto de teste para o texto (PDF vetorial + PNG + trecho LaTeX).

Lê as previsões em results/preds_*_{tarefa}.csv (as mesmas usadas por tcc.experiments.evaluate) e o
rótulo-ouro (`adjudicado`, divisão v1) do banco. Para o BERTimbau usa o voto majoritário das sementes,
igual ao evaluate.py, de modo que os números batem com as tabelas do Capítulo 4.

Figuras geradas em results/figuras/:
    cm_qualificacao.{pdf,png}       qualificado x não qualificado, um painel por método
    cm_objecao_presenca.{pdf,png}   com objeção x sem objeção (saída multirrótulo agregada)
    cm_objecao_rotulos.{pdf,png}    uma matriz 2x2 por categoria presente no teste e por método
    figuras_confusao.tex            ambientes figure prontos para colar no Overleaf

Uso (na pasta do projeto; no Colab, depois de os.chdir(PROJ)):
    python scripts/matrizes_confusao.py
    python scripts/matrizes_confusao.py --db data/processed/tcc.db --results results --out results/figuras
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

from tcc.dataset import TASKS, load_task_frame  # noqa: E402
from tcc.experiments.evaluate import load_methods  # noqa: E402
from tcc.ingest import connect  # noqa: E402

# Ordem e nomes dos métodos no texto (mesma ordem das tabelas do Capítulo 4).
METODOS = [("tfidf_lr", "TF-IDF + regressão logística"),
           ("bertimbau", "BERTimbau (voto de 5 sementes)"),
           ("llm_fewshot", "Qwen 2.5 7B few-shot")]

# Escala sequencial de um só matiz (azul), do quase-branco ao escuro: intensidade = proporção da linha.
CMAP = LinearSegmentedColormap.from_list(
    "seq_azul", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
NOMES_OBJ = {"preco": "Preço", "horario_disponibilidade": "Horário/disponib.", "localizacao": "Localização",
             "confianca_resultado": "Confiança/resultado", "convenio_reembolso": "Convênio/reembolso",
             "adiamento": "Adiamento", "concorrencia": "Concorrência",
             "servico_nao_oferecido": "Serviço não oferecido"}
TINTA, TINTA_SEC, SUPERFICIE ="#1f1f1f", "#5f5f5f", "#ffffff"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.edgecolor": SUPERFICIE, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def nome_metodo(chave: str) -> str:
    for prefixo, nome in METODOS:
        if chave.startswith(prefixo):
            return nome
    return chave


def ordenar(methods: dict) -> list[str]:
    ordem = [p for p, _ in METODOS]
    return sorted(methods, key=lambda m: next((i for i, p in enumerate(ordem) if m.startswith(p)), 99))


def matriz(y_true: list, y_pred: list, classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def cor_texto(valor: float) -> str:
    r, g, b = to_rgb(CMAP(valor))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return SUPERFICIE if lum < 0.45 else TINTA


def desenhar(ax, cm: np.ndarray, rotulos: list[str], titulo: str, mostrar_y: bool, pct: bool = True):
    tot = cm.sum(axis=1, keepdims=True)
    prop = np.divide(cm, tot, out=np.zeros(cm.shape, dtype=float), where=tot > 0)
    ax.imshow(prop, cmap=CMAP, vmin=0, vmax=1)
    n = len(rotulos)
    # 2 px de superfície entre as células
    for k in range(n + 1):
        ax.axhline(k - 0.5, color=SUPERFICIE, lw=2); ax.axvline(k - 0.5, color=SUPERFICIE, lw=2)
    for i in range(n):
        for j in range(n):
            c = cor_texto(prop[i, j])
            ax.text(j, i - (0.09 if pct else 0), f"{cm[i, j]}", ha="center", va="center",
                    fontsize=12 if pct else 10, fontweight="bold", color=c)
            if pct and tot[i, 0] > 0:
                ax.text(j, i + 0.22, f"{100 * prop[i, j]:.0f}%", ha="center", va="center", fontsize=7.5, color=c)
    ax.set_xticks(range(n), rotulos); ax.set_yticks(range(n), rotulos if mostrar_y else [""] * n)
    ax.tick_params(length=0, colors=TINTA)
    ax.set_xlabel("Previsto", color=TINTA_SEC)
    if mostrar_y:
        ax.set_ylabel("Real", color=TINTA_SEC)
    ax.set_title(titulo, color=TINTA, pad=6)
    for s in ax.spines.values():
        s.set_visible(False)


def painel_por_metodo(mats: dict, rotulos: list[str], arquivo: Path, rodape: str):
    ms = list(mats)
    fig, axes = plt.subplots(1, len(ms), figsize=(2.15 * len(ms) + 0.9, 2.55), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for k, (ax, m) in enumerate(zip(axes, ms)):
        desenhar(ax, mats[m], rotulos, nome_metodo(m), mostrar_y=(k == 0))
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1)), ax=axes, shrink=0.8,
                      fraction=0.04, pad=0.02)
    cb.set_label(rodape, color=TINTA_SEC); cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    salvar(fig, arquivo)


def salvar(fig, arquivo: Path):
    fig.savefig(arquivo.with_suffix(".pdf"), facecolor=SUPERFICIE)
    fig.savefig(arquivo.with_suffix(".png"), dpi=300, facecolor=SUPERFICIE)
    plt.close(fig)
    print("  ", arquivo.with_suffix(".pdf").name, "/", arquivo.with_suffix(".png").name)


def imprimir(titulo: str, mats: dict, rotulos: list[str]):
    print(f"\n{titulo}  (linhas = real, colunas = previsto: {rotulos})")
    for m in mats:
        print(f"  {nome_metodo(m):34s}", mats[m].tolist())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(RAIZ / "data/processed/tcc.db"))
    ap.add_argument("--results", default=str(RAIZ / "results"))
    ap.add_argument("--out", default=str(RAIZ / "results/figuras"))
    ap.add_argument("--split", default="v1")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    conn = connect(a.db)

    # ---------- Qualificação ----------
    df = load_task_frame(conn, "qualificacao", a.split)
    te = df[df.part == "test"].reset_index(drop=True)
    classes = TASKS["qualificacao"]["labels"]                       # ['qualificado', 'nao_qualificado']
    met = load_methods("qualificacao", a.results, list(te.conversation_id), multilabel=False)
    mats_q = {m: matriz(list(te.y), met[m]["vote"], classes) for m in ordenar(met)}
    imprimir(f"Qualificação (teste, n = {len(te)})", mats_q, classes)
    print("Figuras:")
    painel_por_metodo(mats_q, ["Qualificado", "Não qualif."], out / "cm_qualificacao",
                      "Proporção da classe real")

    # ---------- Objeções ----------
    df = load_task_frame(conn, "objecao", a.split)
    te = df[df.part == "test"].reset_index(drop=True)
    met = load_methods("objecao", a.results, list(te.conversation_id), multilabel=True)
    ordem = ordenar(met)
    yt = [set(y) for y in te.y]
    pres = lambda s: "com" if s else "sem"  # noqa: E731
    mats_p = {m: matriz([pres(y) for y in yt], [pres(p) for p in met[m]["vote"]], ["com", "sem"]) for m in ordem}
    imprimir("Objeções, presença (com, sem)", mats_p, ["com", "sem"])
    painel_por_metodo(mats_p, ["Com objeção", "Sem objeção"], out / "cm_objecao_presenca",
                      "Proporção da classe real")

    presentes = [l for l in TASKS["objecao"]["labels"] if any(l in y for y in yt)]
    fig, axes = plt.subplots(len(ordem), len(presentes), figsize=(1.55 * len(presentes) + 1.2, 1.75 * len(ordem) + 0.5),
                             constrained_layout=True, squeeze=False)
    print("\nObjeções por categoria (linhas = real [sim, não], colunas = previsto [sim, não])")
    for i, m in enumerate(ordem):
        for j, lab in enumerate(presentes):
            sim = lambda s: "sim" if lab in s else "nao"  # noqa: E731
            cm = matriz([sim(y) for y in yt], [sim(p) for p in met[m]["vote"]], ["sim", "nao"])
            print(f"  {nome_metodo(m):34s} {lab:24s}", cm.tolist())
            ax = axes[i, j]
            desenhar(ax, cm, ["Sim", "Não"], NOMES_OBJ.get(lab, lab) if i == 0 else "", mostrar_y=(j == 0), pct=False)
            if i < len(ordem) - 1:
                ax.set_xlabel(""); ax.set_xticks(range(2), ["", ""])
            if j == 0:
                ax.set_ylabel(f"{nome_metodo(m).split(' (')[0].replace(' + regressão logística', '')}\nReal",
                              color=TINTA_SEC)
    salvar(fig, out / "cm_objecao_rotulos")

    # ---------- Trecho LaTeX ----------
    n = len(te)
    tex = rf"""% Gerado por scripts/matrizes_confusao.py. Envie os PDFs de results/figuras/ para o Overleaf.
\begin{{figure}}[htbp]
\centering
\caption{{Matrizes de confusão da qualificação no conjunto de teste ($n = {n}$)}}
\label{{fig-cm-qualificacao}}
\includegraphics[width=\textwidth]{{cm_qualificacao.pdf}}
\fonte{{Elaborado pelo autor (2026). Linhas: classe real; colunas: classe prevista. A cor e o percentual indicam a proporção da classe real. Para o BERTimbau, voto majoritário das cinco sementes.}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\caption{{Matrizes de confusão da presença de objeção no conjunto de teste ($n = {n}$)}}
\label{{fig-cm-objecao-presenca}}
\includegraphics[width=\textwidth]{{cm_objecao_presenca.pdf}}
\fonte{{Elaborado pelo autor (2026). Uma conversa é considerada com objeção quando ao menos uma categoria é atribuída.}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\caption{{Matrizes de confusão por categoria de objeção no conjunto de teste ($n = {n}$)}}
\label{{fig-cm-objecao-rotulos}}
\includegraphics[width=\textwidth]{{cm_objecao_rotulos.pdf}}
\fonte{{Elaborado pelo autor (2026). Apenas as categorias com ocorrência no teste. Linhas: categoria presente na referência; colunas: categoria prevista.}}
\end{{figure}}
"""
    (out / "figuras_confusao.tex").write_text(tex, encoding="utf-8")
    print("   figuras_confusao.tex\n\nSaída em", out)


if __name__ == "__main__":
    main()
