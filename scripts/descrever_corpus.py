"""Tabelas descritivas do corpus rotulado para o capítulo de Resultados (não usa texto, só rótulos e divisão).

Mostra, por parte da divisão (treino/val/teste): funil, qualificação derivada, objeções e gargalos; a referência
de classe majoritária no teste; e a qualidade das regras de gargalo frente ao rótulo humano.

Uso:  python scripts/descrever_corpus.py [--db data/processed/tcc.db] [--split v1] [--annotator adjudicado]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from sklearn.metrics import f1_score                      # noqa: E402

from tcc.ingest import connect, load_conversations        # noqa: E402
from tcc.labels import get_labels                         # noqa: E402
from tcc.rules import rule_gargalos                       # noqa: E402
from tcc.taxonomy import FUNIL, GARGALOS, OBJECOES, QUALIFICACAO   # noqa: E402

PARTES = ["train", "val", "test"]


def tabela(titulo, linhas, cont, totais):
    print(f"\n## {titulo}\n")
    print("| | " + " | ".join(PARTES) + " | total |")
    print("|---|" + "---:|" * (len(PARTES) + 1))
    for ln in linhas:
        vals = [cont[p][ln] for p in PARTES]
        print(f"| {ln} | " + " | ".join(str(v) for v in vals) + f" | {sum(vals)} |")
    print("| **conversas** | " + " | ".join(str(totais[p]) for p in PARTES) + f" | {sum(totais.values())} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(RAIZ / "data/processed/tcc.db"))
    ap.add_argument("--split", default="v1")
    ap.add_argument("--annotator", default="adjudicado")
    a = ap.parse_args()
    conn = connect(a.db)
    gold = get_labels(conn, a.annotator)
    parts = {r["conversation_id"]: r["part"] for r in conn.execute(
        "SELECT conversation_id, part FROM splits WHERE split_name=?", (a.split,))}
    if not gold:
        raise SystemExit(f"Sem rótulos do anotador '{a.annotator}'. Rode import-labels e build-gold antes.")
    if not parts:
        print(f"(aviso) divisão '{a.split}' não existe; tudo será contado como 'train'. Rode make-splits.")
    ids = sorted(gold)
    part = {c: parts.get(c, "train") for c in ids}
    totais = Counter(part.values())
    esc, fun, qual, obj, garg = (dict((p, Counter()) for p in PARTES) for _ in range(5))
    for c in ids:
        g, p = gold[c], part[c]
        esc[p][g["escopo"]] += 1
        if g["funil"]:
            fun[p][g["funil"]] += 1; qual[p][g["qualificacao"]] += 1
        for o in g["objecao"] or ["(nenhuma)"]:
            obj[p][o] += 1
        for x in g["gargalo"] or ["(nenhuma)"]:
            garg[p][x] += 1
    print(f"# Corpus rotulado — anotador '{a.annotator}', divisão '{a.split}'")
    tabela("Escopo", ["em_escopo", "fora_de_escopo"], esc, totais)
    tabela("Funil", FUNIL, fun, totais)
    tabela("Qualificação (derivada)", QUALIFICACAO, qual, totais)
    tabela("Objeções (multirrótulo)", OBJECOES + ["(nenhuma)"], obj, totais)
    tabela("Gargalos (multirrótulo)", GARGALOS + ["(nenhuma)"], garg, totais)

    # referência de classe majoritária (maioria do TREINO aplicada ao TESTE)
    tr = [gold[c]["qualificacao"] for c in ids if part[c] == "train" and gold[c]["qualificacao"]]
    te = [gold[c]["qualificacao"] for c in ids if part[c] == "test" and gold[c]["qualificacao"]]
    if tr and te:
        maj = Counter(tr).most_common(1)[0][0]
        f1 = f1_score(te, [maj] * len(te), labels=QUALIFICACAO, average="macro", zero_division=0)
        acc = sum(y == maj for y in te) / len(te)
        print(f"\n## Referência de classe majoritária (qualificação, teste n={len(te)})\n")
        print(f"classe = {maj} | F1 macro = {f1:.3f} | acurácia = {acc:.3f}")

    # regras de gargalo x rótulo humano
    print("\n## Regras de gargalo x rótulo humano (todas as conversas rotuladas)\n")
    print("| gargalo | VP | FP | FN | precisão | revocação |")
    print("|---|---:|---:|---:|---:|---:|")
    convs = {c["conversation_id"]: c for c in load_conversations(conn)}
    for lab in ("demora_resposta", "sem_followup"):
        vp = fp = fn = 0
        for c in ids:
            pred = lab in rule_gargalos(convs[c]); real = lab in gold[c]["gargalo"]
            vp += pred and real; fp += pred and not real; fn += real and not pred
        prec = vp / (vp + fp) if vp + fp else float("nan")
        rev = vp / (vp + fn) if vp + fn else float("nan")
        print(f"| {lab} | {vp} | {fp} | {fn} | {prec:.2f} | {rev:.2f} |")


if __name__ == "__main__":
    main()
