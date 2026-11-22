"""Comandos auxiliares: importar rótulos, exportar para rotulagem, concordância e divisões."""
from __future__ import annotations

import argparse
import json

from .ingest import connect, init_db
from .labels import (agreement, build_gold, disagreements, export_disagreements, export_for_annotation,
                     import_labels)
from .splits import make_splits, read_ids_csv


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tcc.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export-annotation"); p.add_argument("--db", required=True); p.add_argument("--out", required=True)
    p.add_argument("--only-unlabeled-by", default=None)
    p = sub.add_parser("import-labels"); p.add_argument("--db", required=True); p.add_argument("--csv", required=True)
    p.add_argument("--annotator", required=True, help="a1 | a2 | adjudicado")
    p = sub.add_parser("agreement"); p.add_argument("--db", required=True)
    p.add_argument("--a", default="a1"); p.add_argument("--b", default="a2")
    p = sub.add_parser("export-disagreements", help="CSV das divergencias a1 x a2 para adjudicar")
    p.add_argument("--db", required=True); p.add_argument("--out", required=True)
    p.add_argument("--a", default="a1"); p.add_argument("--b", default="a2")
    p = sub.add_parser("build-gold", help="monta a camada 'adjudicado' (ouro) a partir de a1, a2 e das adjudicacoes")
    p.add_argument("--db", required=True); p.add_argument("--a", default="a1"); p.add_argument("--b", default="a2")
    p = sub.add_parser("make-splits"); p.add_argument("--db", required=True)
    p.add_argument("--name", default="v1"); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--annotator", default="adjudicado")
    p.add_argument("--test-csv", default=None,
                   help="CSV com conversation_id do teste FIXO (ex.: data/processed/rotular_a2_cego.csv)")
    a = ap.parse_args(argv)
    conn = connect(a.db); init_db(conn)
    if a.cmd == "export-annotation":
        print(export_for_annotation(conn, a.out, a.only_unlabeled_by), "conversas exportadas")
    elif a.cmd == "import-labels":
        print(json.dumps(import_labels(conn, a.csv, a.annotator), ensure_ascii=False, indent=2))
    elif a.cmd == "agreement":
        print(json.dumps(agreement(conn, a.a, a.b), ensure_ascii=False, indent=2))
        print("divergências:", len(disagreements(conn, a.a, a.b)))
    elif a.cmd == "export-disagreements":
        print(export_disagreements(conn, a.out, a.a, a.b), "conversas divergentes exportadas")
    elif a.cmd == "build-gold":
        print(json.dumps(build_gold(conn, a.a, a.b), ensure_ascii=False, indent=2))
    elif a.cmd == "make-splits":
        print(make_splits(conn, a.name, a.seed, a.annotator,
                          test_ids=read_ids_csv(a.test_csv) if a.test_csv else None))


if __name__ == "__main__":
    main()
