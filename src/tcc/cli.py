"""Linha de comando: python -m tcc.cli <comando>"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from .ingest import connect, init_db, ingest_file
from .pseudonymize import Pseudonymizer


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tcc")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init-db"); p.add_argument("--db", required=True)

    p = sub.add_parser("ingest", help="importa .txt exportados do WhatsApp")
    p.add_argument("--db", required=True)
    p.add_argument("--input", required=True, help="pasta com .txt ou um arquivo")
    p.add_argument("--attendants", required=True, help="nomes dos atendentes, separados por vírgula")
    p.add_argument("--names", default="", help="outros nomes a mascarar, separados por vírgula")
    p.add_argument("--gap-hours", type=float, default=24.0)

    a = ap.parse_args(argv)
    if a.cmd == "init-db":
        conn = connect(a.db); init_db(conn); print("ok"); return 0
    secret = os.environ.get("TCC_SECRET", "")
    ps = Pseudonymizer(secret=secret,
                       attendant_names=[x.strip() for x in a.attendants.split(",") if x.strip()],
                       known_names=[x.strip() for x in a.names.split(",") if x.strip()])
    conn = connect(a.db); init_db(conn)
    files = [a.input] if os.path.isfile(a.input) else sorted(glob.glob(os.path.join(a.input, "*.txt")))
    for f in files:
        try:
            print(json.dumps(ingest_file(conn, f, ps, gap_hours=a.gap_hours), ensure_ascii=False))
        except Exception as e:  # um arquivo ruim não derruba o lote
            print(json.dumps({"status": "error", "file": os.path.basename(f), "error": str(e)},
                             ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
