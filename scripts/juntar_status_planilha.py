"""Cola o status da planilha (sua classificação original) em rotular_a1.csv, SÓ nas linhas abertas.

Precisa do TCC_SECRET (o mesmo da ingestão) para calcular o lead_id de cada arquivo. O segredo fica no seu
terminal; o script não o grava. Linhas CEGAS não recebem nada (preserva a anotação independente).
Colunas novas: status_planilha, qual_planilha (qualificado | nao_qualificado).

Uso:  $env:TCC_SECRET = "..."   e depois   python scripts/juntar_status_planilha.py
"""
import csv
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from tcc.parser import chat_title_from_filename   # noqa: E402
from tcc.prelabel import MODEL_FIELDS, PRE_COL    # noqa: E402
from tcc.pseudonymize import hash_identifier      # noqa: E402

NOVAS = ["status_planilha", "qual_planilha"]


def main():
    secret = os.environ.get("TCC_SECRET", "")
    if len(secret) < 16:
        raise SystemExit("Defina TCC_SECRET no terminal (mesmo valor da ingestão).")
    status = {}
    with open(RAIZ / "data/raw/_planilha_status.csv", newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["tipo"] == "ok":
                lead = hash_identifier(chat_title_from_filename(r["arquivo"]), secret)
                status[lead] = r["status"].strip()
    alvo = RAIZ / "data/processed/rotular_a1.csv"
    with open(alvo, newline="", encoding="utf-8-sig") as fh:
        linhas = list(csv.DictReader(fh))
    achou = 0
    for r in linhas:
        r.setdefault("status_planilha", ""); r.setdefault("qual_planilha", "")
        if r[PRE_COL] == "cega":
            continue
        s = status.get(r["conversation_id"].rsplit("-", 1)[0])
        if s:
            r["status_planilha"] = s
            r["qual_planilha"] = "nao_qualificado" if s.lower().startswith("desqual") else "qualificado"
            achou += 1
    with open(alvo, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=MODEL_FIELDS + NOVAS); w.writeheader(); w.writerows(linhas)
    print(f"abertas com status: {achou}  (de {sum(1 for r in linhas if r[PRE_COL] != 'cega')} abertas)")
    if achou == 0:
        print("Nenhuma casou: confira se o TCC_SECRET é o mesmo da ingestão.")


if __name__ == "__main__":
    main()
