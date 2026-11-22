"""Lista em que LINHA da planilha está cada conversa do subconjunto CEGO.

NÃO inclui status, fase nem observação (para não contaminar a rotulagem independente).
Precisa do TCC_SECRET (o mesmo da ingestão). Saída local: data/processed/cegas_linhas_planilha.csv

Uso:  $env:TCC_SECRET = "..."   e depois   python scripts/linhas_cegas_planilha.py
"""
import csv
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from tcc.parser import chat_title_from_filename   # noqa: E402
from tcc.prelabel import PRE_COL                  # noqa: E402
from tcc.pseudonymize import hash_identifier      # noqa: E402


def main():
    secret = os.environ.get("TCC_SECRET", "")
    if len(secret) < 16:
        raise SystemExit("Defina TCC_SECRET no terminal (mesmo valor da ingestão).")
    por_lead = {}
    with open(RAIZ / "data/raw/_planilha_status.csv", newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["tipo"] == "ok":
                por_lead[hash_identifier(chat_title_from_filename(r["arquivo"]), secret)] = r
    with open(RAIZ / "data/processed/rotular_a2_cego.csv", newline="", encoding="utf-8-sig") as fh:
        cegas = list(csv.DictReader(fh))
    saida, achou = [], 0
    for c in cegas:
        r = por_lead.get(c["conversation_id"].rsplit("-", 1)[0])
        achou += bool(r)
        saida.append({"conversation_id": c["conversation_id"], "linha_planilha": r["linha"] if r else "",
                      "nome_planilha": r["nome"] if r else "", "data_contato": r["data_contato"] if r else "",
                      "arquivo_txt": r["arquivo"] if r else ""})
    saida.sort(key=lambda x: int(x["linha_planilha"] or 10**9))
    alvo = RAIZ / "data/processed/cegas_linhas_planilha.csv"
    with open(alvo, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(saida[0])); w.writeheader(); w.writerows(saida)
    print(f"{achou} de {len(cegas)} cegas localizadas -> {alvo}")
    print("Linhas:", ", ".join(x["linha_planilha"] for x in saida if x["linha_planilha"]))
    if achou == 0:
        print("Nenhuma casou: confira se o TCC_SECRET é o mesmo da ingestão.")


if __name__ == "__main__":
    main()
