"""Empacota o que vai para o Google Colab: src/ e sql/ (código). NUNCA inclui data/raw nem segredos.

Uso (na pasta do projeto, com a venv ativa):
    python scripts/empacotar_colab.py                 # só código  -> dist/colab_pacote.zip
    python scripts/empacotar_colab.py --com-banco     # + data/processed/tcc.db (já pseudonimizado e rotulado)
"""
import argparse
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
IGNORAR = {"__pycache__", ".pytest_cache"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--com-banco", action="store_true", help="inclui data/processed/tcc.db")
    a = ap.parse_args()
    saida = RAIZ / "dist" / "colab_pacote.zip"
    saida.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(saida, "w", zipfile.ZIP_DEFLATED) as z:
        for pasta in ("src", "sql"):
            for f in sorted((RAIZ / pasta).rglob("*")):
                if f.is_file() and not (set(f.parts) & IGNORAR) and f.suffix != ".pyc":
                    z.write(f, f.relative_to(RAIZ).as_posix()); n += 1
        if a.com_banco:
            db = RAIZ / "data" / "processed" / "tcc.db"
            if not db.exists():
                raise SystemExit("data/processed/tcc.db não existe.")
            z.write(db, "data/processed/tcc.db"); n += 1
    print(f"{n} arquivos -> {saida} ({saida.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
