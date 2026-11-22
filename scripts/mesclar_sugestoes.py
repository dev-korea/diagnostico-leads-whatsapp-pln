"""Mescla sugestões de pré-rótulo em data/processed/rotular_a1.csv, VALIDANDO pelas regras do guia.

Arquivo de sugestões (CSV): conversation_id,escopo,funil,objecao,gargalo,para_terceiro,observacao
  objecao / gargalo: valores separados por ';' (vazio = nenhuma). NÃO inclua demora_resposta: vem da regra (> 30 min).
Conversas do subconjunto cego são RECUSADAS (devem ficar em branco). Sugestão inválida é listada e não entra.

Uso:  python scripts/mesclar_sugestoes.py data/processed/sugestoes_lote_01.csv [--modelo claude-sonnet-5-5]
"""
import argparse
import csv
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from tcc.labels import NONE, validate_row       # noqa: E402
from tcc.prelabel import MODEL_FIELDS, PRE_COL, _demora   # noqa: E402


def _lista(v: str) -> list[str]:
    return [x.strip() for x in (v or "").replace(",", ";").split(";") if x.strip() and x.strip() != NONE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sugestoes")
    ap.add_argument("--alvo", default=str(RAIZ / "data/processed/rotular_a1.csv"))
    ap.add_argument("--modelo", default="claude-sonnet-5-5")
    a = ap.parse_args()

    with open(a.alvo, newline="", encoding="utf-8-sig") as fh:
        linhas = list(csv.DictReader(fh))
    idx = {r["conversation_id"]: r for r in linhas}
    ok, erros = 0, []
    with open(a.sugestoes, newline="", encoding="utf-8-sig") as fh:
        for s in csv.DictReader(fh):
            cid = s["conversation_id"].strip()
            r = idx.get(cid)
            if r is None:
                erros.append(f"{cid}: id inexistente"); continue
            if r[PRE_COL] == "cega":
                erros.append(f"{cid}: conversa CEGA, não pode receber sugestão"); continue
            garg = [g for g in _lista(s.get("gargalo", "")) if g != "demora_resposta"]
            if _demora(r["primeira_resposta_min"], r["mediana_resposta_min"]):
                garg.append("demora_resposta")
            novo = {"escopo": s["escopo"].strip(), "funil": s.get("funil", "").strip(),
                    "objecao": ";".join(sorted(set(_lista(s.get("objecao", "")))) ) or NONE,
                    "gargalo": ";".join(sorted(set(garg))) or NONE,
                    "para_terceiro": s.get("para_terceiro", "").strip(), "desfecho": s.get("desfecho", "").strip()}
            try:
                validate_row(novo)
            except ValueError as e:
                erros.append(f"{cid}: {e}"); continue
            r.update(novo)
            r["observacao"] = s.get("observacao", "").strip()
            r[PRE_COL] = f"sugerido:claude-code-sessao:{a.modelo}"
            ok += 1
    with open(a.alvo, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=MODEL_FIELDS); w.writeheader(); w.writerows(linhas)
    print(f"mescladas={ok} erros={len(erros)}")
    for e in erros:
        print(" -", e)


if __name__ == "__main__":
    main()
