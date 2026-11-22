"""Análise de sensibilidade do limiar de segmentação (12 / 24 / 48 h). Só lê o banco; não altera nada.

Para cada limiar: nº de conversas, leads com mais de uma conversa e mediana de mensagens por conversa; e quantos
leads teriam a segmentação alterada em relação ao padrão de 24 h.

Uso:  python scripts/sensibilidade_segmentacao.py [--db data/processed/tcc.db] [--gaps 12 24 48]
"""
import argparse
import statistics
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from tcc.ingest import ISO, connect          # noqa: E402
from tcc.segment import segment              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(RAIZ / "data/processed/tcc.db"))
    ap.add_argument("--gaps", type=float, nargs="+", default=[12, 24, 48])
    ap.add_argument("--base", type=float, default=24)
    a = ap.parse_args()
    conn = connect(a.db)
    leads = {}
    for r in conn.execute("SELECT lead_id, ts, sender_role, text FROM messages ORDER BY lead_id, ts, rowid"):
        leads.setdefault(r["lead_id"], []).append(
            {"ts": datetime.strptime(r["ts"], ISO), "sender_role": r["sender_role"], "text": r["text"]})

    def cortes(gap):
        return {lid: tuple(len(c.messages) for c in segment(msgs, gap_hours=gap)) for lid, msgs in leads.items()}

    base = cortes(a.base)
    print(f"leads: {len(leads)} | mensagens: {sum(len(m) for m in leads.values())}\n")
    print("| limiar (h) | conversas | leads com >1 conversa | mediana msgs/conversa | leads que mudam vs "
          f"{a.base:g} h |")
    print("|---:|---:|---:|---:|---:|")
    for gap in a.gaps:
        seg = cortes(gap)
        tam = [n for t in seg.values() for n in t]
        print(f"| {gap:g} | {len(tam)} | {sum(len(t) > 1 for t in seg.values())} | "
              f"{statistics.median(tam):g} | {sum(seg[l] != base[l] for l in leads)} |")


if __name__ == "__main__":
    main()
