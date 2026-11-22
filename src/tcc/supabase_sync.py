"""Sincroniza o banco local (SQLite, já pseudonimizado) com o Supabase (PostgreSQL).

  push  : envia leads, conversas e mensagens locais -> Supabase (idempotente: ON CONFLICT DO NOTHING)
          e cria linhas 'pendente' na tabela `rotulagem` para o anotador informado.
  pull  : traz da tabela `rotulagem` os rótulos com status 'rotulada' -> tabela local `labels`.

Conexão: variável de ambiente SUPABASE_DB_URL (Project Settings > Database > Connection string, modo
"Session pooler"). NUNCA coloque a URL ou senhas no repositório.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

SCHEMA_PG = Path(__file__).resolve().parents[2] / "sql" / "schema_postgres.sql"


def _pg():
    import psycopg2
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        sys.exit("Defina SUPABASE_DB_URL (string de conexão do Supabase).")
    return psycopg2.connect(url)


def init_remote(pg) -> None:
    with pg.cursor() as cur:
        cur.execute(SCHEMA_PG.read_text(encoding="utf-8"))
    pg.commit()


def push(db: str, pg, annotator: str | None = None, batch: int = 1000) -> dict:
    from psycopg2.extras import execute_values
    lo = sqlite3.connect(db); lo.row_factory = sqlite3.Row
    out = {}
    with pg.cursor() as cur:
        q = {
            "leads": ("INSERT INTO leads(lead_id, first_seen, last_seen, source) VALUES %s ON CONFLICT DO NOTHING",
                      "SELECT lead_id, first_seen, last_seen, source FROM leads"),
            "conversations": ("""INSERT INTO conversations(conversation_id, lead_id, seq_in_lead, started_at, ended_at,
                                 n_messages, n_lead_messages, n_agent_messages, first_response_sec, median_response_sec)
                                 VALUES %s ON CONFLICT DO NOTHING""",
                              """SELECT conversation_id, lead_id, seq_in_lead, started_at, ended_at, n_messages,
                                 n_lead_messages, n_agent_messages, first_response_sec, median_response_sec
                                 FROM conversations"""),
            "messages": ("""INSERT INTO messages(message_id, conversation_id, lead_id, ts, sender_role, text, is_media,
                            n_redactions) VALUES %s ON CONFLICT DO NOTHING""",
                         """SELECT message_id, conversation_id, lead_id, ts, sender_role, text, is_media, n_redactions
                            FROM messages"""),
        }
        for t, (ins, sel) in q.items():
            rows = [tuple(r) for r in lo.execute(sel)]
            if t == "messages":
                rows = [r[:6] + (bool(r[6]), r[7]) for r in rows]
            for i in range(0, len(rows), batch):
                execute_values(cur, ins, rows[i:i + batch])
            out[t] = len(rows)
        if annotator:
            cur.execute("""INSERT INTO rotulagem(conversation_id, annotator)
                           SELECT conversation_id, %s FROM conversations ON CONFLICT DO NOTHING""", (annotator,))
            out["rotulagem_linhas_criadas_para"] = annotator
    pg.commit()
    return out


def update_text(db: str, pg, batch: int = 1000) -> dict:
    """Reenvia o TEXTO das mensagens (o `push` não sobrescreve). Use depois de `python -m tcc.scrub apply`,
    para que nomes mascarados localmente também sumam da cópia no Supabase."""
    from psycopg2.extras import execute_values
    lo = sqlite3.connect(db)
    rows = [tuple(r) for r in lo.execute("SELECT message_id, text, n_redactions FROM messages")]
    n = 0
    with pg.cursor() as cur:
        for i in range(0, len(rows), batch):
            execute_values(cur, """UPDATE messages AS m SET text = v.text, n_redactions = v.n_redactions
                                   FROM (VALUES %s) AS v(message_id, text, n_redactions)
                                   WHERE m.message_id = v.message_id AND m.text IS DISTINCT FROM v.text""",
                           rows[i:i + batch])
            n += cur.rowcount
    pg.commit()
    return {"mensagens_locais": len(rows), "mensagens_atualizadas_no_supabase": n}


def pull_labels(db: str, pg) -> dict:
    """Rótulos 'rotulada' do Supabase -> tabela local `labels` (substitui o que o anotador já tinha)."""
    lo = sqlite3.connect(db)
    n = 0
    with pg.cursor() as cur:
        cur.execute("""SELECT conversation_id, annotator, escopo, funil, qualificacao, objecao, gargalo,
                              desfecho, para_terceiro
                       FROM rotulagem WHERE status = 'rotulada'""")
        rows = cur.fetchall()
    for conv, ann, esc, funil, qual, objs, garg, desf, terc in rows:
        lo.execute("DELETE FROM labels WHERE conversation_id=? AND annotator=?", (conv, ann))
        pares = [("escopo", esc), ("funil", funil), ("qualificacao", qual), ("desfecho", desf),
                 ("para_terceiro", terc)]
        pares += [("objecao", o) for o in (objs or ["nenhuma"])]
        pares += [("gargalo", g) for g in (garg or ["nenhuma"])]
        for dim, val in pares:
            if val:
                lo.execute("INSERT OR IGNORE INTO labels(conversation_id, annotator, dimension, value) VALUES (?,?,?,?)",
                           (conv, ann, dim, val))
        n += 1
    lo.commit()
    return {"conversas_rotuladas_importadas": n}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-remote", help="cria as tabelas no Supabase (ou cole sql/schema_postgres.sql no SQL Editor)")
    p = sub.add_parser("push"); p.add_argument("--db", required=True)
    p.add_argument("--annotator", default=None, help="cria a fila 'pendente' para este anotador (ex.: a1)")
    p = sub.add_parser("pull-labels"); p.add_argument("--db", required=True)
    p = sub.add_parser("update-text", help="reenvia o texto das mensagens (após tcc.scrub apply)")
    p.add_argument("--db", required=True)
    a = ap.parse_args(argv)
    pg = _pg()
    try:
        if a.cmd == "init-remote":
            init_remote(pg); print("ok")
        elif a.cmd == "push":
            print(push(a.db, pg, a.annotator))
        elif a.cmd == "update-text":
            print(update_text(a.db, pg))
        else:
            print(pull_labels(a.db, pg))
    finally:
        pg.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
