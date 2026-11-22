"""Ingestão idempotente de exportações .txt no banco (SQLite)."""
from __future__ import annotations

import hashlib
import os
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from .parser import chat_title_from_filename, infer_dayfirst, parse_text
from .pseudonymize import Pseudonymizer
from .segment import conversation_stats, segment

SCHEMA = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"
ISO = "%Y-%m-%dT%H:%M:%S"


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    # Migração v1.0 -> v1.1: a tabela `labels` antiga tem CHECK sem 'funil'; recria preservando as linhas.
    old = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='labels'").fetchone()
    migrate = bool(old and old[0] and "'funil'" not in old[0])
    if migrate:
        conn.execute("ALTER TABLE labels RENAME TO labels_old")
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    if migrate:
        conn.execute("""INSERT INTO labels(label_id, conversation_id, annotator, dimension, value, created_at)
                        SELECT label_id, conversation_id, annotator, dimension, value, created_at FROM labels_old""")
        conn.execute("DROP TABLE labels_old")
    conn.commit()


def _message_ids(lead_id: str, rows: list[dict]) -> list[str]:
    """ID estável; ocorrências idênticas no mesmo minuto recebem contador (Android só tem minutos)."""
    seen: Counter = Counter()
    ids = []
    for r in rows:
        key = (r["ts"].strftime(ISO), r["sender_role"], r["text"])
        seen[key] += 1
        raw = "|".join([lead_id, *key, str(seen[key])])
        ids.append(hashlib.sha256(raw.encode()).hexdigest()[:32])
    return ids


def ingest_file(conn: sqlite3.Connection, path: str, pseud: Pseudonymizer,
                gap_hours: float = 24.0, chat_title: Optional[str] = None) -> dict:
    """Importa um arquivo. Retorna resumo; reimportar o mesmo arquivo é no-op."""
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if conn.execute("SELECT 1 FROM imports WHERE file_sha256=?", (sha,)).fetchone():
        return {"status": "skipped_duplicate_file", "file": os.path.basename(path)}

    content = raw.decode("utf-8-sig", errors="replace")
    msgs, fmt = parse_text(content, dayfirst=infer_dayfirst(content))
    title = chat_title or chat_title_from_filename(os.path.basename(path))
    lead_id, rows = pseud.pseudonymize_messages(msgs, title)
    if not rows:
        return {"status": "empty", "file": os.path.basename(path)}
    ids = _message_ids(lead_id, rows)
    for r, i in zip(rows, ids):
        r["message_id"] = i

    # junta com o que já existe do lead (re-exportação com mais mensagens)
    existing = conn.execute(
        "SELECT message_id, ts, sender_role, text, is_media, n_redactions FROM messages WHERE lead_id=?",
        (lead_id,)).fetchall()
    known = {e["message_id"] for e in existing}
    merged = [dict(message_id=e["message_id"], ts=datetime.strptime(e["ts"], ISO),
                   sender_role=e["sender_role"], text=e["text"], is_media=e["is_media"],
                   n_redactions=e["n_redactions"]) for e in existing]
    new_rows = [r for r in rows if r["message_id"] not in known]
    merged += new_rows

    convs = segment(merged, gap_hours=gap_hours)
    # estabilidade dos IDs de conversa (rótulos dependem deles)
    old = {r["conversation_id"]: r["started_at"] for r in conn.execute(
        "SELECT conversation_id, started_at FROM conversations WHERE lead_id=?", (lead_id,))}
    new_map = {f"{lead_id}-{c.seq:03d}": c.started_at.strftime(ISO) for c in convs}
    for cid, started in old.items():
        if new_map.get(cid) != started:
            raise RuntimeError(
                f"Re-segmentação alteraria a conversa {cid}; reimporte do zero ou revise os rótulos.")

    with conn:  # transação
        conn.execute("INSERT OR IGNORE INTO leads(lead_id, first_seen, last_seen) VALUES (?,?,?)",
                     (lead_id, convs[0].started_at.strftime(ISO), convs[-1].ended_at.strftime(ISO)))
        conn.execute("UPDATE leads SET first_seen=?, last_seen=? WHERE lead_id=?",
                     (convs[0].started_at.strftime(ISO), convs[-1].ended_at.strftime(ISO), lead_id))
        conn.execute(
            "INSERT INTO imports(file_sha256, lead_id, device_format, n_messages) VALUES (?,?,?,?)",
            (sha, lead_id, fmt, len(rows)))
        for c in convs:
            cid = f"{lead_id}-{c.seq:03d}"
            st = conversation_stats(c)
            conn.execute(
                """INSERT INTO conversations(conversation_id, lead_id, seq_in_lead, started_at, ended_at,
                       n_messages, n_lead_messages, n_agent_messages, first_response_sec, median_response_sec)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(conversation_id) DO UPDATE SET ended_at=excluded.ended_at,
                       n_messages=excluded.n_messages, n_lead_messages=excluded.n_lead_messages,
                       n_agent_messages=excluded.n_agent_messages,
                       first_response_sec=excluded.first_response_sec,
                       median_response_sec=excluded.median_response_sec""",
                (cid, lead_id, c.seq, c.started_at.strftime(ISO), c.ended_at.strftime(ISO),
                 st["n_messages"], st["n_lead_messages"], st["n_agent_messages"],
                 st["first_response_sec"], st["median_response_sec"]))
            for m in c.messages:
                conn.execute(
                    """INSERT INTO messages(message_id, conversation_id, lead_id, ts, sender_role, text,
                           is_media, n_redactions) VALUES (?,?,?,?,?,?,?,?)
                       ON CONFLICT(message_id) DO UPDATE SET conversation_id=excluded.conversation_id""",
                    (m["message_id"], cid, lead_id, m["ts"].strftime(ISO), m["sender_role"], m["text"],
                     m["is_media"], m["n_redactions"]))
    return {"status": "ok", "file": os.path.basename(path), "format": fmt, "lead_id": lead_id,
            "new_messages": len(new_rows), "conversations": len(convs)}


def load_conversations(conn: sqlite3.Connection):
    """Lista de dicts {conversation_id, lead_id, seq_in_lead, messages:[...]} para os experimentos."""
    rows = conn.execute("SELECT * FROM conversations ORDER BY lead_id, seq_in_lead").fetchall()
    out = []
    for r in rows:
        msgs = [dict(m) for m in conn.execute(
            "SELECT ts, sender_role, text FROM messages WHERE conversation_id=? ORDER BY ts, rowid",
            (r["conversation_id"],))]
        out.append({**dict(r), "messages": msgs})
    return out
