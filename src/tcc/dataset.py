"""Monta o DataFrame dos experimentos a partir do banco (texto + rótulo-ouro + split)."""
from __future__ import annotations

import sqlite3

import pandas as pd

from .ingest import load_conversations
from .labels import get_labels
from .segment import conversation_text
from .taxonomy import OBJECOES, QUALIFICACAO, ESCOPO

TASKS = {
    "escopo":       {"labels": ESCOPO, "multilabel": False, "filter_in_scope": False},
    "qualificacao": {"labels": QUALIFICACAO, "multilabel": False, "filter_in_scope": True},
    "objecao":      {"labels": OBJECOES, "multilabel": True, "filter_in_scope": True},
}


def load_task_frame(conn: sqlite3.Connection, task: str, split_name: str = "v1",
                    annotator: str = "adjudicado") -> pd.DataFrame:
    spec = TASKS[task]
    gold = get_labels(conn, annotator)
    parts = {r["conversation_id"]: r["part"] for r in conn.execute(
        "SELECT conversation_id, part FROM splits WHERE split_name=?", (split_name,))}
    rows = []
    for c in load_conversations(conn):
        cid = c["conversation_id"]
        if cid not in gold or cid not in parts:
            continue
        g = gold[cid]
        if spec["filter_in_scope"] and g["escopo"] != "em_escopo":
            continue
        y = g[task] if task != "objecao" else g["objecao"]
        rows.append({"conversation_id": cid, "lead_id": c["lead_id"], "seq_in_lead": c["seq_in_lead"],
                     "text": conversation_text(c["messages"]), "y": y, "part": parts[cid]})
    return pd.DataFrame(rows)
