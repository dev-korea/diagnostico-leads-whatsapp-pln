"""Etapa 1 do pipeline: verificação determinística de escopo por léxico (sem modelo estatístico).

Passos: (i) pegar as N primeiras mensagens do lead; (ii) normalizar texto e termos (minúsculas, sem acento,
sem pontuação, espaços únicos); (iii) correspondência por palavra inteira / expressão na ordem;
(iv) em escopo se ao menos um termo casar. Os termos que casaram são devolvidos para rastreabilidade.
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class ScopeResult:
    in_scope: bool
    matched: list[str]


def check_scope(messages: list[dict], terms: list[str], n_first: int = 3) -> ScopeResult:
    lead_texts = [m["text"] for m in messages if m["sender_role"] == "lead"
                  and m["text"] not in ("[MIDIA]", "[APAGADA]", "[SISTEMA]")][:n_first]
    hay = f" {normalize(' '.join(lead_texts))} "
    matched = []
    for term in terms:
        nt = normalize(term)
        if nt and f" {nt} " in hay:           # espaços nas bordas = palavra inteira / expressão na ordem
            matched.append(term)
    return ScopeResult(bool(matched), matched)


def load_terms(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT term FROM catalog_terms ORDER BY term_id")]


def add_terms(conn: sqlite3.Connection, catalog_item: str, terms: list[str]) -> int:
    n = 0
    with conn:
        for t in terms:
            cur = conn.execute("INSERT OR IGNORE INTO catalog_terms(catalog_item, term, term_normalized) VALUES (?,?,?)",
                               (catalog_item, t, normalize(t)))
            n += cur.rowcount
    return n
