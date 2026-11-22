"""Divisão treino/validação/teste FIXA e versionada, agrupada por lead (sem vazamento).

Conversas do mesmo lead (recorrência) ficam sempre na mesma parte; senão o modelo veria o
mesmo cliente no treino e no teste. Estratificação aproximada pelo rótulo do lead.

Com `test_ids` (ex.: o subconjunto CEGO da pré-rotulagem), o teste é exatamente esse conjunto (mais as
demais conversas dos mesmos leads) e só a validação é sorteada. É o desenho do estudo real: o teste não
pode conter conversas rotuladas a partir de sugestão de LLM, senão a comparação com o método LLM fica viciada.
"""
from __future__ import annotations

import csv
import sqlite3
from typing import Iterable, Optional

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .labels import get_labels


def make_splits(conn: sqlite3.Connection, split_name: str = "v1", seed: int = 42,
                annotator: str = "adjudicado", test_frac: float = 0.2, val_frac: float = 0.2,
                test_ids: Optional[Iterable[str]] = None) -> dict:
    gold = get_labels(conn, annotator)
    rows = conn.execute("SELECT conversation_id, lead_id FROM conversations ORDER BY conversation_id").fetchall()
    rows = [r for r in rows if r["conversation_id"] in gold]
    if len(rows) < 10:
        raise ValueError("Poucas conversas rotuladas para dividir.")
    ids = np.array([r["conversation_id"] for r in rows])
    groups = np.array([r["lead_id"] for r in rows])
    strat = np.array([("fora" if gold[c]["escopo"] == "fora_de_escopo" else gold[c]["qualificacao"]) for c in ids])
    if test_ids is not None:
        wanted = set(test_ids)
        test_leads = {g for c, g in zip(ids, groups) if c in wanted}
        missing = wanted - set(ids)
        if missing:
            raise ValueError(f"{len(missing)} conversas do teste fixo sem rótulo '{annotator}' "
                             f"(ex.: {sorted(missing)[:3]}). Rotule/adjudique antes de dividir.")
        test_idx = np.array([i for i, g in enumerate(groups) if g in test_leads])
        if len(test_idx) in (0, len(ids)):
            raise ValueError("Teste fixo vazio ou igual ao total.")
    else:
        n_splits = round(1 / test_frac)  # 5 -> um fold = 20%
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [te for _, te in skf.split(ids, strat, groups)]
        test_idx = folds[0]
    rest = np.setdiff1d(np.arange(len(ids)), test_idx)
    n_val_splits = max(2, round((1 - test_frac) / val_frac))
    skf2 = StratifiedGroupKFold(n_splits=n_val_splits, shuffle=True, random_state=seed + 1)
    _, val_rel = next(iter(skf2.split(ids[rest], strat[rest], groups[rest])))
    val_idx = rest[val_rel]
    part = {i: "train" for i in range(len(ids))}
    for i in test_idx: part[i] = "test"
    for i in val_idx: part[i] = "val"
    with conn:
        conn.execute("DELETE FROM splits WHERE split_name=?", (split_name,))
        conn.executemany("INSERT INTO splits(conversation_id, split_name, part) VALUES (?,?,?)",
                         [(ids[i], split_name, part[i]) for i in part])
    counts = {p: sum(1 for v in part.values() if v == p) for p in ("train", "val", "test")}
    return counts


def read_ids_csv(path: str) -> list[str]:
    """Lê a coluna conversation_id de um CSV (ex.: rotular_a2_cego.csv)."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [r["conversation_id"].strip() for r in csv.DictReader(fh) if r.get("conversation_id", "").strip()]
