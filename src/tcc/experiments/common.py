"""Formato único de previsões (CSV) compartilhado pelos três métodos."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

NONE = "nenhuma"
COLS = ["conversation_id", "task", "method", "model_version", "seed", "pred"]


def encode_pred(pred) -> str:
    if isinstance(pred, (set, list, tuple)):
        return ";".join(sorted(pred)) or NONE
    return str(pred)


def decode_pred(s: str, multilabel: bool):
    if not multilabel:
        return s
    return set() if s == NONE else set(s.split(";"))


def save_predictions(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for r in rows:
            w.writerow({**r, "pred": encode_pred(r["pred"])}); n += 1
    return n


def load_predictions(path: str | Path, multilabel: bool) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [{**r, "seed": int(r["seed"]) if r["seed"] != "" else None,
                 "pred": decode_pred(r["pred"], multilabel)} for r in csv.DictReader(fh)]
