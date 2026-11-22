"""Baseline clássica: TF-IDF (palavras 1-2 gramas + caracteres 3-5) + Regressão Logística.

Hiperparâmetro C escolhido na VALIDAÇÃO (nunca no teste). Roda em minutos, sem GPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from ..dataset import TASKS, load_task_frame
from ..ingest import connect
from ..metrics import to_matrix
from .common import save_predictions

C_GRID = [0.1, 1.0, 10.0, 100.0]


def _features():
    return FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True, strip_accents="unicode")),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                              lowercase=True, strip_accents="unicode")),
    ])


def _make(C: float, multilabel: bool, seed: int):
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)
    return Pipeline([("f", _features()), ("clf", OneVsRestClassifier(lr) if multilabel else lr)])


def run(df, task: str, seed: int = 0):
    spec = TASKS[task]; labels = spec["labels"]; ml = spec["multilabel"]
    tr, va, te = (df[df.part == p] for p in ("train", "val", "test"))
    if ml:
        Ytr, Yva = to_matrix(list(tr.y), labels), to_matrix(list(va.y), labels)
        keep = np.where(Ytr.sum(0) > 0)[0]  # rótulos com ao menos 1 positivo no treino
        keep = keep[(Ytr[:, keep].sum(0) < len(tr))]
        if len(keep) == 0:
            raise ValueError("Nenhum rótulo com variação no treino.")
        def fit(C):
            m = _make(C, True, seed); m.fit(tr.text, Ytr[:, keep]); return m
        def pred_sets(m, texts):
            P = m.predict(texts)
            return [{labels[keep[j]] for j in np.where(row)[0]} for row in P]
        score = lambda m: f1_score(Yva[:, keep], m.predict(va.text), average="macro", zero_division=0)
    else:
        def fit(C):
            m = _make(C, False, seed); m.fit(tr.text, list(tr.y)); return m
        pred_sets = lambda m, texts: list(m.predict(texts))
        score = lambda m: f1_score(list(va.y), m.predict(va.text), average="macro", zero_division=0)
    best = max((fit(C) for C in C_GRID), key=score) if len(va) else fit(1.0)
    best_C = best.named_steps["clf"].estimator.C if ml else best.named_steps["clf"].C
    preds = pred_sets(best, te.text)
    return te.conversation_id.tolist(), preds, best_C


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True); ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--split", default="v1"); ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    df = load_task_frame(connect(a.db), a.task, a.split)
    ids, preds, C = run(df, a.task)
    n = save_predictions(Path(a.out) / f"preds_tfidf_{a.task}.csv", (
        {"conversation_id": i, "task": a.task, "method": "tfidf_lr", "model_version": f"C={C}", "seed": 0, "pred": p}
        for i, p in zip(ids, preds)))
    print(f"tfidf_lr/{a.task}: {n} previsões (C={C})")


if __name__ == "__main__":
    main()
