"""Métricas e inferência estatística para comparar métodos no MESMO conjunto de teste."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score


# ---------- métricas -------------------------------------------------------------------
def single_label_report(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(labels), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f)),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, l in enumerate(labels)},
    }


def to_matrix(sets: Sequence[set], labels: Sequence[str]) -> np.ndarray:
    return np.array([[int(l in s) for l in labels] for s in sets], dtype=int)


def multilabel_report(true_sets: Sequence[set], pred_sets: Sequence[set], labels: Sequence[str]) -> dict:
    Yt, Yp = to_matrix(true_sets, labels), to_matrix(pred_sets, labels)
    p, r, f, s = precision_recall_fscore_support(Yt, Yp, average=None, zero_division=0)
    return {
        "micro_f1": float(f1_score(Yt, Yp, average="micro", zero_division=0)),
        "macro_f1": float(np.mean(f)),
        "exact_match": float(np.mean((Yt == Yp).all(axis=1))),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, l in enumerate(labels)},
    }


def macro_f1_fn(labels: Sequence) -> Callable:
    """F1-macro sobre as classes PRESENTES no y_true da amostra (essencial no bootstrap: uma classe
    ausente numa reamostra não pode virar F1=0 e deprimir artificialmente a estatística)."""
    def f(yt, yp):
        present = [l for l in labels if l in set(yt)]
        if not present:
            return float("nan")
        return float(np.mean(precision_recall_fscore_support(yt, yp, labels=present, zero_division=0)[2]))
    return f


def multilabel_macro_f1_fn(labels: Sequence[str]) -> Callable:
    def f(yt, yp):
        Yt, Yp = to_matrix(yt, labels), to_matrix(yp, labels)
        keep = Yt.sum(0) > 0
        if not keep.any():
            return float("nan")
        return float(f1_score(Yt[:, keep], Yp[:, keep], average="macro", zero_division=0))
    return f


# ---------- bootstrap por cluster (lead) ------------------------------------------------
def _cluster_indices(groups: Sequence) -> list[np.ndarray]:
    groups = np.asarray(groups)
    return [np.where(groups == g)[0] for g in np.unique(groups)]


def _resample(clusters: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    pick = rng.integers(0, len(clusters), len(clusters))
    return np.concatenate([clusters[i] for i in pick])


def bootstrap_ci(metric: Callable, y_true: Sequence, y_pred: Sequence, groups: Sequence,
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """IC percentil por bootstrap reamostrando LEADS (conversas do mesmo lead são dependentes)."""
    yt, yp = list(y_true), list(y_pred)
    clusters = _cluster_indices(groups)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = _resample(clusters, rng)
        stats.append(metric([yt[i] for i in idx], [yp[i] for i in idx]))
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(metric(yt, yp)), "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot}


def paired_bootstrap_diff(metric: Callable, y_true: Sequence, pred_a: Sequence, pred_b: Sequence,
                          groups: Sequence, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Diferença (A − B) com IC; se o IC não contém 0, a diferença é significativa a nível alpha."""
    yt, a, b = list(y_true), list(pred_a), list(pred_b)
    clusters = _cluster_indices(groups)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _resample(clusters, rng)
        t = [yt[i] for i in idx]
        diffs.append(metric(t, [a[i] for i in idx]) - metric(t, [b[i] for i in idx]))
    lo, hi = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(metric(yt, a) - metric(yt, b)), "ci_low": float(lo), "ci_high": float(hi),
            "significant": bool(lo > 0 or hi < 0)}


# ---------- McNemar exato ------------------------------------------------------------------
def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """Teste exato (binomial) para discordâncias pareadas entre A e B."""
    ca, cb = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b01 = int(np.sum(ca & ~cb))   # A acerta, B erra
    c10 = int(np.sum(~ca & cb))   # A erra, B acerta
    n = b01 + c10
    p = 1.0 if n == 0 else float(binomtest(min(b01, c10), n, 0.5, alternative="two-sided").pvalue)
    return {"a_only_correct": b01, "b_only_correct": c10, "p_value": p}


def holm(pvals: Sequence[float]) -> list[float]:
    """Correção de Holm-Bonferroni (múltiplos rótulos/comparações)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def mcnemar_per_label(true_sets, pred_a, pred_b, labels) -> dict:
    """McNemar por rótulo (acerto binário daquele rótulo) com ajuste de Holm."""
    res, ps = {}, []
    for l in labels:
        t = np.array([l in s for s in true_sets]); a = np.array([l in s for s in pred_a]); b = np.array([l in s for s in pred_b])
        r = mcnemar_exact(a == t, b == t); res[l] = r; ps.append(r["p_value"])
    for l, adj in zip(labels, holm(ps)):
        res[l]["p_holm"] = adj
    return res


def bootstrap_ci_seeds(metric: Callable, y_true: Sequence, preds_by_seed: Sequence[Sequence], groups: Sequence,
                       n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Como bootstrap_ci, mas a estatística é a MÉDIA da métrica sobre as sementes de treino."""
    yt = list(y_true); P = [list(p) for p in preds_by_seed]
    clusters = _cluster_indices(groups)
    rng = np.random.default_rng(seed)
    f = lambda idx: float(np.mean([metric([yt[i] for i in idx], [p[i] for i in idx]) for p in P]))
    stats = [f(_resample(clusters, rng)) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    all_idx = np.arange(len(yt))
    per_seed = [metric(yt, p) for p in P]
    return {"estimate": f(all_idx), "ci_low": float(lo), "ci_high": float(hi),
            "std_seeds": float(np.std(per_seed, ddof=1)) if len(P) > 1 else 0.0, "n_seeds": len(P)}
