"""Avaliação comparativa no MESMO conjunto de teste: F1 + IC (bootstrap por lead) + McNemar.

Métodos com várias sementes (BERTimbau): métricas = média das sementes; o IC reamostra leads e
recalcula a média. Para McNemar usa-se o VOTO MAJORITÁRIO das sementes (documente na Metodologia).
"""
from __future__ import annotations

import argparse
import glob
import itertools
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..dataset import TASKS, load_task_frame
from ..ingest import connect
from ..metrics import (bootstrap_ci_seeds, holm, macro_f1_fn, mcnemar_exact, mcnemar_per_label,
                       multilabel_macro_f1_fn, multilabel_report, paired_bootstrap_diff, single_label_report)
from .common import load_predictions


def _vote(preds_by_seed: list[list], multilabel: bool) -> list:
    out = []
    for votes in zip(*preds_by_seed):
        if multilabel:
            labs = set().union(*votes)
            out.append({l for l in labs if sum(l in v for v in votes) > len(votes) / 2})
        else:
            out.append(max(set(votes), key=lambda x: (votes.count(x), x)))
    return out


def load_methods(task: str, results_dir: str, ids: list[str], multilabel: bool) -> dict[str, dict]:
    """{método: {'by_seed': [[pred...]], 'vote': [...], 'version': str}} alinhado a `ids`."""
    methods: dict[str, dict] = {}
    for f in sorted(glob.glob(str(Path(results_dir) / f"preds_*_{task}.csv"))):
        rows = load_predictions(f, multilabel)
        if not rows:
            continue
        by_method = defaultdict(lambda: defaultdict(dict))
        ver = {}
        for r in rows:
            by_method[r["method"]][r["seed"]][r["conversation_id"]] = r["pred"]; ver[r["method"]] = r["model_version"]
        for m, seeds in by_method.items():
            lists = []
            for s, d in sorted(seeds.items(), key=lambda x: (x[0] is None, x[0])):
                missing = [i for i in ids if i not in d]
                if missing:
                    raise ValueError(f"{m} seed={s} não cobre {len(missing)} conversas do teste (ex.: {missing[0]})")
                lists.append([d[i] for i in ids])
            methods[m] = {"by_seed": lists, "vote": _vote(lists, multilabel), "version": ver[m]}
    return methods


def evaluate(db: str, task: str, results_dir: str = "results", split: str = "v1", n_boot: int = 2000,
             baseline: str = "tfidf_lr", seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = TASKS[task]; labels, ml = spec["labels"], spec["multilabel"]
    df = load_task_frame(connect(db), task, split)
    te = df[df.part == "test"].reset_index(drop=True)
    if te.empty:
        raise ValueError("Conjunto de teste vazio.")
    y, groups, ids = list(te.y), list(te.lead_id), list(te.conversation_id)
    methods = load_methods(task, results_dir, ids, ml)
    if not methods:
        raise ValueError(f"Sem previsões em {results_dir} para {task}.")
    metric = multilabel_macro_f1_fn(labels) if ml else macro_f1_fn(labels)
    summary, perclass = [], []
    for m, d in methods.items():
        ci = bootstrap_ci_seeds(metric, y, d["by_seed"], groups, n_boot=n_boot, seed=seed)
        rep = (multilabel_report if ml else single_label_report)(y, d["vote"], labels)
        row = {"task": task, "method": m, "model_version": d["version"], "n_test": len(y), "n_leads": len(set(groups)),
               "macro_f1": ci["estimate"], "ci95_low": ci["ci_low"], "ci95_high": ci["ci_high"],
               "std_seeds": ci["std_seeds"], "n_seeds": ci["n_seeds"]}
        row["micro_f1" if ml else "accuracy"] = rep["micro_f1" if ml else "accuracy"]
        summary.append(row)
        for l, v in rep["per_class"].items():
            perclass.append({"task": task, "method": m, "label": l, **v})
    pairs = []
    for a, b in itertools.combinations(methods, 2):
        va, vb = methods[a]["vote"], methods[b]["vote"]
        db_ = paired_bootstrap_diff(metric, y, va, vb, groups, n_boot=n_boot, seed=seed)
        if ml:
            per = mcnemar_per_label(y, va, vb, labels)
            p_min = min(r["p_holm"] for r in per.values())
            note = "menor p (Holm) entre rótulos"
        else:
            r = mcnemar_exact([x == t for x, t in zip(va, y)], [x == t for x, t in zip(vb, y)])
            p_min, note = r["p_value"], f"A só acerta={r['a_only_correct']}, B só acerta={r['b_only_correct']}"
        pairs.append({"task": task, "A": a, "B": b, "delta_macro_f1(A-B)": db_["diff"], "ci95_low": db_["ci_low"],
                      "ci95_high": db_["ci_high"], "ic_exclui_zero": db_["significant"],
                      "mcnemar_p": p_min, "nota": note})
    if pairs:  # Holm entre comparações
        for r, adj in zip(pairs, holm([r["mcnemar_p"] for r in pairs])):
            r["mcnemar_p_holm"] = adj
    return pd.DataFrame(summary), pd.DataFrame(perclass), pd.DataFrame(pairs)


def to_markdown(summary: pd.DataFrame, pairs: pd.DataFrame) -> str:
    s = summary.copy()
    s["F1-macro [IC95%]"] = [f"{r.macro_f1:.3f} [{r.ci95_low:.3f}; {r.ci95_high:.3f}]" +
                             (f" ±{r.std_seeds:.3f} ({r.n_seeds} sementes)" if r.n_seeds > 1 else "")
                             for r in s.itertuples()]
    cols = ["method", "model_version", "n_test", "F1-macro [IC95%]"]
    txt = s[cols].to_markdown(index=False)
    if len(pairs):
        p = pairs.copy()
        p["Δ F1 (A−B) [IC95%]"] = [f"{r._4:+.3f} [{r.ci95_low:+.3f}; {r.ci95_high:+.3f}]" for r in p.itertuples()]
        txt += "\n\nComparações pareadas (mesmo teste):\n\n" + p[["A", "B", "Δ F1 (A−B) [IC95%]", "mcnemar_p", "mcnemar_p_holm"]].to_markdown(index=False, floatfmt=".4f")
    return txt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True); ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--results", default="results"); ap.add_argument("--split", default="v1")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    summary, perclass, pairs = evaluate(a.db, a.task, a.results, a.split, a.n_boot)
    out = Path(a.results)
    summary.to_csv(out / f"summary_{a.task}.csv", index=False)
    perclass.to_csv(out / f"perclass_{a.task}.csv", index=False)
    pairs.to_csv(out / f"pairs_{a.task}.csv", index=False)
    md = to_markdown(summary, pairs)
    (out / f"summary_{a.task}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
