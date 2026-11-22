import numpy as np
import pytest
from tcc.metrics import (bootstrap_ci, holm, mcnemar_exact, macro_f1_fn, multilabel_macro_f1_fn, paired_bootstrap_diff,
                         single_label_report, multilabel_report, mcnemar_per_label, bootstrap_ci_seeds)


def test_macro_f1_ignores_absent_class():
    f = macro_f1_fn(["a", "b"])
    assert f(["a", "a"], ["a", "a"]) == 1.0           # classe 'b' ausente não zera a média


def test_single_label_report_values():
    r = single_label_report(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert r["accuracy"] == 0.75 and abs(r["per_class"]["a"]["recall"] - 0.5) < 1e-9
    assert abs(r["macro_f1"] - np.mean([2 / 3, 0.8])) < 1e-9


def test_multilabel_report():
    t = [{"x"}, {"y"}, set(), {"x", "y"}]
    p = [{"x"}, set(), set(), {"x", "y"}]
    r = multilabel_report(t, p, ["x", "y"])
    assert r["exact_match"] == 0.75 and r["per_class"]["y"]["recall"] == 0.5


def test_mcnemar_exact_known():
    # b=8, c=2 -> p exato bilateral = 0.109375
    a = [True] * 8 + [False] * 2 + [True] * 10
    b = [False] * 8 + [True] * 2 + [True] * 10
    r = mcnemar_exact(a, b)
    assert (r["a_only_correct"], r["b_only_correct"]) == (8, 2) and abs(r["p_value"] - 0.109375) < 1e-9
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_bootstrap_ci_contains_estimate_and_is_deterministic():
    rng = np.random.default_rng(0)
    y = list(rng.choice(["a", "b"], 80)); p = [t if rng.random() < .8 else ("b" if t == "a" else "a") for t in y]
    g = list(np.repeat(np.arange(40), 2))
    f = macro_f1_fn(["a", "b"])
    r1 = bootstrap_ci(f, y, p, g, n_boot=300, seed=1); r2 = bootstrap_ci(f, y, p, g, n_boot=300, seed=1)
    assert r1 == r2 and r1["ci_low"] <= r1["estimate"] <= r1["ci_high"] and r1["ci_high"] - r1["ci_low"] > 0


def test_cluster_bootstrap_wider_than_naive_when_clusters_correlated():
    y = ["a", "b"] * 40
    p = [("b" if t == "a" else "a") if i < 16 else t for i, t in enumerate(y)]   # erros só nos leads 0 e 1
    f = macro_f1_fn(["a", "b"])
    naive = bootstrap_ci(f, y, p, list(range(80)), n_boot=500, seed=2)
    clus = bootstrap_ci(f, y, p, list(np.repeat(np.arange(10), 8)), n_boot=500, seed=2)
    assert (clus["ci_high"] - clus["ci_low"]) > (naive["ci_high"] - naive["ci_low"])


def test_paired_bootstrap_detects_clear_difference():
    y = ["a", "b"] * 50; good = list(y); bad = ["a"] * 100
    r = paired_bootstrap_diff(macro_f1_fn(["a", "b"]), y, good, bad, list(range(100)), n_boot=300)
    assert r["diff"] > 0.3 and r["significant"]


def test_multilabel_mcnemar_and_seeds_ci():
    t = [{"x"}, set()] * 20; a = list(t); b = [set(), {"x"}] * 20
    res = mcnemar_per_label(t, a, b, ["x"])
    assert res["x"]["p_value"] < 0.001 and res["x"]["p_holm"] < 0.001
    ci = bootstrap_ci_seeds(multilabel_macro_f1_fn(["x"]), t, [a, b], list(range(40)), n_boot=100)
    assert ci["n_seeds"] == 2 and ci["std_seeds"] > 0
