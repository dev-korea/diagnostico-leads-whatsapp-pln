import json
from pathlib import Path

import pandas as pd
import pytest

from tcc.dataset import TASKS, load_task_frame
from tcc.experiments import llm, tfidf, bert, evaluate as ev
from tcc.experiments.common import save_predictions, load_predictions
from tcc.ingest import ingest_file
from tcc.labels import import_labels
from tcc.splits import make_splits


@pytest.fixture
def ready(db, pseud, syn_dir):
    for f in sorted(Path(syn_dir).glob("*.txt")):
        ingest_file(db, str(f), pseud)
    import_labels(db, str(Path(syn_dir) / "labels_gold.csv"), "adjudicado")
    make_splits(db, "v1", seed=3)
    return db


def test_predictions_roundtrip(tmp_path):
    save_predictions(tmp_path / "p.csv", [{"conversation_id": "a", "task": "objecao", "method": "m", "model_version": "v",
                                           "seed": 1, "pred": {"preco", "adiamento"}},
                                          {"conversation_id": "b", "task": "objecao", "method": "m", "model_version": "v",
                                           "seed": 1, "pred": set()}])
    rows = load_predictions(tmp_path / "p.csv", True)
    assert rows[0]["pred"] == {"preco", "adiamento"} and rows[1]["pred"] == set() and rows[0]["seed"] == 1


def test_task_frames_filter_scope(ready):
    esc = load_task_frame(ready, "escopo"); qual = load_task_frame(ready, "qualificacao")
    assert len(qual) < len(esc) and set(qual.y) <= {"qualificado", "nao_qualificado"}
    assert set(esc.part) == {"train", "val", "test"}


def test_parse_response_valid_and_invalid():
    assert llm.parse_response("qualificacao", 'ok {"label": "qualificado"} fim') == "qualificado"
    assert llm.parse_response("qualificacao", '{"label": "talvez"}') is None
    assert llm.parse_response("qualificacao", "sem json") is None
    assert llm.parse_response("objecao", '{"labels": ["preco", "adiamento"]}') == {"preco", "adiamento"}
    assert llm.parse_response("objecao", '{"labels": []}') == set()
    assert llm.parse_response("objecao", '{"labels": ["inventada"]}') is None
    assert llm.parse_response("objecao", '```json\n{"labels": ["nenhuma"]}\n```') == set()


def test_few_shot_examples_only_from_train(ready):
    df = load_task_frame(ready, "qualificacao")
    train = df[df.part == "train"]
    ex = llm.select_examples(train, "qualificacao", k=2, seed=0)
    texts = set(train.text.str[:900])
    assert len(ex) == 4 and all(t in texts for t, _ in ex)
    assert {y for _, y in ex} == {"qualificado", "nao_qualificado"}
    assert llm.select_examples(train, "qualificacao", 2, 0) == ex           # determinístico
    exm = llm.select_examples(load_task_frame(ready, "objecao").query("part=='train'"), "objecao", 2, 0)
    assert len(exm) >= 8


def test_prompt_contains_definitions_and_format():
    s, u = llm.build_prompt("objecao", "Cliente: caro", [("Cliente: oi", {"preco"})])
    assert "JSON" in s and "preco:" in s and '"labels": ["preco"]' in u and u.endswith("### Resposta\n")


class Flaky(llm.Provider):
    name, model = "flaky", "flaky-1"
    def __init__(self, replies): self.replies, self.calls = list(replies), 0
    def complete(self, system, user):
        self.calls += 1; return self.replies.pop(0)


def test_retry_on_invalid_then_valid_and_cache(tmp_path):
    cache = llm.Cache(str(tmp_path / "c.sqlite"))
    p = Flaky(["lixo", '{"label": "qualificado"}'])
    out, ok = llm.classify_one(p, cache, "qualificacao", "Cliente: quero marcar", [])
    assert ok and out == "qualificado" and p.calls == 2
    out2, ok2 = llm.classify_one(p, cache, "qualificacao", "Cliente: quero marcar", [])
    assert ok2 and out2 == "qualificado" and p.calls == 2          # veio do cache


def test_invalid_after_retries_flagged(tmp_path):
    cache = llm.Cache(str(tmp_path / "c.sqlite"))
    p = Flaky(["x", "y", "z"])
    out, ok = llm.classify_one(p, cache, "qualificacao", "t", [])
    assert not ok and out is None


def test_llm_run_with_mock_and_cache_reuse(ready, tmp_path):
    df = load_task_frame(ready, "objecao")
    cache = llm.Cache(str(tmp_path / "c.sqlite"))
    ids, preds, info = llm.run(df, "objecao", llm.MockProvider(), cache, k=2, workers=2)
    assert info["invalid"] == 0 and len(ids) == len(preds) == (df.part == "test").sum()
    ids2, preds2, _ = llm.run(df, "objecao", llm.MockProvider(), cache, k=2, workers=2)
    assert preds == preds2


def test_tfidf_runs_all_tasks(ready):
    for t in TASKS:
        df = load_task_frame(ready, t)
        ids, preds, C = tfidf.run(df, t)
        assert len(ids) == (df.part == "test").sum() and C in tfidf.C_GRID
        if t == "objecao":
            assert all(isinstance(p, set) for p in preds)


def test_bert_smoke_single_and_multi(ready):
    for t in ("qualificacao", "objecao"):
        df = load_task_frame(ready, t); spec = TASKS[t]
        tok, factory = bert.build_tiny(list(df.text), len(spec["labels"]), spec["multilabel"])
        ids, preds, info = bert.fit_predict(df, t, tok, factory, seed=0, epochs=2, batch_size=8, max_len=64)
        assert len(ids) == len(preds) and "best_val_macro_f1" in info
        # determinismo com mesma semente
        _, preds2, _ = bert.fit_predict(df, t, tok, factory, seed=0, epochs=2, batch_size=8, max_len=64)
        assert preds == preds2


def test_head_tail_truncation():
    from transformers import BertTokenizerFast
    tok, _ = bert.build_tiny(["a b c d e f g h i j"], 2, False)
    ids = bert.head_tail_ids(tok, " ".join("abcdefghij"), max_len=6)
    assert len(ids) == 6 and ids[0] == tok.cls_token_id and ids[-1] == tok.sep_token_id


def test_evaluate_end_to_end(ready, tmp_path):
    t = "qualificacao"; df = load_task_frame(ready, t)
    ids, p1, _ = tfidf.run(df, t)
    ids2, p2, _ = llm.run(df, t, llm.MockProvider(), llm.Cache(str(tmp_path / "c.sqlite")), k=2)
    save_predictions(tmp_path / f"preds_tfidf_{t}.csv", [{"conversation_id": i, "task": t, "method": "tfidf_lr",
                     "model_version": "x", "seed": 0, "pred": p} for i, p in zip(ids, p1)])
    save_predictions(tmp_path / f"preds_llm_{t}.csv", [{"conversation_id": i, "task": t, "method": "llm:mock",
                     "model_version": "m", "seed": 0, "pred": p} for i, p in zip(ids2, p2)])
    db_path = ready.execute("PRAGMA database_list").fetchone()[2]
    summ, per, pairs = ev.evaluate(db_path, t, str(tmp_path), n_boot=100)
    assert set(summ.method) == {"tfidf_lr", "llm:mock"} and len(pairs) == 1
    assert (summ.ci95_low <= summ.macro_f1).all() and (summ.macro_f1 <= summ.ci95_high).all()
    assert "F1-macro" in ev.to_markdown(summ, pairs)


def test_evaluate_rejects_incomplete_predictions(ready, tmp_path):
    t = "qualificacao"; df = load_task_frame(ready, t)
    te = df[df.part == "test"]
    save_predictions(tmp_path / f"preds_x_{t}.csv", [{"conversation_id": i, "task": t, "method": "x",
                     "model_version": "m", "seed": 0, "pred": "qualificado"} for i in te.conversation_id[:-1]])
    db_path = ready.execute("PRAGMA database_list").fetchone()[2]
    with pytest.raises(ValueError, match="não cobre"):
        ev.evaluate(db_path, t, str(tmp_path), n_boot=10)
