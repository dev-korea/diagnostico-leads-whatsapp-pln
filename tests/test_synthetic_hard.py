from tcc.synthetic import generate
from tcc.parser import parse_text
from pathlib import Path


def test_hard_generates_parseable_and_larger_noise(tmp_path):
    r = generate(str(tmp_path / "h"), n_leads=40, seed=3, hard=True)
    assert r["hard"] and r["n_conversations"] >= 40
    files = list(Path(tmp_path / "h").glob("*.txt"))
    assert len(files) == 40
    for f in files:
        msgs, _ = parse_text(f.read_text(encoding="utf-8"))
        assert len(msgs) >= 2


def test_default_unchanged_flag(tmp_path):
    r = generate(str(tmp_path / "e"), n_leads=10, seed=3)
    assert r["hard"] is False
