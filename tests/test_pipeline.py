import csv
from pathlib import Path

from tcc.ingest import ingest_file, load_conversations
from tcc.labels import import_labels, get_labels, agreement, export_for_annotation, validate_row, disagreements
from tcc.segment import segment, response_times
from tcc.splits import make_splits
from tcc.rules import rule_gargalos
from datetime import datetime, timedelta
import pytest


def _ingest_all(db, pseud, d):
    res = [ingest_file(db, str(f), pseud) for f in sorted(Path(d).glob("*.txt"))]
    assert all(r["status"] == "ok" for r in res), res
    return res


def test_ingest_idempotent_and_no_pii(db, pseud, syn_dir):
    _ingest_all(db, pseud, syn_dir)
    n_msg = db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    again = [ingest_file(db, str(f), pseud) for f in sorted(Path(syn_dir).glob("*.txt"))]
    assert all(r["status"] == "skipped_duplicate_file" for r in again)
    assert db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == n_msg
    # nenhuma PII residual no banco
    import re
    texts = [r[0] for r in db.execute("SELECT text FROM messages")]
    blob = "\n".join(texts)
    assert not re.search(r"\d{3}\.\d{3}\.\d{3}-\d{2}", blob)       # CPF
    assert not re.search(r"\(53\)\s*9\d{4}-\d{4}", blob)            # telefone
    titles = [f.stem.replace("Conversa do WhatsApp com ", "") for f in Path(syn_dir).glob("*.txt")]
    # nome completo de qualquer lead não aparece nos textos
    assert not any(t in blob for t in titles)
    # primeiros nomes dos leads também mascarados (mesmo em 'Meu nome é X')
    import unicodedata
    low = unicodedata.normalize("NFKD", blob).lower()
    assert "meu nome e carla" not in low and "me chamo carla" not in low


def test_conversation_ids_and_recurrence(db, pseud, syn_dir):
    _ingest_all(db, pseud, syn_dir)
    rows = db.execute("SELECT lead_id, MAX(seq_in_lead) m, COUNT(*) c FROM conversations GROUP BY lead_id").fetchall()
    assert any(r["m"] >= 2 for r in rows)            # há recorrência
    assert all(r["m"] == r["c"] for r in rows)       # seq contínuo


def test_gold_labels_align_with_ingested_conversations(db, pseud, syn_dir):
    _ingest_all(db, pseud, syn_dir)
    res = import_labels(db, str(Path(syn_dir) / "labels_gold.csv"), "adjudicado")
    assert res["errors"] == [] and res["imported"] == db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    gold = get_labels(db)
    assert len(gold) == res["imported"]


def test_reexport_with_more_messages_keeps_ids(db, pseud, tmp_path):
    p = pseud
    f1 = tmp_path / "Conversa do WhatsApp com Ana Lima.txt"
    base = ("12/03/2025 10:00 - Ana Lima: oi\n12/03/2025 10:05 - Clinica Vida Plena: ola\n")
    f1.write_text(base, encoding="utf-8")
    r1 = ingest_file(db, str(f1), p)
    f1.write_text(base + "12/03/2025 10:06 - Ana Lima: valor?\n20/03/2025 09:00 - Ana Lima: voltei\n", encoding="utf-8")
    r2 = ingest_file(db, str(f1), p)
    assert r1["conversations"] == 1 and r2["conversations"] == 2 and r2["new_messages"] == 2
    assert db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 4


def test_same_minute_duplicates_kept(db, pseud, tmp_path):
    f = tmp_path / "Conversa do WhatsApp com Bia Costa.txt"
    f.write_text("12/03/2025 10:00 - Bia Costa: ok\n12/03/2025 10:00 - Bia Costa: ok\n", encoding="utf-8")
    ingest_file(db, str(f), pseud)
    assert db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2


def test_segmentation_window_and_response_time():
    t = datetime(2025, 1, 1, 10)
    m = lambda dt, role: {"ts": t + dt, "sender_role": role, "text": "x"}
    msgs = [m(timedelta(0), "lead"), m(timedelta(minutes=10), "atendente"),
            m(timedelta(hours=23), "lead"), m(timedelta(hours=60), "lead")]
    assert len(segment(msgs, 24)) == 2 and len(segment(msgs, 48)) == 1 and len(segment(msgs, 1)) == 3
    assert response_times(msgs[:2]) == [600.0]


def test_labels_validation():
    ok = {"escopo": "em_escopo", "funil": "intencao", "objecao": "preco;adiamento", "gargalo": "nenhuma"}
    v = validate_row(ok)
    assert v["objecao"] == ["adiamento", "preco"] and v["funil"] == ["intencao"]
    assert v["qualificacao"] == ["qualificado"]                       # binário DERIVADO do funil
    for bad in ({**ok, "funil": ""}, {**ok, "funil": "talvez"}, {**ok, "objecao": ""},
                {**ok, "objecao": "preco;nenhuma"}, {**ok, "gargalo": "foo"},
                {**ok, "escopo": "fora_de_escopo"},                  # fora de escopo não tem funil
                {**ok, "qualificacao": "nao_qualificado"},           # contradiz o funil
                {**ok, "desfecho": "desmarcou"},                     # desmarcou exige funil = agendado
                {**ok, "para_terceiro": "talvez"}):
        with pytest.raises(ValueError):
            validate_row(bad)
    fora = validate_row({**ok, "escopo": "fora_de_escopo", "funil": ""})
    assert "funil" not in fora and "qualificacao" not in fora


def test_funil_deriva_binario_e_metadados():
    from tcc.taxonomy import FUNIL, derive_qualificacao
    assert [derive_qualificacao(f) for f in FUNIL] == ["nao_qualificado", "nao_qualificado", "qualificado", "qualificado"]
    base = {"escopo": "em_escopo", "objecao": "servico_nao_oferecido", "gargalo": "sem_vaga"}
    v = validate_row({**base, "funil": "agendado", "desfecho": "desmarcou", "para_terceiro": "sim"})
    assert v["desfecho"] == ["desmarcou"] and v["funil"] == ["agendado"] and v["qualificacao"] == ["qualificado"]
    assert validate_row({**base, "funil": "sem_dialogo"})["qualificacao"] == ["nao_qualificado"]


def test_import_grava_funil_e_qualificacao_derivada(db, pseud, syn_dir):
    _ingest_all(db, pseud, syn_dir)
    import_labels(db, str(Path(syn_dir) / "labels_gold.csv"), "adjudicado")
    g = get_labels(db)
    em = [v for v in g.values() if v["escopo"] == "em_escopo"]
    assert em and all(v["funil"] and v["qualificacao"] for v in em)
    assert all(v["funil"] is None and v["qualificacao"] is None for v in g.values() if v["escopo"] == "fora_de_escopo")


def test_kappa_ponderado_do_funil(db, pseud, syn_dir, tmp_path):
    _ingest_all(db, pseud, syn_dir)
    gold = Path(syn_dir) / "labels_gold.csv"
    import_labels(db, str(gold), "a1")
    rows = list(csv.DictReader(open(gold, encoding="utf-8")))
    for r in rows:                                  # a2 discorda só entre estágios vizinhos (agendado -> intencao)
        if r["funil"] == "agendado":
            r["funil"] = "intencao"
    p2 = tmp_path / "a2.csv"
    with open(p2, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    import_labels(db, str(p2), "a2")
    f = agreement(db)["funil"]
    assert f["kappa_ponderado"] is not None and f["kappa_ponderado"] >= f["kappa_simples"]
    la, lb = get_labels(db, "a1"), get_labels(db, "a2")
    assert disagreements(db)                        # o funil diverge (agendado vs intencao)...
    assert all(la[c]["qualificacao"] == lb[c]["qualificacao"] for c in la)   # ...mas o binário derivado é igual


def test_init_db_migra_labels_v1_0(tmp_path):
    import sqlite3
    from tcc.ingest import init_db
    conn = sqlite3.connect(str(tmp_path / "old.db"))
    conn.executescript("""CREATE TABLE conversations(conversation_id TEXT PRIMARY KEY);
        INSERT INTO conversations VALUES ('c1');
        CREATE TABLE labels(label_id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
          annotator TEXT NOT NULL, dimension TEXT NOT NULL CHECK (dimension IN ('escopo','qualificacao','objecao','gargalo')),
          value TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        INSERT INTO labels(conversation_id, annotator, dimension, value) VALUES ('c1','a1','escopo','em_escopo');""")
    init_db(conn)
    assert conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 1            # linha preservada
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("INSERT INTO labels(conversation_id, annotator, dimension, value) VALUES ('c1','a1','funil','agendado')")
    assert conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 2            # CHECK novo aceita 'funil'


def test_agreement_and_disagreements(db, pseud, syn_dir, tmp_path):
    _ingest_all(db, pseud, syn_dir)
    gold = Path(syn_dir) / "labels_gold.csv"
    import_labels(db, str(gold), "a1")
    rows = list(csv.DictReader(open(gold, encoding="utf-8")))
    rows[0]["escopo"] = "fora_de_escopo" if rows[0]["escopo"] == "em_escopo" else "em_escopo"
    rows[0]["funil"] = "" if rows[0]["escopo"] == "fora_de_escopo" else "agendado"
    rows[0]["qualificacao"] = "" if rows[0]["escopo"] == "fora_de_escopo" else "qualificado"
    p2 = tmp_path / "a2.csv"
    with open(p2, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    import_labels(db, str(p2), "a2")
    ag = agreement(db)
    assert ag["n"] > 50 and 0.9 < ag["escopo"] < 1.0 and disagreements(db) == [rows[0]["conversation_id"]]


def test_splits_group_no_leakage_and_stable(db, pseud, syn_dir):
    _ingest_all(db, pseud, syn_dir)
    import_labels(db, str(Path(syn_dir) / "labels_gold.csv"), "adjudicado")
    c1 = make_splits(db, "v1", seed=1)
    parts = {r["conversation_id"]: r["part"] for r in db.execute("SELECT * FROM splits WHERE split_name='v1'")}
    lead_parts = {}
    for r in db.execute("SELECT conversation_id, lead_id FROM conversations"):
        lead_parts.setdefault(r["lead_id"], set()).add(parts[r["conversation_id"]])
    assert all(len(v) == 1 for v in lead_parts.values())            # lead em uma só parte
    assert sum(c1.values()) == len(parts) and c1["test"] > 0 and c1["val"] > 0 and c1["train"] > c1["test"]
    before = dict(parts)
    make_splits(db, "v1", seed=1)
    after = {r["conversation_id"]: r["part"] for r in db.execute("SELECT * FROM splits WHERE split_name='v1'")}
    assert before == after                                          # determinístico


def test_splits_fixed_test_set(db, pseud, syn_dir):
    """Teste fixo (subconjunto cego): o teste é exatamente os leads indicados; nada deles vaza para treino/val."""
    _ingest_all(db, pseud, syn_dir)
    import_labels(db, str(Path(syn_dir) / "labels_gold.csv"), "adjudicado")
    leads = sorted(r[0] for r in db.execute("SELECT lead_id FROM leads"))[:10]
    marks = ",".join("?" * len(leads))
    fixed = [r[0] for r in db.execute(f"SELECT conversation_id FROM conversations WHERE lead_id IN ({marks})", leads)]
    rotuladas = set(get_labels(db, "adjudicado"))
    fixed = [c for c in fixed if c in rotuladas]          # o gerador sintético pode deixar conversa sem rótulo
    # omite uma conversa de um lead que tem outra na lista: ela deve voltar ao teste junto com o lead
    por_lead = {}
    for cid in sorted(fixed):
        por_lead.setdefault(cid.rsplit("-", 1)[0], []).append(cid)
    omitir = next((v[-1] for v in por_lead.values() if len(v) > 1), None)
    c = make_splits(db, "v1", seed=1, test_ids=[x for x in fixed if x != omitir])
    parts = {r["conversation_id"]: r["part"] for r in db.execute("SELECT * FROM splits WHERE split_name='v1'")}
    assert {cid for cid, p in parts.items() if p == "test"} == set(fixed)   # inclui as demais conversas do lead
    assert c["val"] > 0 and c["train"] > 0
    with pytest.raises(ValueError):
        make_splits(db, "v1", seed=1, test_ids=["id-que-nao-existe"])


def test_export_for_annotation(db, pseud, syn_dir, tmp_path):
    _ingest_all(db, pseud, syn_dir)
    n = export_for_annotation(db, str(tmp_path / "a.csv"))
    rows = list(csv.DictReader(open(tmp_path / "a.csv", encoding="utf-8-sig")))
    assert n == len(rows) and rows[0]["escopo"] == "" and "Cliente:" in rows[0]["texto"]


def test_rule_gargalos_matches_gold_reasonably(db, pseud, syn_dir):
    _ingest_all(db, pseud, syn_dir)
    import_labels(db, str(Path(syn_dir) / "labels_gold.csv"), "adjudicado")
    gold = get_labels(db)
    tp = fp = fn = 0
    for c in load_conversations(db):
        pred = "demora_resposta" in rule_gargalos(c)
        true = "demora_resposta" in gold[c["conversation_id"]]["gargalo"]
        tp += pred and true; fp += pred and not true; fn += (not pred) and true
    assert tp > 0 and fp == 0 and fn == 0     # nos sintéticos, a regra de tempo é exata


def test_build_gold_e_adjudicacao(db, pseud, syn_dir, tmp_path):
    """Ouro = concordantes + anotação única; divergentes ficam pendentes até serem adjudicadas."""
    from tcc.labels import build_gold, export_disagreements
    _ingest_all(db, pseud, syn_dir)
    gold = Path(syn_dir) / "labels_gold.csv"
    import_labels(db, str(gold), "a1")
    rows = list(csv.DictReader(open(gold, encoding="utf-8")))
    sub = [dict(r) for r in rows[:20]]                    # a2 só rotula 20 (o "subconjunto cego")
    sub[0]["escopo"] = "fora_de_escopo" if sub[0]["escopo"] == "em_escopo" else "em_escopo"
    sub[0]["funil"] = "" if sub[0]["escopo"] == "fora_de_escopo" else "agendado"
    sub[0]["qualificacao"] = ""
    p2 = tmp_path / "a2.csv"
    with open(p2, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sub[0].keys()); w.writeheader(); w.writerows(sub)
    import_labels(db, str(p2), "a2")

    st = build_gold(db)
    assert st["pendentes"] == [sub[0]["conversation_id"]] and st["concordantes"] == 19
    assert st["anotacao_unica"] == len(rows) - 20 and st["total_no_ouro"] == len(rows) - 1

    adj = tmp_path / "adjudicar.csv"
    assert export_disagreements(db, str(adj)) == 1
    lin = list(csv.DictReader(open(adj, encoding="utf-8-sig")))
    assert lin[0]["escopo"] == "" and lin[0]["a1_escopo"] != lin[0]["a2_escopo"] and "escopo" in lin[0]["divergem_em"]
    for k in ("escopo", "funil", "objecao", "gargalo"):   # consenso: fica com a1
        lin[0][k] = lin[0][f"a1_{k}"]
    with open(adj, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=lin[0].keys()); w.writeheader(); w.writerows(lin)
    assert import_labels(db, str(adj), "adjudicado")["imported"] == 1
    st = build_gold(db)                                   # idempotente; agora nada pendente
    assert st["pendentes"] == [] and st["total_no_ouro"] == len(rows)
    assert get_labels(db, "adjudicado") == get_labels(db, "a1")
