from tcc.scope import check_scope, normalize, add_terms, load_terms

TERMS = ["quiropraxia", "dor lombar", "coluna", "hérnia de disco", "ajuste"]
M = lambda *t: [{"sender_role": "lead", "text": x} for x in t]


def test_normalize():
    assert normalize("  Hérnia,  de DISCO!! ") == "hernia de disco"


def test_in_scope_with_accent_and_phrase():
    r = check_scope(M("Oi, queria saber o valor da avaliação da minha coluna"), TERMS)
    assert r.in_scope and r.matched == ["coluna"]
    assert check_scope(M("tenho hernia de disco"), TERMS).matched == ["hérnia de disco"]


def test_out_of_scope():
    r = check_scope(M("Bom dia, represento uma empresa de software de gestão"), TERMS)
    assert not r.in_scope and r.matched == []


def test_whole_word_only():
    assert not check_scope(M("preciso de um reajustes gerais"), TERMS).in_scope   # 'ajuste' dentro de outra palavra
    assert not check_scope(M("dor na lombar"), TERMS).in_scope                    # expressão exige a ordem 'dor lombar'


def test_only_first_n_lead_messages_and_roles():
    msgs = M("oi", "tudo bem?", "ok", "tenho dor lombar")
    assert not check_scope(msgs, TERMS, n_first=3).in_scope
    assert check_scope(msgs, TERMS, n_first=4).in_scope
    atend = [{"sender_role": "atendente", "text": "fazemos quiropraxia"}] + M("oi")
    assert not check_scope(atend, TERMS).in_scope                                 # só fala do lead conta


def test_db_terms_idempotent(db):
    assert add_terms(db, "avaliacao", ["Coluna", "coluna!"]) == 1                # duplicata normalizada ignorada
    assert load_terms(db) == ["Coluna"]
