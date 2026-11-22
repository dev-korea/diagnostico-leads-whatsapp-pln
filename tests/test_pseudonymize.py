import pytest
from tcc.pseudonymize import Pseudonymizer, hash_identifier


def test_hash_phone_formats_equal_and_secret_dependent():
    a = hash_identifier("+55 (53) 99123-4567", "s" * 16)
    b = hash_identifier("5553991234567", "s" * 16)
    c = hash_identifier("5553991234567", "t" * 16)
    assert a == b and a != c and len(a) == 24


def test_requires_secret():
    with pytest.raises(ValueError):
        Pseudonymizer(secret="curta")


@pytest.mark.parametrize("raw,tag", [
    ("meu email é carla.s@gmail.com ok", "[EMAIL]"),
    ("veja https://site.com/x?a=1 e www.foo.com.br", "[LINK]"),
    ("meu CPF 123.456.789-09", "[CPF]"),
    ("CNPJ 12.345.678/0001-95", "[CNPJ]"),
    ("whats (53) 99123-4567", "[TELEFONE]"),
    ("+55 53 991234567", "[TELEFONE]"),
    ("CEP 96400-000", "[CEP]"),
    ("me segue @carla_souza", "[USUARIO]"),
    ("moro na Rua das Flores, 123, centro", "[ENDERECO]"),
    ("Meu nome é Carla Souza", "[NOME]"),
])
def test_patterns(pseud, raw, tag):
    out, n = pseud.scrub(raw)
    assert tag in out and n >= 1


def test_does_not_mask_prices_hours_dates(pseud):
    s = "A consulta custa R$ 250,00 na quinta às 18h30, dia 12/03, são 3 sessões de 50 minutos. R$ 1500"
    out, n = pseud.scrub(s)
    assert out == s and n == 0


def test_names_accent_insensitive_and_roles():
    p = Pseudonymizer(secret="x" * 16, attendant_names=["Helena Recepção"], known_names=["João Silva"])
    out, _ = p.scrub("Oi Joao, aqui é a helena. Falei com SILVA.")
    assert "Joao" not in out and "helena" not in out and "SILVA" not in out
    assert p.role_of("helena recepcao") == "atendente" and p.role_of("Maria") == "lead" and p.role_of(None) == "sistema"


def test_group_chat_rejected(pseud):
    from tcc.parser import parse_text
    msgs, _ = parse_text("12/03/2024 10:00 - A: oi\n12/03/2024 10:01 - B: oi\n12/03/2024 10:02 - Clinica Vida Plena: oi")
    with pytest.raises(ValueError):
        pseud.pseudonymize_messages(msgs, "Grupo")


def test_media_system_replaced(pseud):
    from tcc.parser import parse_text
    msgs, _ = parse_text("12/03/2024 10:00 - Mensagens protegidas com a criptografia\n12/03/2024 10:01 - Ana: <Mídia oculta>")
    _, rows = pseud.pseudonymize_messages(msgs, "Ana")
    assert rows[0]["text"] == "[SISTEMA]" and rows[1]["text"] == "[MIDIA]"


def test_redaction_count_exact(pseud):
    out, n = pseud.scrub("tel 5553991234567 e email a@b.com")
    assert n == 2 and out == "tel [TELEFONE] e email [EMAIL]"


def test_card_luhn(pseud):
    assert pseud.scrub("cartão 4111 1111 1111 1111")[0] == "cartão [CARTAO]"
    assert "[CARTAO]" not in pseud.scrub("protocolo 1234 5678 9012 3456")[0]
