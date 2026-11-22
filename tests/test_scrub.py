from collections import Counter

from tcc.scrub import build_names_re, scrub_conversation


def _conv(*pairs):
    return [{"sender_role": r, "text": t} for r, t in pairs]


def test_mascara_resposta_ao_pedido_de_nome():
    out = scrub_conversation(_conv(("atendente", "pode me passar o seu nome completo por favor"),
                                   ("lead", "fulano de tal da silva"),
                                   ("atendente", "esta agendado")))
    assert out == ["pode me passar o seu nome completo por favor", "[NOME]", "esta agendado"]


def test_fala_so_com_nome_proprio_mas_nao_saudacao():
    out = scrub_conversation(_conv(("lead", "Beltrana Souza Lima"), ("lead", "Bom Dia"), ("lead", "Quero Marcar")))
    assert out == ["[NOME]", "Bom Dia", "Quero Marcar"]


def test_lista_de_nomes_sem_caixa_e_preserva_sao_paulo():
    cont = Counter()
    rx = build_names_re(["Rafael", "Paulo"])
    out = scrub_conversation(_conv(("atendente", "perfeito rafael .. esta agendado"),
                                   ("lead", "Amanhã estarei em São Paulo"),
                                   ("lead", "Meu nome Ciclano quero saber o valor")), rx, cont)
    assert out[0] == "perfeito [NOME] .. esta agendado"
    assert out[1] == "Amanhã estarei em São Paulo"
    assert out[2].startswith("Meu nome [NOME]")
    assert cont["lista_de_nomes"] == 1


def test_nao_altera_marcadores_existentes():
    rx = build_names_re(["Midia"])
    out = scrub_conversation(_conv(("lead", "[MIDIA]"), ("lead", "Meu nome [NOME]")), rx)
    assert out == ["[MIDIA]", "Meu nome [NOME]"]
