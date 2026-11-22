from datetime import datetime
import pytest
from tcc.parser import parse_text, detect_format, infer_dayfirst, chat_title_from_filename

ANDROID = """12/03/2024 14:35 - As mensagens e ligações são protegidas com a criptografia de ponta a ponta.
12/03/2024 14:36 - Carla Souza: Oi, tudo bem?
Queria saber o valor
12/03/2024 14:40 - Clínica: Olá! R$ 200
13/03/2024 09:01 - Carla Souza: <Mídia oculta>
13/03/2024 09:02 - Clínica: Esta mensagem foi apagada
"""
IPHONE = """‎[12/03/2024 14:36:10] Carla Souza: Oi
[12/03/2024 14:40:00] Clínica: Olá: tudo bem? Valor: R$ 200
[12/03/2024 14:41:00] Carla Souza: ‎imagem ocultada
"""


def test_android_basic_and_multiline():
    msgs, fmt = parse_text(ANDROID)
    assert fmt == "android" and len(msgs) == 5
    assert msgs[0].is_system and msgs[0].sender is None
    assert msgs[1].sender == "Carla Souza" and msgs[1].text == "Oi, tudo bem?\nQueria saber o valor"
    assert msgs[1].ts == datetime(2024, 3, 12, 14, 36)
    assert msgs[3].is_media and msgs[4].is_deleted


def test_iphone_and_colon_in_text():
    msgs, fmt = parse_text(IPHONE)
    assert fmt == "iphone" and len(msgs) == 3
    assert msgs[1].sender == "Clínica" and msgs[1].text == "Olá: tudo bem? Valor: R$ 200"
    assert msgs[2].is_media
    assert msgs[0].ts.second == 10


def test_ampm_and_two_digit_year():
    msgs, _ = parse_text("12/03/24, 2:35 PM - Ana: oi\n12/03/24, 12:05 AM - Ana: boa")
    assert msgs[0].ts == datetime(2024, 3, 12, 14, 35) and msgs[1].ts == datetime(2024, 3, 12, 0, 5)


def test_narrow_nbsp_before_ampm():
    msgs, _ = parse_text("3/12/24, 2:35 PM - Ana: oi", dayfirst=False)
    assert msgs[0].ts == datetime(2024, 3, 12, 14, 35)


def test_dayfirst_inference():
    assert infer_dayfirst("25/12/2024 10:00 - A: x") is True
    assert infer_dayfirst("12/25/2024 10:00 - A: x") is False


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        parse_text("isto não é uma exportação")


def test_title_from_filename():
    assert chat_title_from_filename("/x/Conversa do WhatsApp com Carla Souza.txt") == "Carla Souza"
    assert chat_title_from_filename("WhatsApp Chat with +55 53 99999-0000.txt") == "+55 53 99999-0000"
