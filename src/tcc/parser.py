"""Parser de exportações de conversa do WhatsApp (.txt, "sem mídia").

Formatos suportados (pt-BR e en):
  Android : ``12/03/2024 14:35 - Nome: texto``  (também ``12/03/24 2:35 PM - ...``)
  iPhone  : ``[12/03/2024 14:35:22] Nome: texto``  (também com AM/PM)
Mensagens de várias linhas são unidas à mensagem anterior. Mensagens de sistema
(sem ``Nome:``) recebem sender=None. Datas são interpretadas como DIA/MÊS/ANO
por padrão (``dayfirst=True``); use ``dayfirst=False`` para exportações em MM/DD.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

# Caracteres invisíveis inseridos pelo WhatsApp (LRM, espaço fino "narrow nbsp" etc.)
_INVISIBLE = dict.fromkeys(map(ord, "‎‏‪‫‬‭‮﻿"), None)
_SPACES = {0x202F: " ", 0x00A0: " "}

_DATE = r"(\d{1,2})/(\d{1,2})/(\d{2,4})"
_TIME = r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([APap]\.?[Mm]\.?)?"
# Android: "12/03/2024 14:35 - resto"  (aceita vírgula após a data: "12/03/2024, 14:35 - ")
_RE_ANDROID = re.compile(rf"^{_DATE},?\s+{_TIME}\s+-\s+(.*)$")
# iPhone: "[12/03/2024 14:35:22] resto" (aceita vírgula)
_RE_IPHONE = re.compile(rf"^\[{_DATE},?\s+{_TIME}\]\s+(.*)$")

_MEDIA_MARKERS = (
    "<mídia oculta>", "<midia oculta>", "<media omitted>", "imagem ocultada",
    "áudio ocultado", "audio ocultado", "vídeo ocultado", "video ocultado",
    "figurinha omitida", "sticker omitted", "image omitted", "audio omitted",
    "video omitted", "documento omitido", "(arquivo anexado)", "<anexado:",
    "<attached:", "gif omitido", "contato omitido",
)
_DELETED_MARKERS = ("esta mensagem foi apagada", "você apagou esta mensagem",
                    "this message was deleted", "you deleted this message",
                    "mensagem apagada")
_SYSTEM_HINTS = (
    "mensagens e ligações são protegidas", "as mensagens e chamadas são protegidas",
    "messages and calls are end-to-end encrypted", "criou o grupo", "created group",
    "mudou o número", "changed their phone number", "mudou seu código de segurança",
    "security code changed", "chamada de voz perdida", "chamada de vídeo perdida",
    "missed voice call", "missed video call", "você foi adicionado", "saiu", "entrou usando",
)


@dataclass
class ParsedMessage:
    ts: datetime
    sender: Optional[str]          # None = mensagem de sistema
    text: str
    is_media: bool = False
    is_deleted: bool = False
    is_system: bool = False


def _clean(line: str) -> str:
    return line.translate(_INVISIBLE).translate(_SPACES).rstrip("\r\n")


def _to_datetime(d: str, m: str, y: str, hh: str, mm: str, ss: Optional[str],
                 ampm: Optional[str], dayfirst: bool) -> datetime:
    day, month = (int(d), int(m)) if dayfirst else (int(m), int(d))
    year = int(y)
    if year < 100:
        year += 2000
    hour = int(hh)
    if ampm:
        pm = ampm.lower().startswith("p")
        hour = hour % 12 + (12 if pm else 0)
    return datetime(year, month, day, hour, int(mm), int(ss or 0))


def detect_format(lines: Iterable[str]) -> str:
    """Retorna 'android', 'iphone' ou 'unknown' olhando as primeiras linhas úteis."""
    score = {"android": 0, "iphone": 0}
    for i, raw in enumerate(lines):
        if i > 200:
            break
        line = _clean(raw)
        if _RE_IPHONE.match(line):
            score["iphone"] += 1
        elif _RE_ANDROID.match(line):
            score["android"] += 1
    if score["android"] == score["iphone"] == 0:
        return "unknown"
    return max(score, key=score.get)


def _split_sender(rest: str) -> tuple[Optional[str], str]:
    """Separa 'Nome: texto'. Mensagens de sistema não possuem ': ' após um nome curto."""
    if ": " not in rest:
        return None, rest
    name, text = rest.split(": ", 1)
    # nome de remetente é curto e não contém quebra; evita falso-positivo em frases de sistema
    if len(name) > 60:
        return None, rest
    low = rest.lower()
    if any(h in low for h in _SYSTEM_HINTS) and ":" not in name:
        return None, rest
    return name.strip(), text


def parse_text(content: str, dayfirst: bool = True) -> tuple[list[ParsedMessage], str]:
    """Faz o parsing do conteúdo completo do .txt. Retorna (mensagens, formato)."""
    lines = content.splitlines()
    fmt = detect_format(lines)
    if fmt == "unknown":
        raise ValueError("Formato de exportação não reconhecido (nem Android nem iPhone).")
    regex = _RE_IPHONE if fmt == "iphone" else _RE_ANDROID
    msgs: list[ParsedMessage] = []
    for raw in lines:
        line = _clean(raw)
        m = regex.match(line)
        if m:
            d, mo, y, hh, mm, ss, ampm, rest = m.groups()
            try:
                ts = _to_datetime(d, mo, y, hh, mm, ss, ampm, dayfirst)
            except ValueError:
                # data inválida (ex.: formato MM/DD forçado errado) -> trata como continuação
                if msgs:
                    msgs[-1].text += "\n" + line
                continue
            sender, text = _split_sender(rest)
            msgs.append(ParsedMessage(ts=ts, sender=sender, text=text, is_system=sender is None))
        elif msgs:
            msgs[-1].text += "\n" + line  # continuação de mensagem multilinha
        # linhas antes da primeira mensagem são ignoradas
    for msg in msgs:
        low = msg.text.strip().lower()
        msg.is_media = any(low.startswith(k) or k in low for k in _MEDIA_MARKERS) and len(low) < 80
        msg.is_deleted = any(k in low for k in _DELETED_MARKERS) and len(low) < 80
        msg.text = msg.text.strip()
    return msgs, fmt


def parse_file(path: str, dayfirst: bool = True) -> tuple[list[ParsedMessage], str]:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        return parse_text(fh.read(), dayfirst=dayfirst)


def infer_dayfirst(content: str) -> bool:
    """Heurística: se algum primeiro campo > 12 => DD/MM; se algum segundo > 12 => MM/DD."""
    first_gt12 = second_gt12 = False
    for raw in content.splitlines()[:2000]:
        line = _clean(raw)
        m = _RE_IPHONE.match(line) or _RE_ANDROID.match(line)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            first_gt12 |= a > 12
            second_gt12 |= b > 12
    if second_gt12 and not first_gt12:
        return False
    return True


def chat_title_from_filename(filename: str) -> str:
    """'Conversa do WhatsApp com Fulano.txt' / 'WhatsApp Chat with Fulano.txt' -> 'Fulano'."""
    base = re.sub(r"\.txt$", "", filename.split("/")[-1], flags=re.I)
    base = re.sub(r"^(conversa do whatsapp com|whatsapp chat with)\s+", "", base, flags=re.I)
    return base.strip()
