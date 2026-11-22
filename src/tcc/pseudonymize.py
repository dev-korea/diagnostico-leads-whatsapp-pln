"""Pseudonimização (LGPD): identificadores viram HMAC; PII no texto vira marcadores.

Princípios
- O telefone/título do chat NUNCA é salvo: só o HMAC-SHA256 com chave secreta (TCC_SECRET),
  mantida fora do repositório. Sem a chave não há como reverter nem recompor a tabela.
- Remetentes viram papéis ('lead' / 'atendente'); nomes próprios conhecidos viram [NOME].
- Regex cobre: e-mail, URL, CPF, CNPJ, telefone BR, CEP, cartão, @usuário, endereço, nome
  autodeclarado ("meu nome é X", "aqui é a X").
- Nada substitui a revisão humana: `review_candidates` lista nomes próprios que sobraram.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .parser import ParsedMessage

_ACCENTS = {
    "a": "aàáâãä", "e": "eèéêë", "i": "iìíîï", "o": "oòóôõö", "u": "uùúûü", "c": "cç", "n": "nñ",
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _name_pattern(name: str) -> str:
    out = []
    for ch in _strip_accents(name).lower():
        out.append(f"[{_ACCENTS[ch]}]" if ch in _ACCENTS else re.escape(ch))
    return "".join(out)


def _luhn(s: str) -> bool:
    d = [int(c) for c in s if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    tot = 0
    for i, x in enumerate(reversed(d)):
        if i % 2:
            x = x * 2 - 9 if x * 2 > 9 else x * 2
        tot += x
    return tot % 10 == 0


def hash_identifier(value: str, secret: str, n_chars: int = 24) -> str:
    """HMAC-SHA256 do identificador normalizado (dígitos para telefone; minúsculas para texto)."""
    digits = re.sub(r"\D", "", value)
    norm = digits if len(digits) >= 8 else _strip_accents(value).strip().lower()
    return hmac.new(secret.encode(), norm.encode(), hashlib.sha256).hexdigest()[:n_chars]


# ---- padrões (ordem importa: mais específicos primeiro) ---------------------------------
_P = re.compile
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("[EMAIL]", _P(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("[LINK]", _P(r"(?:https?://|www\.)\S+", re.I)),
    ("[CNPJ]", _P(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")),
    ("[CPF]", _P(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("[CARTAO]", _P(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")),   # só mascara se passar no Luhn
    ("[CEP]", _P(r"\b\d{5}-?\d{3}\b")),
    ("[TELEFONE]", _P(r"(?<![\d$])(?:\+?55[\s.-]?)?(?:\(?\d{2}\)?[\s.-]?)?9?\d{4}[\s.-]?\d{4}(?!\d)")),
    ("[USUARIO]", _P(r"(?<![\w.])@[\w.]{3,}")),
    ("[ENDERECO]", _P(
        r"\b(?:rua|r\.|avenida|av\.?|travessa|alameda|praça|estrada|rodovia)\s+"
        r"[\wÀ-ÿ.' -]{2,40}?(?:,?\s*(?:n[ºo°.]*\s*)?\d{1,5})?(?=[,.;\n]|$|\s+(?:bairro|apto|ap\.?|casa))",
        re.I)),
]
# nome autodeclarado: captura o NOME (grupo 2) e mantém o prefixo
_SELF_INTRO = _P(
    r"\b((?i:meu nome (?:é|e)|me chamo|aqui (?:é|e) (?:a |o )?|sou (?:a |o )|"
    r"é a |é o |falando com (?:a |o )?)\s*)([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+)?)")


@dataclass
class Pseudonymizer:
    secret: str
    attendant_names: list[str] = field(default_factory=list)
    known_names: list[str] = field(default_factory=list)   # nomes a mascarar (cliente, familiares…)
    extra_stoplist: set[str] = field(default_factory=set)

    def __post_init__(self):
        if not self.secret or len(self.secret) < 16:
            raise ValueError("Defina TCC_SECRET (≥16 caracteres) fora do repositório.")
        self._att_norm = {_strip_accents(n).lower().strip() for n in self.attendant_names}
        self._name_re: Optional[re.Pattern] = None
        self._rebuild()

    # nomes são mascarados por palavra (primeiro nome, sobrenome…), com tolerância a acentos
    def add_names(self, names: Iterable[str]) -> None:
        for n in names:
            for part in re.split(r"\s+", n.strip()):
                if len(part) >= 3 and part.lower() not in self.extra_stoplist:
                    self.known_names.append(part)
        self._rebuild()

    def _rebuild(self) -> None:
        words = sorted({w for n in (self.known_names + self.attendant_names)
                        for w in re.split(r"\s+", n.strip()) if len(w) >= 3}, key=len, reverse=True)
        self._name_re = (_P(r"(?<![\wÀ-ÿ])(?:" + "|".join(_name_pattern(w) for w in words) + r")(?![\wÀ-ÿ])", re.I)
                         if words else None)

    # ---- API ----------------------------------------------------------------------------
    def role_of(self, sender: Optional[str]) -> str:
        if sender is None:
            return "sistema"
        return "atendente" if _strip_accents(sender).lower().strip() in self._att_norm else "lead"

    def scrub(self, text: str) -> tuple[str, int]:
        n = 0

        def sub(pat: re.Pattern, repl, s: str) -> str:
            nonlocal n
            s, k = pat.subn(repl, s)
            n += k
            return s

        for tag, pat in PATTERNS:
            if tag == "[CARTAO]":
                def _card(m, tag=tag):
                    nonlocal n
                    if _luhn(m.group(0)):
                        n += 1
                        return tag
                    return m.group(0)
                text = pat.sub(_card, text)
            else:
                text = sub(pat, tag, text)
        text = sub(_SELF_INTRO, lambda m: m.group(1) + "[NOME]", text)
        if self._name_re is not None:
            text = sub(self._name_re, "[NOME]", text)
        return text, n

    def pseudonymize_messages(self, msgs: list[ParsedMessage], chat_title: str):
        """Retorna (lead_id, lista de dicts prontos para o banco)."""
        lead_id = hash_identifier(chat_title, self.secret)
        # título do chat costuma ser o nome do lead -> entra na lista de nomes a mascarar
        if not re.fullmatch(r"[\d\s+()-]+", chat_title):
            self.add_names([chat_title])
        senders = {m.sender for m in msgs if m.sender}
        non_att = [s for s in senders if self.role_of(s) == "lead"]
        if len(non_att) > 1:
            raise ValueError(
                f"Mais de um remetente não-atendente ({len(non_att)}): parece grupo. "
                "Exporte apenas conversas individuais ou informe os atendentes.")
        out = []
        for m in msgs:
            role = self.role_of(m.sender)
            text, k = ("", 0) if m.is_media or m.is_deleted else self.scrub(m.text)
            if m.is_media:
                text = "[MIDIA]"
            elif m.is_deleted:
                text = "[APAGADA]"
            elif role == "sistema":
                text, k = "[SISTEMA]", 0   # não guardamos texto de sistema (pode conter números/nomes)
            out.append({"ts": m.ts, "sender_role": role, "text": text,
                        "is_media": int(m.is_media), "n_redactions": k})
        return lead_id, out

    def review_candidates(self, texts: Iterable[str], top: int = 40) -> list[tuple[str, int]]:
        """Palavras capitalizadas no meio de frases (possíveis nomes que escaparam)."""
        c: Counter = Counter()
        stop = {"Whatsapp", "WhatsApp", "Instagram", "Google", "Pix", "Bom", "Boa", "Olá", "Ola", "Oi"}
        stop |= self.extra_stoplist
        for t in texts:
            for m in re.finditer(r"(?<=[a-zà-ÿ,;] )([A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-zà-ÿ]{2,})", t):
                w = m.group(1)
                if w not in stop and not w.startswith("["):
                    c[w] += 1
        return c.most_common(top)
