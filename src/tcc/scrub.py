"""Segunda passada de pseudonimização, direto no banco (não precisa do TCC_SECRET).

A ingestão mascara o que as regex enxergam mensagem a mensagem. Sobram nomes que só o CONTEXTO revela:
- a resposta do cliente logo após o atendente pedir "nome completo";
- fala do cliente que é só um nome próprio;
- "meu nome X" / "me chamo X" sem o verbo "é";
- vocativos do atendente em minúsculas ("perfeito rafael", "certo marcia .. esta agendado").

Esta passada corrige o texto em `messages.text` (os IDs de mensagem e de conversa não mudam, então rótulos,
divisões e predições continuam válidos). Rode SEMPRE antes de enviar texto para fora da máquina
(Supabase, Colab, API de LLM).

Uso:
  python -m tcc.scrub candidates --db data/processed/tcc.db            # lista possíveis nomes para revisão humana
  python -m tcc.scrub apply --db data/processed/tcc.db --names-file data/processed/nomes_extra.txt --dry-run
  python -m tcc.scrub apply --db data/processed/tcc.db --names-file data/processed/nomes_extra.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable

from .ingest import connect
from .pseudonymize import _name_pattern

TAG = "[NOME]"
_PALAVRA = r"[A-ZÀ-Ú][A-Za-zÀ-ú'’]+"
# fala que é só um nome próprio (2 a 6 palavras capitalizadas)
_RE_SO_NOME = re.compile(rf"^\s*({_PALAVRA}(?: (?:d[aeo]s?|e) {_PALAVRA}| {_PALAVRA}){{1,5}})\s*$")
_NAO_NOME = re.compile(
    r"(?i)\b(bom|boa|dia|tarde|noite|tudo|bem|obrigad[oa]|primeira|vez|vou|sim|não|nao|ok|certo|valor|"
    r"consulta|sessão|sessao|quero|queria|gostaria|oi|olá|ola|por favor|segunda|terça|quarta|quinta|sexta|"
    r"sábado|sabado|domingo|manhã|manha|pode|ser|marcar|agendar|avaliação|avaliacao|quiropraxia|dor|coluna)\b")
_RE_PEDE_NOME = re.compile(r"nome completo|qual (?:é |e )?(?:o )?seu nome|com quem (?:eu )?falo", re.I)
_RE_MEU_NOME = re.compile(
    r"(?i)\b((?:meu nome|me chamo|aqui é|aqui e|sou [ao])[ \t]+(?:é[ \t]+|e[ \t]+)?)(?!\[NOME\])"
    r"(\w+(?:[ \t]+(?:d[aeo]s?[ \t]+)?[A-ZÀ-Ú]\w+)*)")
# vocativo do atendente: palavra logo após um marcador de confirmação/saudação
_RE_VOCATIVO = re.compile(
    r"(?i)\b(?:ol[aá]|oi|certo|perfeito|entendo|entendi|agendad[oa]|obrigad[oa]|magina|bom dia|boa tarde|boa noite)"
    r"[ \t]+([A-Za-zÀ-ú]{3,})\b")
_STOP_VOC = set("""
que para por com sem uma uns umas ele ela voce você vc seu sua meu minha esta está esse essa isso aqui
bom boa dia tarde noite tudo bem sim nao não mas pois entao então qual quando como onde pode posso podemos
vou vamos fico fica ficou ficarei seria sera será tenho tem temos hoje amanha amanhã agora depois ainda
obrigado obrigada certo claro perfeito magina gostaria quer quero qualquer caso consigo consegue desculpe
pela pelo nos nas dos das mesmo assim tambem também muito pouco mais menos ate até so só segunda terça terca
quarta quinta sexta sabado sábado domingo senhor senhora doutor doutora dra ola olá
""".split())


_RE_VIZINHO = re.compile(r"([A-ZÀ-Ú][a-zà-ú'’]{2,}) (?=\[NOME\])|(?<=\[NOME\]) ([A-ZÀ-Ú][a-zà-ú'’]{2,})")


def _vizinho(m: re.Match) -> str:
    w = m.group(1) or m.group(2)
    if w.lower() in _STOP_VOC or _NAO_NOME.search(w):
        return m.group(0)
    return (TAG + " ") if m.group(1) else (" " + TAG)


def build_names_re(names: Iterable[str]):
    words = sorted({w for n in names for w in re.split(r"\s+", n.strip()) if len(w) >= 3}, key=len, reverse=True)
    if not words:
        return None
    return re.compile(r"(?<![\wÀ-ÿ\[])(?<!São )(?<!Sao )(?<!são )(?<!sao )(?:" + "|".join(_name_pattern(w) for w in words) + r")(?![\wÀ-ÿ\]])", re.I)


def scrub_conversation(msgs: list[dict], names_re=None, cont: Counter | None = None) -> list[str]:
    """Recebe as mensagens de UMA conversa em ordem ({sender_role, text}) e devolve os textos corrigidos."""
    cont = cont if cont is not None else Counter()
    out, pedir = [], False
    for m in msgs:
        t, role = m["text"] or "", m["sender_role"]
        if role == "lead" and t and not t.startswith("["):
            m_so = _RE_SO_NOME.match(t) if "\n" not in t.strip() else None
            if pedir:
                t = TAG; cont["apos_pedido_de_nome"] += 1; pedir = False
            elif m_so and not _NAO_NOME.search(m_so.group(1)):
                t = TAG; cont["fala_so_nome"] += 1
        elif role == "atendente":
            pedir = bool(_RE_PEDE_NOME.search(t))
        t, k = _RE_MEU_NOME.subn(lambda mm: mm.group(1) + TAG, t)
        cont["meu_nome"] += k
        if names_re is not None:
            t, k = names_re.subn(TAG, t)
            cont["lista_de_nomes"] += k
        # sobrenome/nome capitalizado colado a um [NOME] (ex.: "Fulana [NOME] [NOME] Sobrenome")
        for _ in range(4):
            t, k = _RE_VIZINHO.subn(_vizinho, t)
            cont["vizinho_de_nome"] += k
            if not k:
                break
        out.append(t)
    return out


def candidates(conn: sqlite3.Connection, top: int = 80) -> list[tuple[str, int]]:
    """Possíveis nomes que sobraram: vocativos do atendente e palavras capitalizadas no meio de frases."""
    c: Counter = Counter()
    for r in conn.execute("SELECT sender_role, text FROM messages WHERE text IS NOT NULL"):
        t = r["text"]
        if r["sender_role"] == "atendente":
            for m in _RE_VOCATIVO.finditer(t):
                w = m.group(1)
                if w.lower() not in _STOP_VOC:
                    c[w.lower()] += 1
        for m in re.finditer(r"(?<=[a-zà-ÿ,;] )([A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-zà-ÿ]{2,})", t):
            if m.group(1).lower() not in _STOP_VOC:
                c[m.group(1).lower()] += 1
    return c.most_common(top)


def apply(conn: sqlite3.Connection, names: Iterable[str] = (), dry_run: bool = False) -> dict:
    names_re = build_names_re(names)
    cont: Counter = Counter()
    changed = 0
    convs = [r[0] for r in conn.execute("SELECT conversation_id FROM conversations ORDER BY conversation_id")]
    updates = []
    for cid in convs:
        msgs = [dict(m) for m in conn.execute(
            "SELECT message_id, sender_role, text FROM messages WHERE conversation_id=? ORDER BY ts, rowid", (cid,))]
        new = scrub_conversation(msgs, names_re, cont)
        for m, t in zip(msgs, new):
            if t != (m["text"] or ""):
                updates.append((t, m["message_id"]))
    changed = len(updates)
    if not dry_run and updates:
        with conn:
            conn.executemany("UPDATE messages SET text=?, n_redactions=n_redactions+1 WHERE message_id=?", updates)
    return {"conversas": len(convs), "mensagens_alteradas": changed, "dry_run": dry_run,
            "por_regra": {k: v for k, v in cont.items() if v}}


def read_names_file(path: str | None) -> list[str]:
    if not path:
        return []
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tcc.scrub")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("candidates"); p.add_argument("--db", required=True); p.add_argument("--top", type=int, default=80)
    p = sub.add_parser("apply"); p.add_argument("--db", required=True)
    p.add_argument("--names", default="", help="nomes extras, separados por vírgula")
    p.add_argument("--names-file", default=None, help="arquivo com um nome por linha (fora do versionamento)")
    p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    conn = connect(a.db)
    if a.cmd == "candidates":
        for w, n in candidates(conn, a.top):
            print(f"{n:4d}  {w}")
    else:
        names = [x.strip() for x in a.names.split(",") if x.strip()] + read_names_file(a.names_file)
        print(json.dumps(apply(conn, names, a.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
