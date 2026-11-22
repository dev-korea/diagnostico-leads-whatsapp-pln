"""Pré-rotulagem por LLM para ACELERAR a anotação humana (nunca substitui o gabarito).

Desenho que preserva o rigor (declare na Metodologia):
- um subconjunto CEGO (por lead, ~25 %) NÃO recebe sugestão: vai em branco para a1 e a2 rotularem de forma
  independente. A concordância (kappa) e o conjunto de teste devem sair desse subconjunto;
- só o anotador a1 recebe sugestões nas demais conversas; o a2 nunca as vê;
- `demora_resposta` NÃO vem do modelo: é aplicada por regra do guia (qualquer tempo > 30 min, ou sem resposta);
- toda sugestão é validada pelas mesmas regras do `import-labels`; sugestão inválida fica em branco;
- cache em disco, temperatura 0, versão do modelo registrada no CSV (coluna `pre_rotulo`).

Uso:
  python -m tcc.prelabel --db data/processed/tcc.db --provider anthropic --model <id-exato> \
      --out-a1 rotular_a1.csv --out-a2 rotular_a2_cego.csv --blind-frac 0.25 --seed 13
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
from typing import Optional

from .experiments.llm import Cache, Provider, make_provider
from .ingest import connect, init_db
from .labels import FIELDS, NONE, validate_row
from .segment import conversation_text
from .taxonomy import DEFINICOES, GARGALOS, OBJECOES

DEMORA_MIN = 30.0
PRE_COL = "pre_rotulo"
MODEL_FIELDS = FIELDS + [PRE_COL]

SYSTEM = f"""Você apoia a anotação de conversas de WhatsApp de uma clínica de quiropraxia no Brasil.
O texto está pseudonimizado ([NOME], [TELEFONE]... são marcadores). Rotule SÓ com o que aparece na conversa.

Escopo: em_escopo ({DEFINICOES['em_escopo']}) ou fora_de_escopo ({DEFINICOES['fora_de_escopo']}).
Funil (somente se em_escopo; null se fora_de_escopo). Classifique pelo ponto MAIS AVANÇADO alcançado nesta conversa:
- sem_dialogo: {DEFINICOES['sem_dialogo']}
- informativo: {DEFINICOES['informativo']}
- intencao: {DEFINICOES['intencao']}
- agendado: {DEFINICOES['agendado']}
Objeções (só falas do CLIENTE; lista, vazia se nenhuma): {', '.join(OBJECOES)}.
  Pergunta simples não é objeção. "Vou pensar" sem motivo = só adiamento; com motivo, marque o motivo e adiamento.
  horario_disponibilidade = agenda do CLIENTE incompatível. servico_nao_oferecido = pede algo que a clínica não faz.
Gargalos de atendimento (lista, vazia se nenhum): sem_followup, info_incompleta, sem_vaga.
- sem_followup: terminou sem desfecho e o atendente não retomou. info_incompleta: pergunta do cliente ficou sem resposta.
- sem_vaga: a clínica não tinha horário compatível com o pedido do cliente.
para_terceiro: "sim" se quem escreve contata em nome de outra pessoa; senão null.

Responda SOMENTE com JSON válido, sem texto extra:
{{"escopo": "...", "funil": "... ou null", "objecao": [...], "gargalo": [...], "para_terceiro": "sim ou null"}}"""


def _demora(first_min, median_min) -> bool:
    """Regra do guia: qualquer tempo > 30 min; se ambos vazios (atendente nunca respondeu), marca."""
    vals = [v for v in (first_min, median_min) if v not in ("", None)]
    if not vals:
        return True
    return any(float(v) > DEMORA_MIN for v in vals)


def parse_suggestion(raw: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def to_row_values(obj: dict, first_min, median_min) -> Optional[dict]:
    """Converte a resposta do modelo nos valores do CSV; None se não passar na validação do guia."""
    esc = obj.get("escopo")
    funil = obj.get("funil") if esc == "em_escopo" else ""
    funil = "" if funil in (None, "null") else funil
    objs = [x for x in (obj.get("objecao") or []) if isinstance(x, str)]
    # demora_resposta vem da regra do guia, nunca do modelo
    garg = [x for x in (obj.get("gargalo") or []) if isinstance(x, str) and x in GARGALOS and x != "demora_resposta"]
    if _demora(first_min, median_min):
        garg.append("demora_resposta")
    ter = obj.get("para_terceiro")
    row = {"escopo": esc or "", "funil": funil or "",
           "objecao": ";".join(sorted(set(objs))) or NONE,
           "gargalo": ";".join(sorted(set(garg))) or NONE,
           "para_terceiro": "sim" if ter == "sim" else ""}
    try:
        validate_row(row)
    except ValueError:
        return None
    return row


def suggest(provider: Provider, cache: Cache, text: str, first_min, median_min, retries: int = 2):
    user = (f"primeira_resposta_min: {first_min if first_min != '' else 'sem resposta'}\n"
            f"mediana_resposta_min: {median_min if median_min != '' else 'sem resposta'}\n\n"
            f"### Conversa\n{text}\n### Resposta\n")
    k = Cache.key(provider.name, provider.model, SYSTEM, user)
    for _ in range(retries + 1):
        raw = cache.get(k)
        if raw is None:
            raw = provider.complete(SYSTEM, user)
        obj = parse_suggestion(raw)
        vals = to_row_values(obj, first_min, median_min) if obj else None
        if vals is not None:
            cache.put(k, raw)
            return vals
    return None


def pick_blind_leads(conn: sqlite3.Connection, frac: float, seed: int) -> set[str]:
    leads = sorted(r[0] for r in conn.execute("SELECT lead_id FROM leads"))
    n = max(1, round(len(leads) * frac))
    return set(random.Random(seed).sample(leads, n))


def run(conn: sqlite3.Connection, provider: Provider, cache: Cache, out_a1: str, out_a2: str,
        blind_frac: float = 0.25, seed: int = 13, limit: Optional[int] = None) -> dict:
    blind = pick_blind_leads(conn, blind_frac, seed)
    convs = conn.execute("SELECT conversation_id, lead_id, seq_in_lead, first_response_sec, median_response_sec "
                         "FROM conversations").fetchall()
    order = [dict(c) for c in convs]
    random.Random(seed).shuffle(order)
    stats = {"total": len(order), "cegas": 0, "sugeridas": 0, "invalidas": 0}
    rows_a1, rows_a2 = [], []
    done = 0
    for c in order:
        cid = c["conversation_id"]
        msgs = [dict(m) for m in conn.execute(
            "SELECT sender_role, text FROM messages WHERE conversation_id=? ORDER BY ts, rowid", (cid,))]
        first = "" if c["first_response_sec"] is None else round(c["first_response_sec"] / 60, 1)
        med = "" if c["median_response_sec"] is None else round(c["median_response_sec"] / 60, 1)
        base = {"conversation_id": cid, "seq_in_lead": c["seq_in_lead"], "primeira_resposta_min": first,
                "mediana_resposta_min": med, "texto": conversation_text(msgs),
                "escopo": "", "funil": "", "objecao": "", "gargalo": "", "desfecho": "", "para_terceiro": "",
                "observacao": "", PRE_COL: ""}
        if c["lead_id"] in blind:
            stats["cegas"] += 1
            base[PRE_COL] = "cega"
            rows_a1.append(dict(base)); rows_a2.append(dict(base))
            continue
        if limit is not None and done >= limit:
            rows_a1.append(base); continue
        vals = suggest(provider, cache, base["texto"], first, med)
        done += 1
        if vals is None:
            stats["invalidas"] += 1
            base["observacao"] = "pré-rótulo LLM inválido: rotular do zero"
            base[PRE_COL] = "falhou"
        else:
            stats["sugeridas"] += 1
            base.update(vals)
            base[PRE_COL] = f"sugerido:{provider.name}:{provider.model}"
        rows_a1.append(base)
    for path, rows in ((out_a1, rows_a1), (out_a2, rows_a2)):
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=MODEL_FIELDS); w.writeheader(); w.writerows(rows)
    stats["linhas_a1"], stats["linhas_a2_cego"] = len(rows_a1), len(rows_a2)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tcc.prelabel")
    ap.add_argument("--db", required=True)
    ap.add_argument("--provider", required=True, choices=["anthropic", "openai", "mock"])
    ap.add_argument("--model", default=None, help="id EXATO do modelo (obrigatório, exceto mock)")
    ap.add_argument("--out-a1", required=True, help="CSV do anotador a1 (com sugestões, exceto no subconjunto cego)")
    ap.add_argument("--out-a2", required=True, help="CSV do anotador a2 (só o subconjunto cego, em branco)")
    ap.add_argument("--blind-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=None, help="só N sugestões (teste de custo)")
    ap.add_argument("--cache", default="results/prelabel_cache.sqlite")
    a = ap.parse_args(argv)
    conn = connect(a.db); init_db(conn)
    print(json.dumps(run(conn, make_provider(a.provider, a.model), Cache(a.cache), a.out_a1, a.out_a2,
                         a.blind_frac, a.seed, a.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
