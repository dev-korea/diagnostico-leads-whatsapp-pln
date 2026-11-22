"""Rotulagem: exportar para planilha, importar rótulos validados, concordância e ouro."""
from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict
from itertools import combinations
from typing import Optional

from sklearn.metrics import cohen_kappa_score

from .segment import conversation_text
from .taxonomy import DIMENSIONS, FUNIL, MULTILABEL, OBJECOES, GARGALOS, derive_qualificacao

NONE = "nenhuma"
# v1.1: o anotador preenche `funil`; `qualificacao` é derivada (não está no CSV exportado).
FIELDS = ["conversation_id", "seq_in_lead", "primeira_resposta_min", "mediana_resposta_min", "texto", "escopo",
          "funil", "objecao", "gargalo", "desfecho", "para_terceiro", "observacao"]
INPUT_KEYS = ("escopo", "funil", "qualificacao", "objecao", "gargalo")


def export_for_annotation(conn: sqlite3.Connection, path: str, only_unlabeled_by: Optional[str] = None,
                          shuffle_seed: Optional[int] = 13) -> int:
    """Gera CSV para o anotador. Ordem embaralhada (evita viés de ordem/lead consecutivo)."""
    import random
    rows = conn.execute("SELECT conversation_id, seq_in_lead, first_response_sec, median_response_sec FROM conversations").fetchall()
    ids = [(r["conversation_id"], r["seq_in_lead"]) for r in rows]
    mins = {r["conversation_id"]: tuple("" if v is None else round(v / 60, 1) for v in
                                        (r["first_response_sec"], r["median_response_sec"])) for r in rows}
    if only_unlabeled_by:
        done = {r[0] for r in conn.execute("SELECT DISTINCT conversation_id FROM labels WHERE annotator=?",
                                           (only_unlabeled_by,))}
        ids = [x for x in ids if x[0] not in done]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(ids)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        for cid, seq in ids:
            msgs = [dict(m) for m in conn.execute(
                "SELECT sender_role, text FROM messages WHERE conversation_id=? ORDER BY ts, rowid", (cid,))]
            w.writerow({"conversation_id": cid, "seq_in_lead": seq, "primeira_resposta_min": mins[cid][0],
                        "mediana_resposta_min": mins[cid][1], "texto": conversation_text(msgs),
                        "escopo": "", "funil": "", "objecao": "", "gargalo": "", "desfecho": "",
                        "para_terceiro": "", "observacao": ""})
    return len(ids)


def _split_multi(v: str) -> list[str]:
    return [x.strip() for x in v.replace(",", ";").split(";") if x.strip()]


def validate_row(row: dict) -> dict[str, list[str]]:
    """Valida uma linha preenchida; devolve {dimensão: [valores]} ou levanta ValueError."""
    esc = row.get("escopo", "").strip()
    out: dict[str, list[str]] = {}
    if esc not in DIMENSIONS["escopo"]:
        raise ValueError(f"escopo inválido: {esc!r}")
    out["escopo"] = [esc]
    f = row.get("funil", "").strip()
    q = row.get("qualificacao", "").strip()
    if esc == "em_escopo":
        if f not in FUNIL:
            raise ValueError(f"funil obrigatório e válido para em_escopo: {f!r}")
        out["funil"] = [f]
        out["qualificacao"] = [derive_qualificacao(f)]          # derivada: única fonte da verdade
        if q and q != out["qualificacao"][0]:
            raise ValueError(f"qualificacao {q!r} contradiz o funil {f!r} (derivada: {out['qualificacao'][0]})")
    elif f or q:
        raise ValueError("funil e qualificacao devem ficar vazios quando fora_de_escopo")
    des = row.get("desfecho", "").strip()
    if des:
        if des not in DIMENSIONS["desfecho"]:
            raise ValueError(f"desfecho inválido: {des!r}")
        if out.get("funil") != ["agendado"]:
            raise ValueError("desfecho 'desmarcou' só vale com funil = agendado")
        out["desfecho"] = [des]
    ter = row.get("para_terceiro", "").strip()
    if ter:
        if ter not in DIMENSIONS["para_terceiro"]:
            raise ValueError(f"para_terceiro inválido: {ter!r}")
        out["para_terceiro"] = [ter]
    for dim, allowed in (("objecao", OBJECOES), ("gargalo", GARGALOS)):
        vals = _split_multi(row.get(dim, ""))
        if not vals:
            raise ValueError(f"{dim}: preencha valores ou 'nenhuma'")
        if vals == [NONE]:
            out[dim] = [NONE]
            continue
        bad = [v for v in vals if v not in allowed]
        if bad or NONE in vals:
            raise ValueError(f"{dim} inválido: {vals}")
        out[dim] = sorted(set(vals))
    return out


def import_labels(conn: sqlite3.Connection, csv_path: str, annotator: str) -> dict:
    """Importa CSV preenchido. Linhas totalmente vazias são ignoradas; erros são listados."""
    ok, skipped, errors = 0, 0, []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            if not any(row.get(k, "").strip() for k in INPUT_KEYS):
                skipped += 1; continue
            try:
                parsed = validate_row(row)
                if not conn.execute("SELECT 1 FROM conversations WHERE conversation_id=?",
                                    (row["conversation_id"],)).fetchone():
                    raise ValueError("conversation_id inexistente")
            except ValueError as e:
                errors.append(f"linha {i}: {e}"); continue
            with conn:
                conn.execute("DELETE FROM labels WHERE conversation_id=? AND annotator=?",
                             (row["conversation_id"], annotator))
                for dim, vals in parsed.items():
                    for v in vals:
                        conn.execute("INSERT INTO labels(conversation_id, annotator, dimension, value) VALUES (?,?,?,?)",
                                     (row["conversation_id"], annotator, dim, v))
            ok += 1
    return {"imported": ok, "skipped_empty": skipped, "errors": errors}


def get_labels(conn: sqlite3.Connection, annotator: str = "adjudicado") -> dict[str, dict[str, object]]:
    """{conversation_id: {'escopo': str, 'qualificacao': str|None, 'objecao': set, 'gargalo': set}}"""
    raw: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in conn.execute("SELECT conversation_id, dimension, value FROM labels WHERE annotator=?", (annotator,)):
        raw[r["conversation_id"]][r["dimension"]].append(r["value"])
    out = {}
    for cid, d in raw.items():
        out[cid] = {
            "escopo": d["escopo"][0] if d.get("escopo") else None,
            "funil": d["funil"][0] if d.get("funil") else None,
            "qualificacao": d["qualificacao"][0] if d.get("qualificacao") else None,
            "desfecho": d["desfecho"][0] if d.get("desfecho") else None,
            "para_terceiro": d["para_terceiro"][0] if d.get("para_terceiro") else None,
            "objecao": {v for v in d.get("objecao", []) if v != NONE},
            "gargalo": {v for v in d.get("gargalo", []) if v != NONE},
        }
    return out


def agreement(conn: sqlite3.Connection, a: str = "a1", b: str = "a2") -> dict:
    """Kappa de Cohen entre dois anotadores. Multirrótulo: kappa binário por rótulo + média."""
    la, lb = get_labels(conn, a), get_labels(conn, b)
    common = sorted(set(la) & set(lb))
    if not common:
        return {"n": 0}
    res: dict = {"n": len(common)}
    res["escopo"] = _kappa([la[c]["escopo"] for c in common], [lb[c]["escopo"] for c in common])
    both = [c for c in common if la[c]["qualificacao"] and lb[c]["qualificacao"]]
    res["qualificacao"] = _kappa([la[c]["qualificacao"] for c in both], [lb[c]["qualificacao"] for c in both]) if both else None
    both_f = [c for c in common if la[c]["funil"] and lb[c]["funil"]]
    if both_f:   # escala ordinal: kappa ponderado (linear) penaliza menos o erro entre estágios vizinhos
        xa = [FUNIL.index(la[c]["funil"]) for c in both_f]; xb = [FUNIL.index(lb[c]["funil"]) for c in both_f]
        res["funil"] = {"kappa_ponderado": float(cohen_kappa_score(xa, xb, weights="linear"))
                        if len(set(xa) | set(xb)) > 1 else (1.0 if xa == xb else None),
                        "kappa_simples": _kappa(xa, xb)}
    else:
        res["funil"] = None
    for dim, allowed in (("objecao", OBJECOES), ("gargalo", GARGALOS)):
        per = {}
        for lab in allowed:
            per[lab] = _kappa([int(lab in la[c][dim]) for c in common], [int(lab in lb[c][dim]) for c in common])
        vals = [v for v in per.values() if v is not None]
        res[dim] = {"por_rotulo": per, "media": sum(vals) / len(vals) if vals else None}
    return res


def _kappa(x, y):
    if len(set(x) | set(y)) < 2:
        return 1.0 if x == y else None  # sem variação: kappa indefinido
    return float(cohen_kappa_score(x, y))


def disagreements(conn: sqlite3.Connection, a: str = "a1", b: str = "a2") -> list[str]:
    """IDs com qualquer divergência — lista para adjudicação."""
    la, lb = get_labels(conn, a), get_labels(conn, b)
    core = ("escopo", "funil", "qualificacao", "objecao", "gargalo")   # metadados não geram divergência
    return [c for c in sorted(set(la) & set(lb)) if any(la[c][k] != lb[c][k] for k in core)]


# ---- adjudicação e montagem do ouro -----------------------------------------------------------------------------
CORE = ("escopo", "funil", "qualificacao", "objecao", "gargalo")


def _as_cells(lab: dict) -> dict:
    """Rótulos de uma conversa (saída de get_labels) -> células do CSV."""
    return {"escopo": lab["escopo"] or "", "funil": lab["funil"] or "",
            "objecao": ";".join(sorted(lab["objecao"])) or NONE,
            "gargalo": ";".join(sorted(lab["gargalo"])) or NONE,
            "desfecho": lab["desfecho"] or "", "para_terceiro": lab["para_terceiro"] or ""}


def export_disagreements(conn: sqlite3.Connection, path: str, a: str = "a1", b: str = "a2") -> int:
    """CSV de adjudicação: uma linha por conversa divergente, com os rótulos de a e b lado a lado.

    As colunas de rótulo vêm EM BRANCO para a decisão de consenso; depois importe com
    `import-labels --annotator adjudicado`. As colunas `<a>_*`/`<b>_*` são só referência (ignoradas na importação).
    """
    la, lb = get_labels(conn, a), get_labels(conn, b)
    ids = disagreements(conn, a, b)
    ref = [f"{who}_{k}" for who in (a, b) for k in ("escopo", "funil", "objecao", "gargalo")]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + ["divergem_em"] + ref); w.writeheader()
        for cid in ids:
            c = conn.execute("SELECT seq_in_lead, first_response_sec, median_response_sec FROM conversations "
                             "WHERE conversation_id=?", (cid,)).fetchone()
            msgs = [dict(m) for m in conn.execute(
                "SELECT sender_role, text FROM messages WHERE conversation_id=? ORDER BY ts, rowid", (cid,))]
            row = {k: "" for k in FIELDS}
            row.update({"conversation_id": cid, "seq_in_lead": c["seq_in_lead"],
                        "primeira_resposta_min": "" if c[1] is None else round(c[1] / 60, 1),
                        "mediana_resposta_min": "" if c[2] is None else round(c[2] / 60, 1),
                        "texto": conversation_text(msgs),
                        "divergem_em": ";".join(k for k in CORE if la[cid][k] != lb[cid][k])})
            for who, lab in ((a, la[cid]), (b, lb[cid])):
                cells = _as_cells(lab)
                for k in ("escopo", "funil", "objecao", "gargalo"):
                    row[f"{who}_{k}"] = cells[k]
            w.writerow(row)
    return len(ids)


def build_gold(conn: sqlite3.Connection, a: str = "a1", b: str = "a2", gold: str = "adjudicado") -> dict:
    """Monta a camada-ouro (`adjudicado`) que os experimentos leem.

    - conversa já adjudicada (importada como `gold`): mantida como está;
    - rotulada por a e b e CONCORDANTE nas dimensões centrais: copia os rótulos de a;
    - rotulada por a e b e divergente, sem adjudicação: fica PENDENTE (não entra no ouro);
    - rotulada só por a (anotação única, fora do subconjunto cego): copia os rótulos de a.
    """
    la, lb, lg = get_labels(conn, a), get_labels(conn, b), get_labels(conn, gold)
    stats = {"ja_adjudicadas": 0, "concordantes": 0, "anotacao_unica": 0, "pendentes": []}
    to_copy = []
    for cid in sorted(la):
        if cid in lg:
            stats["ja_adjudicadas"] += 1
        elif cid in lb:
            if all(la[cid][k] == lb[cid][k] for k in CORE):
                stats["concordantes"] += 1; to_copy.append(cid)
            else:
                stats["pendentes"].append(cid)
        else:
            stats["anotacao_unica"] += 1; to_copy.append(cid)
    with conn:
        for cid in to_copy:
            conn.execute("""INSERT OR IGNORE INTO labels(conversation_id, annotator, dimension, value)
                            SELECT conversation_id, ?, dimension, value FROM labels
                            WHERE conversation_id=? AND annotator=?""", (gold, cid, a))
    stats["so_no_segundo_anotador"] = len(set(lb) - set(la))
    stats["total_no_ouro"] = len(get_labels(conn, gold))
    return stats
