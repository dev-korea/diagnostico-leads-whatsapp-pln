"""Prepara a rotulagem assistida: separa o subconjunto CEGO, mascara nomes que escaparam e gera lotes de leitura.

Saídas (tudo em data/processed/, fora do versionamento):
  rotular_a1.csv        todas as conversas, em branco (pre_rotulo = 'cega' no subconjunto cego)
  rotular_a2_cego.csv   só o subconjunto cego, em branco (o a2 nunca vê sugestões)
  lotes/lote_NN.txt     conversas NÃO cegas, em texto legível, para a pré-rotulagem

Uso:  python scripts/preparar_lotes.py [--lote 20] [--blind-frac 0.25] [--seed 13]
"""
import argparse
import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from tcc.ingest import connect, init_db          # noqa: E402
from tcc.prelabel import MODEL_FIELDS, PRE_COL, pick_blind_leads   # noqa: E402
from tcc.segment import conversation_text        # noqa: E402

# Nomes de pessoas que escaparam do mascaramento (revisão de candidatos). Endereços e operadoras NÃO entram.
NOMES = ["Paula", "Paulo", "Fernando", "Miguel", "Gabi", "Aparecida", "Gonçalves", "Marques", "Cardoso",
         "Benette", "Bordini", "Guimarães", "Moraes", "Rossitti", "Camurça", "Pereira", "Silva", "Santos"]
_RE_NOMES = re.compile(r"(?<!São )(?<!Sao )\b(" + "|".join(map(re.escape, NOMES)) + r")\b")


_RE_MEU_NOME = re.compile(r"(?i)\b((?:meu nome|me chamo|aqui é|aqui e)[ \t]+(?:é[ \t]+|e[ \t]+)?)(?!\[NOME\])(\w+(?:[ \t]+(?:d[aeo]s?[ \t]+)?[A-ZÀ-Ú]\w+)*)")
# fala do cliente que é só um nome próprio (2 a 6 palavras capitalizadas), exceto saudações/frases comuns
_PALAVRA = r"[A-ZÀ-Ú][A-Za-zÀ-ú'’]+"
_RE_SO_NOME = re.compile(rf"^Cliente: ({_PALAVRA}(?: (?:d[aeo]s?|e) {_PALAVRA}| {_PALAVRA}){{1,5}})\s*$")
_NAO_NOME = re.compile(r"(?i)\b(bom|boa|dia|tarde|noite|tudo|bem|obrigad[oa]|primeira|vez|vou|sim|não|nao|ok|certo|valor|"
                       r"consulta|sessão|sessao|quero|queria|gostaria|oi|olá|ola|por favor|obrigado|segunda|terça|quarta|"
                       r"quinta|sexta|sábado|manhã|manha)\b")


def mascarar(texto: str, cont: Counter) -> str:
    def sub(m):
        cont[m.group(1)] += 1
        return "[NOME]"
    # 1) nome dado quando o atendente pede "nome completo": a 1ª fala seguinte do cliente é mascarada
    linhas, pedir = [], False
    for ln in texto.split("\n"):
        m_so = _RE_SO_NOME.match(ln)
        if m_so and not _NAO_NOME.search(m_so.group(1)):
            ln = "Cliente: [NOME]"; cont["(só nome)"] += 1
        elif pedir and ln.startswith("Cliente:"):
            ln = "Cliente: [NOME]"; cont["(nome completo)"] += 1; pedir = False
        elif ln.startswith("Atendente:"):
            pedir = bool(re.search(r"nome completo|qual (?:o )?seu nome", ln, re.I))
        linhas.append(ln)
    texto = "\n".join(linhas)
    # 2) "meu nome é X", "me chamo X"
    def meu(m):
        cont["(meu nome)"] += 1
        return m.group(1) + "[NOME]"
    texto = _RE_MEU_NOME.sub(meu, texto)
    # 3) lista de sobrenomes/nomes isolados que escaparam
    return _RE_NOMES.sub(sub, texto)


def comprimir_para_leitura(texto: str, limite: int = 350) -> str:
    """Só nos lotes de leitura: mensagem longa do atendente (texto padrão) vira resumo. O CSV mantém tudo."""
    out = []
    for ln in texto.split("\n"):
        if ln.startswith("Atendente:") and len(ln) > limite:
            ln = ln[:140].rstrip() + f" […texto longo do atendente, {len(ln)} caracteres…]"
        out.append(ln)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(RAIZ / "data/processed/tcc.db"))
    ap.add_argument("--lote", type=int, default=20)
    ap.add_argument("--blind-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    conn = connect(a.db); init_db(conn)
    blind = pick_blind_leads(conn, a.blind_frac, a.seed)
    convs = [dict(c) for c in conn.execute(
        "SELECT conversation_id, lead_id, seq_in_lead, first_response_sec, median_response_sec FROM conversations")]
    random.Random(a.seed).shuffle(convs)           # mesma ordem do tcc.prelabel

    cont: Counter = Counter()
    rows, rows_cegas, abertas = [], [], []
    for c in convs:
        msgs = [dict(m) for m in conn.execute(
            "SELECT sender_role, text FROM messages WHERE conversation_id=? ORDER BY ts, rowid",
            (c["conversation_id"],))]
        texto = mascarar(conversation_text(msgs), cont)
        first = "" if c["first_response_sec"] is None else round(c["first_response_sec"] / 60, 1)
        med = "" if c["median_response_sec"] is None else round(c["median_response_sec"] / 60, 1)
        row = {k: "" for k in MODEL_FIELDS}
        row.update({"conversation_id": c["conversation_id"], "seq_in_lead": c["seq_in_lead"],
                    "primeira_resposta_min": first, "mediana_resposta_min": med, "texto": texto})
        if c["lead_id"] in blind:
            row[PRE_COL] = "cega"; rows_cegas.append(row)
        else:
            abertas.append((row, texto))
        rows.append(row)

    saida = RAIZ / "data/processed"; (saida / "lotes").mkdir(parents=True, exist_ok=True)
    for nome, dados in (("rotular_a1.csv", rows), ("rotular_a2_cego.csv", rows_cegas)):
        with open(saida / nome, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=MODEL_FIELDS); w.writeheader(); w.writerows(dados)
    for f in (saida / "lotes").glob("lote_*.txt"):
        f.unlink()
    for i in range(0, len(abertas), a.lote):
        partes = []
        for row, texto in abertas[i:i + a.lote]:
            partes.append(f"=== ID: {row['conversation_id']} | resp1_min={row['primeira_resposta_min']} "
                          f"med_min={row['mediana_resposta_min']}\n{comprimir_para_leitura(texto)}\n")
        (saida / "lotes" / f"lote_{i // a.lote + 1:02d}.txt").write_text("\n".join(partes), encoding="utf-8")
    print(f"total={len(rows)} cegas={len(rows_cegas)} abertas={len(abertas)} lotes={-(-len(abertas)//a.lote)}")
    print("nomes mascarados:", dict(cont))


if __name__ == "__main__":
    main()
