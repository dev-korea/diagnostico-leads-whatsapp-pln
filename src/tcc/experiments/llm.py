"""Classificação por LLM em regime few-shot.

Boas práticas implementadas (reprodutibilidade — Metodologia):
- temperatura 0, modelo com VERSÃO FIXA registrada nas previsões;
- exemplos few-shot sorteados SOMENTE do conjunto de treino (nunca val/teste), com semente;
- saída em JSON validada; respostas inválidas são re-tentadas e contabilizadas;
- cache em disco por hash (provider|modelo|prompt): reexecutar não gasta API nem muda resultado;
- apenas texto PSEUDONIMIZADO é enviado ao provedor (LGPD). Use API sem retenção/treino.
Provedores: anthropic, openai, ollama (modelo aberto LOCAL: o texto não sai da máquina), mock (offline, para testes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd

from ..dataset import TASKS, load_task_frame
from ..ingest import connect
from ..taxonomy import DEFINICOES
from .common import NONE, save_predictions

TASK_QUESTION = {
    "escopo": "A conversa é de um cliente potencial da clínica (em_escopo) ou é um contato que não é cliente "
              "potencial, como fornecedor, spam, engano, vaga de emprego ou cobrança (fora_de_escopo)?",
    "qualificacao": "O cliente está qualificado (deu sinal explícito de intenção de agendar nesta conversa, tenha ou "
                    "não fechado o agendamento) ou não qualificado? Classifique pelo ponto mais avançado da conversa.",
    "objecao": "Quais objeções de compra o cliente levantou? Pode haver várias, uma ou nenhuma.",
}


# ----------------------------------------------------------------------------- prompt ----
def select_examples(train: pd.DataFrame, task: str, k: int, seed: int, max_chars: int = 900):
    """k exemplos por classe (single-label) ou por rótulo + k sem objeção (multilabel)."""
    spec = TASKS[task]; rng = random.Random(seed); chosen = []
    pools = {}
    if spec["multilabel"]:
        for l in spec["labels"]:
            pools[l] = [r for r in train.itertuples() if l in r.y]
        pools[NONE] = [r for r in train.itertuples() if not r.y]
    else:
        for l in spec["labels"]:
            pools[l] = [r for r in train.itertuples() if r.y == l]
    seen = set()
    for pool in pools.values():
        pool = sorted(pool, key=lambda r: r.conversation_id); rng.shuffle(pool)
        taken = 0
        for r in pool:
            if r.conversation_id in seen:
                continue
            chosen.append(r); seen.add(r.conversation_id); taken += 1
            if taken >= k:
                break
    rng.shuffle(chosen)
    return [(r.text[:max_chars], r.y) for r in chosen]


def _fmt_answer(task: str, y) -> str:
    if TASKS[task]["multilabel"]:
        return json.dumps({"labels": sorted(y) if y else []}, ensure_ascii=False)
    return json.dumps({"label": y}, ensure_ascii=False)


def build_prompt(task: str, text: str, examples) -> tuple[str, str]:
    spec = TASKS[task]; labels = spec["labels"]
    defs = "\n".join(f"- {l}: {DEFINICOES[l]}" for l in labels)
    fmt = ('{"labels": ["rotulo1", "rotulo2"]}  (lista vazia [] se nenhuma objeção)'
           if spec["multilabel"] else '{"label": "<um dos rótulos>"}')
    system = (
        "Você é um analista de atendimento comercial de uma clínica de quiropraxia no Brasil. "
        "Você classifica conversas de WhatsApp (já anonimizadas; marcadores como [NOME] substituem dados pessoais). "
        f"{TASK_QUESTION[task]}\n\nRótulos permitidos e definições:\n{defs}\n\n"
        f"Responda SOMENTE com JSON válido, sem texto extra, neste formato: {fmt}")
    shots = "\n\n".join(f"### Conversa\n{t}\n### Resposta\n{_fmt_answer(task, y)}" for t, y in examples)
    user = (f"{shots}\n\n" if shots else "") + f"### Conversa\n{text}\n### Resposta\n"
    return system, user


def parse_response(task: str, raw: str):
    """Retorna rótulo/conjunto válido ou None."""
    spec = TASKS[task]; allowed = set(spec["labels"])
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if spec["multilabel"]:
        labs = obj.get("labels")
        if not isinstance(labs, list) or any(not isinstance(x, str) for x in labs):
            return None
        labs = {x.strip() for x in labs if x.strip() and x.strip() != NONE}
        return labs if labs <= allowed else None
    lab = obj.get("label")
    return lab if isinstance(lab, str) and lab in allowed else None


# -------------------------------------------------------------------------- provedores ----
class Provider:
    name = "base"; model = ""
    def complete(self, system: str, user: str) -> str:  # pragma: no cover
        raise NotImplementedError


class AnthropicProvider(Provider):
    name = "anthropic"
    def __init__(self, model: str):
        import anthropic  # import tardio: dependência opcional
        self.model = model; self.client = anthropic.Anthropic()  # lê ANTHROPIC_API_KEY
    def complete(self, system, user):
        r = self.client.messages.create(model=self.model, max_tokens=200, temperature=0, system=system,
                                        messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class OpenAIProvider(Provider):
    name = "openai"
    def __init__(self, model: str):
        from openai import OpenAI
        self.model = model; self.client = OpenAI()  # lê OPENAI_API_KEY
    def complete(self, system, user):
        r = self.client.chat.completions.create(model=self.model, temperature=0, seed=0, max_tokens=200,
                                                messages=[{"role": "system", "content": system},
                                                          {"role": "user", "content": user}])
        return r.choices[0].message.content or ""


class OllamaProvider(Provider):
    """Modelo aberto servido localmente pelo Ollama (http://localhost:11434). Nada sai da máquina.

    A versão registrada inclui o digest do modelo (ex.: qwen2.5:1.5b@65ec06548149), que identifica os pesos exatos.
    """
    name = "ollama"
    def __init__(self, model: str, host: Optional[str] = None):
        import urllib.request
        self._rq = urllib.request
        self.host = (host or os.environ.get("OLLAMA_HOST_URL") or "http://localhost:11434").rstrip("/")
        self.tag = model
        tags = json.loads(self._rq.urlopen(self.host + "/api/tags", timeout=30).read())
        dig = next((m.get("digest", "") for m in tags.get("models", []) if m.get("name") == model), "")
        if not dig:
            raise ValueError(f"Modelo {model!r} não está no Ollama local (rode `ollama pull {model}`).")
        self.model = f"{model}@{dig[:12]}"
    def complete(self, system, user):
        body = {"model": self.tag, "stream": False, "format": "json",
                "options": {"temperature": 0, "seed": 0, "num_predict": 200, "num_ctx": 8192},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        req = self._rq.Request(self.host + "/api/chat", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
        return json.loads(self._rq.urlopen(req, timeout=1800).read())["message"]["content"]


class MockProvider(Provider):
    """Provedor offline determinístico (palavras-chave). Só para testar o encanamento."""
    name = "mock"; model = "mock-keywords-v1"
    KW = {
        "preco": r"caro|desconto|parcel|valor est|puxado|acima do que",
        "horario_disponibilidade": r"à noite|a noite|só consigo|so consigo|horário é|horario e|disponibilidade|saio do serviço",
        "localizacao": r"bairro|estacionamento|distante|longe|endereço|endereco",
        "confianca_resultado": r"funciona|medo de piorar|garantia|formado|receio|dá resultado|da resultado",
        "convenio_reembolso": r"plano de saúde|plano de saude|convênio|convenio|reembolso|unimed",
        "adiamento": r"vou pensar|te aviso|retorno depois|mais pra frente|depois te falo",
        "concorrencia": r"outra clínica|outra clinica|fisioterapia em outro|outro lugar|comparar",
    }
    def complete(self, system, user):
        conv = user.split("### Conversa")[-1].split("### Resposta")[0]
        lead_lines = "\n".join(l for l in conv.splitlines() if l.lower().startswith("cliente:")).lower()
        if "TASK:objecao" in system or "objeções de compra" in system:
            labs = [k for k, p in self.KW.items() if re.search(p, lead_lines)]
            return json.dumps({"labels": labs})
        if "fora_de_escopo)?" in system:
            fora = re.search(r"represento|currículo|curriculo|vaga|errei o número|errei o numero|mensagem errada|"
                             r"seguidores|boleto|sistema de agendamento", lead_lines)
            return json.dumps({"label": "fora_de_escopo" if fora else "em_escopo"})
        qual = re.search(r"quero marcar|pode reservar|fechado|confirmo|quinta|amanhã|amanha|horário amanhã|tem horário", lead_lines)
        return json.dumps({"label": "qualificado" if qual else "nao_qualificado"})


def make_provider(name: str, model: Optional[str]) -> Provider:
    if name == "mock":
        return MockProvider()
    if not model:
        raise ValueError("--model é obrigatório (fixe a versão exata do modelo).")
    return {"anthropic": AnthropicProvider, "openai": OpenAIProvider, "ollama": OllamaProvider}[name](model)


# -------------------------------------------------------------------------------- cache ----
class Cache:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()
    @staticmethod
    def key(provider: str, model: str, system: str, user: str) -> str:
        return hashlib.sha256("\x1f".join([provider, model, "t=0", system, user]).encode()).hexdigest()
    def get(self, k):
        with self.lock:
            r = self.conn.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return r[0] if r else None
    def put(self, k, v):
        with self.lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, v))


# --------------------------------------------------------------------------------- run ----
def classify_one(provider: Provider, cache: Cache, task: str, text: str, examples, retries: int = 2,
                 backoff: float = 2.0):
    system, user = build_prompt(task, text, examples)
    k = Cache.key(provider.name, provider.model, system, user)
    cached = cache.get(k)
    attempts = 0
    while True:
        raw = cached
        if raw is None:
            try:
                raw = provider.complete(system, user)
            except Exception:  # rate limit / rede: espera e tenta de novo
                attempts += 1
                if attempts > retries + 3:
                    raise
                time.sleep(backoff ** attempts); continue
        parsed = parse_response(task, raw)
        if parsed is not None:
            cache.put(k, raw)
            return parsed, True
        cached = None  # resposta inválida não é aceita do cache; tenta de novo
        attempts += 1
        if attempts > retries:
            return None, False


def run(df: pd.DataFrame, task: str, provider: Provider, cache: Cache, k: int = 3, seed: int = 0,
        workers: int = 4, limit: Optional[int] = None):
    spec = TASKS[task]
    train, test = df[df.part == "train"], df[df.part == "test"]
    if limit:
        test = test.head(limit)
    examples = select_examples(train, task, k, seed) if k > 0 else []
    fallback = set() if spec["multilabel"] else train.y.mode().iloc[0]
    results = {}; invalid = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(classify_one, provider, cache, task, r.text, examples): r.conversation_id
                for r in test.itertuples()}
        for f in as_completed(futs):
            pred, ok = f.result()
            results[futs[f]] = pred if ok else fallback
            invalid += (not ok)
    ids = [i for i in test.conversation_id]
    return ids, [results[i] for i in ids], {"invalid": invalid, "n": len(ids), "n_examples": len(examples)}


def _slug(prov: Provider) -> str:
    """Sufixo do modelo no nome do método/arquivo, para comparar dois modelos do mesmo provedor (só no ollama)."""
    if prov.name != "ollama":
        return ""
    return "-" + re.sub(r"[^a-z0-9.]+", "-", prov.tag.lower()).strip("-")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True); ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--provider", required=True, choices=["anthropic", "openai", "ollama", "mock"])
    ap.add_argument("--model", default=None, help="versão exata, ex.: claude-sonnet-5-5 ou qwen2.5:1.5b (ollama)")
    ap.add_argument("--k", type=int, default=3, help="exemplos por classe (0 = zero-shot)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="v1"); ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=None, help="só as N primeiras do teste (teste de custo)")
    ap.add_argument("--cache", default="results/llm_cache.sqlite")
    a = ap.parse_args(argv)
    df = load_task_frame(connect(a.db), a.task, a.split)
    prov = make_provider(a.provider, a.model)
    ids, preds, info = run(df, a.task, prov, Cache(a.cache), a.k, a.seed, a.workers, a.limit)
    method = f"llm_fewshot_k{a.k}" if a.k else "llm_zeroshot"
    n = save_predictions(Path(a.out) / f"preds_{method}_{prov.name}{_slug(prov)}_{a.task}.csv", (
        {"conversation_id": i, "task": a.task, "method": f"{method}:{prov.name}{_slug(prov)}", "model_version": prov.model,
         "seed": a.seed, "pred": p} for i, p in zip(ids, preds)))
    print(f"{method}/{prov.name}/{a.task}: {n} previsões; inválidas={info['invalid']}/{info['n']}")


if __name__ == "__main__":
    main()
