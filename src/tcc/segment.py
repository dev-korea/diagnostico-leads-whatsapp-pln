"""Segmentação em conversas (janela de inatividade) e métricas de atendimento.

Regra: uma nova conversa começa quando o intervalo entre duas mensagens consecutivas
(não-sistema) excede `gap_hours` (padrão 24 h). É um parâmetro do estudo: registre-o na
Metodologia e, se possível, faça análise de sensibilidade (12 h / 24 h / 48 h).
`seq_in_lead` > 1 indica recorrência (o lead voltou a falar após uma pausa).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from statistics import median
from typing import Optional


@dataclass
class Conversation:
    seq: int
    messages: list[dict]

    @property
    def started_at(self): return self.messages[0]["ts"]
    @property
    def ended_at(self): return self.messages[-1]["ts"]


def segment(messages: list[dict], gap_hours: float = 24.0) -> list[Conversation]:
    real = sorted((m for m in messages if m["sender_role"] != "sistema"),
                  key=lambda m: (m["ts"], m.get("order", 0)))
    convs: list[Conversation] = []
    gap = timedelta(hours=gap_hours)
    for m in real:
        if not convs or (m["ts"] - convs[-1].messages[-1]["ts"]) > gap:
            convs.append(Conversation(seq=len(convs) + 1, messages=[m]))
        else:
            convs[-1].messages.append(m)
    return convs


def response_times(msgs: list[dict]) -> list[float]:
    """Segundos entre a 1ª mensagem do lead sem resposta e a resposta do atendente."""
    times, pending = [], None
    for m in msgs:
        if m["sender_role"] == "lead":
            pending = pending or m["ts"]
        elif m["sender_role"] == "atendente" and pending is not None:
            times.append((m["ts"] - pending).total_seconds())
            pending = None
    return times


def conversation_stats(conv: Conversation) -> dict:
    rt = response_times(conv.messages)
    n_lead = sum(m["sender_role"] == "lead" for m in conv.messages)
    n_att = sum(m["sender_role"] == "atendente" for m in conv.messages)
    return {
        "n_messages": len(conv.messages), "n_lead_messages": n_lead, "n_agent_messages": n_att,
        "first_response_sec": rt[0] if rt else None,
        "median_response_sec": float(median(rt)) if rt else None,
    }


def conversation_text(msgs: list[dict], include_roles: bool = True) -> str:
    """Texto que alimenta os classificadores: 'Cliente: ...' / 'Atendente: ...'."""
    label = {"lead": "Cliente", "atendente": "Atendente"}
    lines = []
    for m in msgs:
        if m["sender_role"] == "sistema" or m["text"] in ("[MIDIA]", "[APAGADA]"):
            continue
        lines.append(f"{label[m['sender_role']]}: {m['text']}" if include_roles else m["text"])
    return "\n".join(lines)
