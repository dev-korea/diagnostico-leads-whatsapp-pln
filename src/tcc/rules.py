"""Indicadores de gargalo calculáveis por regra (sem PLN): complementam o diagnóstico.

demora_resposta : 1ª resposta do atendente > limiar (padrão 30 min) ou mediana > limiar.
sem_followup    : heurística — a conversa termina com fala do ATENDENTE sem resposta do lead e sem
                  palavras de fechamento (agendado, confirmado, à disposição...). Tem precisão
                  limitada: use como pré-triagem e confirme na rotulagem humana.
`info_incompleta` exige leitura do conteúdo -> tarefa para LLM/humano (ver guia de rotulagem).
"""
from __future__ import annotations

CLOSING_HINTS = ("agendado", "confirmad", "até lá", "ate la", "combinado", "por nada",
                 "à disposição", "a disposição", "sem problemas", "aguardo seu retorno", "obrigado")


def rule_gargalos(conv: dict, slow_threshold_sec: float = 1800.0) -> set[str]:
    out: set[str] = set()
    frs, med = conv.get("first_response_sec"), conv.get("median_response_sec")
    if (frs is not None and frs > slow_threshold_sec) or (med is not None and med > slow_threshold_sec):
        out.add("demora_resposta")
    msgs = [m for m in conv["messages"] if m["sender_role"] in ("lead", "atendente")]
    if msgs and msgs[-1]["sender_role"] == "atendente":
        last = msgs[-1]["text"].lower()
        if not any(h in last for h in CLOSING_HINTS):
            out.add("sem_followup")
    return out
