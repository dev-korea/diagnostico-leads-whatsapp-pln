"""Gerador de exportações SINTÉTICAS de WhatsApp (clínica de quiropraxia) com rótulos-ouro.

ATENÇÃO: serve para TESTAR o pipeline (parser, banco, scripts de experimento). A distribuição é
construída por regras; métricas obtidas aqui NÃO podem ser reportadas como resultado do TCC.
"""
from __future__ import annotations

import csv
import random
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

from .pseudonymize import hash_identifier

NOMES = ["Carla", "Joao", "Mariana", "Pedro", "Luciana", "Rafael", "Fernanda", "Gustavo", "Aline",
         "Bruno", "Patricia", "Diego", "Camila", "Rodrigo", "Juliana", "Marcelo", "Tatiane", "Felipe",
         "Vanessa", "Thiago", "Larissa", "Leandro", "Priscila", "Eduardo", "Renata", "Henrique", "Bianca", "Caio"]
SOBRENOMES = ["Silva", "Souza", "Oliveira", "Santos", "Pereira", "Costa", "Rodrigues", "Almeida",
              "Nascimento", "Lima", "Araujo", "Ribeiro", "Martins", "Carvalho"]
ATENDENTES = ["Clinica Vida Plena", "Helena Recepcao"]
QUEIXAS = ["dor lombar há meses", "dor na cervical que irradia pro braço", "uma hérnia de disco",
           "dor no ombro direito", "ciático que não passa", "dor de cabeça tensional constante",
           "dor nas costas depois de ficar sentado o dia todo", "formigamento na perna"]
ABERTURAS = [
    "Oi, boa tarde! Vi o anúncio de vocês no Instagram. Como funciona a consulta de quiropraxia?",
    "Olá, gostaria de saber valores e horários para uma avaliação.",
    "Bom dia! Vocês atendem {queixa}? Queria agendar.",
    "oi! tô com {queixa} e uma amiga indicou vcs",
    "Boa noite, vi no Google que vocês fazem quiropraxia. Como faço pra marcar?",
    "Oii, queria informações sobre o tratamento, estou com {queixa}.",
]
ATEND_SAUDACAO = [
    "Olá! Tudo bem? Aqui é a equipe da Clínica Vida Plena. Me conta, qual é a sua queixa principal?",
    "Oi, boa tarde! Que bom ter você por aqui. O que está te incomodando?",
    "Olá! Seja bem-vindo(a). Pode me contar um pouco do que sente?",
]
ATEND_EXPLICA = [
    "Entendi. A primeira consulta é uma avaliação completa, dura em torno de 50 minutos e já inclui o primeiro ajuste, se indicado.",
    "Certo! Fazemos uma avaliação postural e neurológica e já definimos o plano. A consulta inicial custa R$ {valor}.",
    "Compreendo. Na primeira sessão avaliamos tudo e explicamos o tratamento. O valor da avaliação é R$ {valor}.",
]
OBJ = {
    "preco": (["Nossa, R$ {valor} está um pouco caro pra mim", "Tem algum desconto? Ficou puxado esse valor",
               "achei caro, tem como parcelar?", "O valor está acima do que eu esperava"],
              ["Entendo! Parcelamos em até 3x sem juros e temos pacote com valor reduzido.",
               "Compreendo. Podemos parcelar, e o pacote de sessões sai mais em conta."]),
    "horario_disponibilidade": (["Trabalho o dia todo, vocês atendem à noite?", "Só consigo sábado, tem horário?",
                                 "meu horário é complicado, só saio do serviço às 19h",
                                 "Não tenho disponibilidade durante a semana de manhã"],
                                ["Temos horários até as 20h e aos sábados pela manhã.",
                                 "Atendemos de segunda a sexta até as 20h. Posso verificar um encaixe."]),
    "localizacao": (["Vocês ficam em que bairro? Moro bem longe", "Tem estacionamento aí perto?",
                     "Fica muito distante, moro em outra cidade", "onde é o endereço? é longe do centro?"],
                    ["Ficamos na região central, com estacionamento conveniado ao lado.",
                     "Estamos no centro, próximo à praça principal, com fácil acesso."]),
    "confianca_resultado": (["Isso realmente funciona? Tenho medo de piorar", "Já fui em outros lugares e não resolveu, "
                             "qual a garantia?", "O profissional é formado? Tenho receio de estalar a coluna",
                             "Quantas sessões até melhorar? Dá resultado mesmo?"],
                            ["O quiropraxista é graduado e especializado; a técnica é segura e personalizada.",
                             "Cada caso é único, mas a maioria sente melhora já nas primeiras sessões."]),
    "convenio_reembolso": (["Vocês atendem pelo plano de saúde?", "Aceita convênio? Se não, dá nota pra reembolso?",
                            "meu plano cobre? vcs trabalham com unimed?"],
                           ["Não atendemos por convênio, mas emitimos recibo para reembolso.",
                            "Trabalhamos de forma particular e fornecemos nota fiscal para reembolso."]),
    "adiamento": (["Vou pensar e depois te falo", "Deixa eu ver aqui e te aviso", "Preciso ver com calma, retorno depois",
                   "ainda não sei, mais pra frente eu vejo"],
                  ["Claro, fico à disposição! Qualquer dúvida é só chamar.", "Sem problemas, aguardo seu retorno."]),
    "concorrencia": (["Estou vendo outra clínica que cobra menos", "Já faço fisioterapia em outro lugar, qual a diferença?",
                      "Pedi orçamento em outra clínica também, vou comparar"],
                     ["Nosso diferencial é a avaliação completa e o acompanhamento individual.",
                      "Entendo, compare à vontade. Aqui o foco é tratar a causa."]),
}
FORA = [
    (["Bom dia, represento uma empresa de software de gestão e gostaria de apresentar nossa solução.",
      "Vocês têm interesse em um sistema de agendamento? Posso mandar uma proposta."],
     ["Obrigado, no momento não temos interesse."]),
    (["Oi, vocês estão contratando? Tenho experiência como recepcionista.", "Gostaria de enviar meu currículo, tem vaga?"],
     ["Por enquanto não temos vagas abertas, obrigado!"]),
    (["Desculpa, errei o número", "Oi? Quem é? Acho que mandei mensagem errada"], ["Sem problemas, tudo bem!"]),
    (["PROMOÇÃO imperdível! Seguidores e curtidas baratos, clique no link"], []),
    (["Boa tarde, sou do laboratório e estamos com o boleto em aberto da compra de materiais."],
     ["Vou repassar ao setor responsável, obrigado."]),
]
FECHA_QUAL = [
    (["Pode ser quinta às 18h?", "Tem horário amanhã de manhã?", "Quero marcar a avaliação, o que tem disponível?"],
     ["Tenho quinta às 18h ou sexta às 9h. Qual prefere?", "Tenho amanhã às 10h. Posso reservar?"],
     ["Perfeito, pode reservar!", "Fechado, quinta às 18h então. Meu CPF é {cpf}", "Confirmo! Me passa o endereço?"],
     ["Agendado! Enviamos a confirmação por aqui. Até lá!"]),
]
FECHA_NAO = [
    (["Ok, obrigado pelas informações", "Entendi, era só pra saber mesmo", "Beleza, valeu"],
     ["Por nada! Qualquer coisa estamos à disposição."]),
]
SILENCIO_FOLLOW = ["Olá! Tudo bem? Conseguiu pensar sobre a avaliação? Temos horários esta semana.",
                   "Oi! Passando para saber se ainda tem interesse em agendar."]
RETORNO = ["Oi, voltei! Ainda tem horário pra avaliação?", "Boa tarde, lembra de mim? Quero marcar aquela consulta.",
           "oi, aquele tratamento ainda tá disponível? quero agendar"]
EMOJI = ["😊", "🙏", "👍", "😅", "😉"]

# ---- Modo difícil (hard=True): falas implícitas, ruído e casos limítrofes para NÃO inflar as métricas ----
OBJ_HARD = {
    "preco": ["Hmm, dependendo do valor total eu vejo", "e quanto fica o pacote completo? to apertada esse mês",
              "poxa, sem condição de pagar tudo de uma vez", "minha filha me falou que em outro lugar era bem mais barato"],
    "horario_disponibilidade": ["acho q só consigo depois das 19h, mas nem sempre", "meu turno muda toda semana, complica",
                                "fim de semana seria o ideal pra mim"],
    "localizacao": ["vim de carro no centro uma vez e foi um sufoco pra estacionar", "é perto de qual avenida?",
                    "moro do outro lado da cidade, uns 40 min"],
    "confianca_resultado": ["meu médico disse pra evitar manipular a coluna, e agora?", "já tentei tanta coisa sem resultado...",
                            "vi uns vídeos falando mal de quiropraxia, queria entender"],
    "convenio_reembolso": ["mando a nota pro meu plano depois, vcs emitem?", "só faço se o convênio cobrir"],
    "adiamento": ["semana que vem eu te chamo", "agora não é um bom momento, mas guarda meu contato",
                  "preciso conversar em casa antes"],
    "concorrencia": ["uma colega indicou outro profissional, vou ver os dois", "tem uma clínica nova aqui perto também"],
}
FORA_HARD = [
    (["Boa tarde, vocês vendem colchão ou travesseiro ortopédico? Tenho dor nas costas.",
      "Vocês revendem palmilha ou cinta postural pra coluna?"],
     ["Não trabalhamos com venda de produtos, somente atendimento clínico."]),
    (["Oi, sou estudante de fisioterapia, posso acompanhar uma consulta de vocês pra um trabalho?",
      "Olá, sou estudante e queria fazer estágio observacional na clínica, é possível?"],
     ["Obrigado pelo contato, no momento não temos esse tipo de parceria."]),
    (["Oi, quero divulgar uma palestra sobre saúde da coluna, vocês toparam apoiar?",
      "Bom dia, represento uma academia e queria propor parceria de divulgação."],
     ["Vou repassar para a direção, obrigado."]),
]
ABERTURAS_HARD = ["Oi", "boa tarde, tudo bem?", "Olá! Queria informações", "oii", "Bom dia! Preciso de ajuda com uma coisa"]
ESCOPO_AMBIGUO_FOLLOW = ["é pra tratar dor, sabe? faz tempo que to assim", "Queria marcar uma consulta, vcs atendem coluna?"]
TYPOS = {"consulta": "consluta", "avaliação": "avalaicao", "horário": "horairo", "tratamento": "tratameto",
         "agendar": "agedar", "valor": "valro", "quiropraxia": "quiropaxia", "sessão": "sessao", "clínica": "clinica"}


def typo(text: str, rng: random.Random) -> str:
    """Erros de digitação plausíveis (troca/omissão de letras) em ~metade das mensagens."""
    for k, v in TYPOS.items():
        if k in text.lower() and rng.random() < 0.5:
            text = re.sub(k, v, text, flags=re.I)
    words = text.split(" ")
    if len(words) > 3 and rng.random() < 0.25:
        i = rng.randrange(len(words))
        w = words[i]
        if len(w) > 4 and w.isalpha():
            j = rng.randrange(1, len(w) - 2)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def _strip(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def noisify(text: str, rng: random.Random) -> str:
    """Ruído típico de chat informal: abreviações, sem acento, minúsculas, risadas, emojis."""
    t = text
    for a, b in [(r"\bvocê\b", "vc"), (r"\bvocês\b", "vcs"), (r"\bque\b", "q"), (r"\bporque\b", "pq"),
                 (r"\btambém\b", "tbm"), (r"\bnão\b", "n"), (r"\bpara\b", "pra"), (r"\bestá\b", "tá")]:
        if rng.random() < 0.6:
            t = re.sub(a, b, t, flags=re.I)
    if rng.random() < 0.5:
        t = _strip(t)
    if rng.random() < 0.5:
        t = t.lower().rstrip(".!?")
    if rng.random() < 0.15:
        t += " " + rng.choice(["kkk", "kkkk", "rs"])
    if rng.random() < 0.2:
        t += " " + rng.choice(EMOJI)
    return t


class _Chat:
    def __init__(self, rng, t0, hard=False):
        self.rng, self.t, self.rows, self.hard = rng, t0, [], hard

    def say(self, role, text, delay=None, noisy=False):
        r = self.rng
        if delay is None:
            delay = r.randint(20, 240) if role == "lead" else r.randint(60, 480)
        self.t += timedelta(seconds=delay)
        if self.hard and role == "lead" and not text.startswith("<"):
            if noisy and r.random() < 0.5:
                text = typo(text, r)
            parts = [text]
            if len(text.split()) > 6 and r.random() < 0.3:  # rajada: lead digita em duas mensagens
                w = text.split(" "); k = len(w) // 2
                parts = [" ".join(w[:k]), " ".join(w[k:])]
            for n, p in enumerate(parts):
                self.rows.append([self.t + timedelta(seconds=n * r.randint(3, 25)),
                                  role, noisify(p, r) if noisy else p])
            return
        self.rows.append([self.t, role, noisify(text, r) if noisy else text])


def _pin(rows, t0):
    for i, rw in enumerate(rows):
        rw[0] = t0 + timedelta(seconds=i * 4)


def _build_conversation(rng, t0, kind, objs, gargalos, noisy, lead_extra, hard=False):
    """kind: fora | qual | nao. Retorna linhas [(ts, role, text)]."""
    c = _Chat(rng, t0, hard)
    queixa = rng.choice(QUEIXAS)
    valor = rng.choice([180, 200, 250, 300])
    fill = lambda s: s.format(queixa=queixa, valor=valor, cpf=lead_extra["cpf"])
    slow = "demora_resposta" in gargalos
    unans = "info_incompleta" in gargalos

    if kind == "fora":
        ls, ats = rng.choice(FORA + FORA_HARD) if hard else rng.choice(FORA)
        c.say("lead", rng.choice(ls), noisy=noisy)
        _pin(c.rows, t0)
        if ats:
            c.say("atendente", ats[0], delay=rng.randint(3600, 30000) if slow else None)
        elif slow:
            gargalos.remove("demora_resposta")  # sem resposta do atendente não há tempo de resposta a medir
        return c.rows

    if hard and rng.random() < 0.35:  # abertura vaga: o escopo só se revela depois
        c.say("lead", rng.choice(ABERTURAS_HARD), noisy=noisy)
        _pin(c.rows, t0)
        c.say("atendente", rng.choice(ATEND_SAUDACAO), delay=rng.randint(7200, 50000) if slow else rng.randint(30, 400))
        c.say("lead", rng.choice(ESCOPO_AMBIGUO_FOLLOW), noisy=noisy)
        c.say("lead", f"Estou com {queixa}", noisy=noisy)
    else:
        c.say("lead", fill(rng.choice(ABERTURAS)), noisy=noisy)
        _pin(c.rows, t0)
        c.say("atendente", rng.choice(ATEND_SAUDACAO), delay=rng.randint(7200, 50000) if slow else rng.randint(30, 400))
        c.say("lead", f"Estou com {queixa}" if rng.random() < .7 else f"Sinto {queixa}", noisy=noisy)
    if rng.random() < .25:  # áudio do lead
        c.say("lead", "<Mídia oculta>")
    if lead_extra.get("intro") and rng.random() < .8:
        c.say("lead", lead_extra["intro"], noisy=False)
    c.say("atendente", fill(rng.choice(ATEND_EXPLICA)))
    for o in objs:
        lead_lines, att_lines = OBJ[o]
        pool = lead_lines + (OBJ_HARD[o] * 2 if hard else [])  # no modo difícil, metade das falas é implícita
        c.say("lead", fill(rng.choice(pool)), noisy=noisy)
        if unans and o == objs[-1]:
            continue  # atendente não responde esta pergunta -> info_incompleta
        c.say("atendente", rng.choice(att_lines))

    if kind == "qual":
        a, b, d, e = FECHA_QUAL[0]
        c.say("lead", fill(rng.choice(a)), noisy=noisy)
        c.say("atendente", rng.choice(b))
        c.say("lead", fill(rng.choice(d)), noisy=noisy)
        c.say("atendente", rng.choice(e))
    else:
        if "sem_followup" in gargalos:
            pass  # lead some após a última fala do atendente e ninguém retoma
        else:
            a, b = FECHA_NAO[0]
            c.say("lead", rng.choice(a), noisy=noisy)
            c.say("atendente", rng.choice(b))
    return c.rows


def _fmt_android(ts, sender, text):
    return f"{ts:%d/%m/%Y %H:%M} - {sender}: {text}"


def _fmt_iphone(ts, sender, text):
    return f"[{ts:%d/%m/%Y %H:%M:%S}] {sender}: {text}"


def generate(out_dir: str, n_leads: int = 60, seed: int = 7, secret: str = "segredo-de-teste-123456",
             hard: bool = False) -> dict:
    rng = random.Random(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    labels = []
    base = datetime(2025, 3, 3, 9, 0)
    used = set()
    for i in range(n_leads):
        while True:
            nome = f"{rng.choice(NOMES)} {rng.choice(SOBRENOMES)}"
            if len(used) >= len(NOMES) * len(SOBRENOMES) * 0.7:  # evita laço infinito com muitos leads
                nome += f" {rng.choice(SOBRENOMES)}"
            if nome not in used:
                used.add(nome); break
        fmt = rng.choice(["android", "iphone"])
        noisy = rng.random() < (0.75 if hard else 0.55)
        phone = f"(53) 9{rng.randint(1000,9999)}-{rng.randint(1000,9999)}"
        cpf = f"{rng.randint(100,999)}.{rng.randint(100,999)}.{rng.randint(100,999)}-{rng.randint(10,99)}"
        extra = {"cpf": cpf, "intro": rng.choice([f"Meu nome é {nome.split()[0]}", f"Me chamo {nome}",
                                                  f"Meu whatsapp é {phone}", None])}
        t0 = base + timedelta(days=rng.randint(0, 150), hours=rng.randint(0, 10), minutes=rng.randint(0, 59))
        scenario = rng.choices(["fora", "qual", "nao", "recorr"], weights=[.15, .30, .40, .15])[0]
        convs = []  # (kind, objs, garg, rows)

        def pick_objs(k):
            p = {"qual": .45, "nao": .6, "recorr": .6}[k]
            pool = list(OBJ)
            return [o for o in pool if rng.random() < p / 3.2]

        def pick_garg(k):
            g = []
            if rng.random() < .15: g.append("demora_resposta")
            if k != "fora" and rng.random() < .10: g.append("info_incompleta")
            return g

        if scenario == "fora":
            convs.append(("fora", [], pick_garg("fora")))
        elif scenario == "qual":
            convs.append(("qual", pick_objs("qual"), pick_garg("qual")))
        elif scenario == "nao":
            g = pick_garg("nao")
            if rng.random() < .35: g.append("sem_followup")
            convs.append(("nao", pick_objs("nao"), g))
        else:  # recorrência: 1ª sem decisão (adiamento), depois retorna e agenda
            o1 = list({"adiamento", *pick_objs("recorr")})
            convs.append(("nao", o1, ["sem_followup"] if rng.random() < .5 else pick_garg("nao")))
            convs.append(("qual", pick_objs("qual"), pick_garg("qual")))
        # idempotência dos rótulos de sem_followup só se "nao"
        lines = []
        lead_id = hash_identifier(nome, secret)
        att = ATENDENTES[0] if rng.random() < .6 else ATENDENTES[1]
        t_cursor = t0
        for seq, (k, objs, garg) in enumerate(convs, start=1):
            rows = _build_conversation(rng, t_cursor, k, objs, garg, noisy, extra, hard)
            if seq == 2:  # na recorrência, a 1ª fala do lead é um retorno
                rows[0][2] = rng.choice(RETORNO)
            lines.extend(rows)
            t_cursor = rows[-1][0] + timedelta(days=rng.randint(2, 9), hours=rng.randint(0, 5))
            labels.append({
                "conversation_id": f"{lead_id}-{seq:03d}", "annotator": "adjudicado",
                "escopo": "fora_de_escopo" if k == "fora" else "em_escopo",
                # v1.1: o sintético gera funil (qual -> agendado; nao -> informativo); o binário bate com a derivação
                "funil": "" if k == "fora" else ("agendado" if k == "qual" else "informativo"),
                "qualificacao": "" if k == "fora" else ("qualificado" if k == "qual" else "nao_qualificado"),
                "objecao": (";".join(o for o in objs if o in OBJ) or "nenhuma") if k != "fora" else "nenhuma",
                "gargalo": ";".join(garg) or "nenhuma",
            })
        lines.sort(key=lambda r: r[0])
        fn = out / f"Conversa do WhatsApp com {nome}.txt"
        header = ("As mensagens e ligações são protegidas com a criptografia de ponta a ponta. Ninguém fora desta conversa pode lê-las."
                  if fmt == "android" else "‎[%s] %s: As mensagens e as chamadas são protegidas com a criptografia de ponta a ponta." % (lines[0][0].strftime("%d/%m/%Y %H:%M:%S"), nome))
        body = []
        if fmt == "android":
            body.append(f"{lines[0][0]:%d/%m/%Y %H:%M} - {header}")
        else:
            body.append(header)
        for ts, role, text in lines:
            sender = nome if role == "lead" else att
            if "\n" not in text and not text.startswith("<") and rng.random() < .05:
                text = text + "\nQualquer dúvida estou por aqui."  # mensagem multilinha
            body.append((_fmt_android if fmt == "android" else _fmt_iphone)(ts, sender, text))
        fn.write_text("\n".join(body) + "\n", encoding="utf-8")
    with open(out / "labels_gold.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["conversation_id", "annotator", "escopo", "funil", "qualificacao", "objecao", "gargalo"])
        w.writeheader(); w.writerows(labels)
    return {"n_leads": n_leads, "n_conversations": len(labels), "dir": str(out), "hard": hard}
