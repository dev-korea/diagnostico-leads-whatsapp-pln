"""Taxonomia única do estudo, v1.1 (usada pelo guia de rotulagem, gerador sintético e classificadores).

v1.1: funil de 4 estágios anotado pelo humano; o binário `qualificacao` (tarefa principal do TCC 1) é DERIVADO
do funil, nunca anotado à parte. Novos: objeção `servico_nao_oferecido`, gargalo `sem_vaga`, metadados
opcionais `desfecho` e `para_terceiro` (descritivos; fora do kappa e dos classificadores).
"""

ESCOPO = ["em_escopo", "fora_de_escopo"]
FUNIL = ["sem_dialogo", "informativo", "intencao", "agendado"]   # ordinal; só para conversas em_escopo
QUALIFICACAO = ["qualificado", "nao_qualificado"]                # DERIVADO do funil; tarefa principal do TCC 1
OBJECOES = ["preco", "horario_disponibilidade", "localizacao", "confianca_resultado",
            "convenio_reembolso", "adiamento", "concorrencia",
            "servico_nao_oferecido"]                             # multirrótulo; vazio = sem objeção
GARGALOS = ["demora_resposta", "sem_followup", "info_incompleta",
            "sem_vaga"]                                          # multirrótulo; vazio = sem gargalo
DESFECHO = ["desmarcou"]                                         # metadado opcional (exige funil = agendado)
PARA_TERCEIRO = ["sim"]                                          # metadado opcional

# funil -> binário (regra única; qualquer outro código deve chamar esta função)
_FUNIL_PARA_QUALIFICACAO = {"sem_dialogo": "nao_qualificado", "informativo": "nao_qualificado",
                            "intencao": "qualificado", "agendado": "qualificado"}


def derive_qualificacao(funil: str) -> str:
    return _FUNIL_PARA_QUALIFICACAO[funil]


DIMENSIONS = {"escopo": ESCOPO, "funil": FUNIL, "qualificacao": QUALIFICACAO, "objecao": OBJECOES,
              "gargalo": GARGALOS, "desfecho": DESFECHO, "para_terceiro": PARA_TERCEIRO}
MULTILABEL = {"objecao", "gargalo"}
TAXONOMY_VERSION = "1.1"

DEFINICOES = {
    "em_escopo": "Pessoa interessada (ou potencialmente interessada) em serviços da clínica.",
    "fora_de_escopo": "Contato que não é cliente potencial: fornecedor, spam, engano, vaga de emprego, cobrança.",
    "sem_dialogo": "Em escopo, mas o cliente não desenvolveu diálogo: só a mensagem automática do site, 'oi' ou mídia sem continuação, ou sumiu logo após a 1ª resposta.",
    "informativo": "Houve diálogo e pedido de informação (preço, como funciona, endereço), mas sem sinal explícito de intenção de agendar.",
    "intencao": "Sinal explícito de intenção de agendar (pede/aceita horário, pergunta 'como marco?', fornece dados para cadastro), sem agendamento fechado na conversa.",
    "agendado": "Agendamento fechado na própria conversa (data e horário combinados/confirmados).",
    "qualificado": "Derivado: funil = intencao ou agendado.",
    "nao_qualificado": "Derivado: funil = sem_dialogo ou informativo.",
    "preco": "Cliente questiona/reclama do valor, pede desconto, acha caro.",
    "horario_disponibilidade": "A AGENDA DO CLIENTE é incompatível (só consegue certo dia/horário). Falta de vaga da clínica é o gargalo sem_vaga.",
    "localizacao": "Distância, deslocamento, estacionamento ou cidade fora da área.",
    "confianca_resultado": "Dúvida sobre eficácia, segurança, formação do profissional, medo de piorar.",
    "convenio_reembolso": "Pergunta/exige plano de saúde, convênio ou reembolso.",
    "adiamento": "Posterga a decisão sem motivo específico ('vou pensar', 'depois te falo').",
    "concorrencia": "Compara com outro profissional/clínica ou já está em tratamento em outro lugar.",
    "servico_nao_oferecido": "Cliente potencial procura um serviço que a clínica não oferece (ex.: RPG, fisioterapia, ventosa). Difere de fora_de_escopo: aqui há interesse real em saúde/tratamento.",
    "demora_resposta": "Atendimento demorou a responder de forma que prejudicou a conversa (ver guia: limiar).",
    "sem_followup": "Conversa terminou sem desfecho (sem agendamento nem encerramento): o cliente parou de responder e o atendente não retomou o contato.",
    "info_incompleta": "Atendente deixou pergunta do cliente sem resposta (preço, horário, endereço…).",
    "sem_vaga": "A CLÍNICA não tinha horário compatível com o que o cliente pediu (o cliente tinha disponibilidade; faltou vaga).",
    "desmarcou": "Metadado: o agendamento feito na conversa foi cancelado/desmarcado na própria conversa. Não rebaixa o funil.",
    "sim": "Metadado para_terceiro: quem escreve contata em nome de outra pessoa (filho, pai, cônjuge).",
}
