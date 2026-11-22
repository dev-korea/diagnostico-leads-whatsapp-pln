---
title: "Guia de rotulagem — conversas de WhatsApp (clínica de quiropraxia)"
subtitle: "Versão 1.1 — TCC: agente de IA para diagnóstico de leads"
---

# 1. Objetivo e unidade de análise

Este guia garante que **todos os anotadores rotulem do mesmo jeito**. Os rótulos produzidos formam o
*conjunto-ouro* usado para treinar e avaliar os três métodos (TF-IDF + regressão logística, LLM *few-shot* e
BERTimbau).

- **Unidade:** uma *conversa* = sequência de mensagens do mesmo cliente sem pausa maior que 24 h
  (a coluna `seq_in_lead` > 1 indica que o cliente voltou depois de uma pausa).
- **Cada conversa é rotulada isoladamente**, apenas com o que aparece nela. Não use o que acontece em conversas
  *posteriores* do mesmo cliente (evita vazamento do futuro para o rótulo).
- Algumas conversas **começam no meio** (efeito da segmentação ou de exportação parcial). Rotule com o que está
  visível e registre em `observacao`.
- O texto já está pseudonimizado: `[NOME]`, `[TELEFONE]`, `[ENDERECO]`, `[MIDIA]` etc. são marcadores, não erros.
  A palavra "quiropraxia" também aparece como `[NOME]` (faz parte do nome da clínica).
  **Nunca tente descobrir quem é a pessoa.** Se encontrar um nome próprio que escapou, anote em `observacao`.

# 2. Como preencher a planilha

Para cada linha (conversa), preencha as colunas abaixo. Use `observacao` para dúvidas e casos estranhos.

| Coluna | Valores permitidos | Obrigatória |
|---|---|---|
| `escopo` | `em_escopo` ou `fora_de_escopo` | sempre |
| `funil` | `sem_dialogo`, `informativo`, `intencao` ou `agendado` | **só** se `em_escopo` (vazia se fora de escopo) |
| `objecao` | um ou mais, separados por `;` — ou `nenhuma` | sempre |
| `gargalo` | um ou mais, separados por `;` — ou `nenhuma` | sempre |
| `desfecho` | `desmarcou` ou vazio | opcional (só com `funil = agendado`) |
| `para_terceiro` | `sim` ou vazio | opcional |

Regras de formato: sem espaços extras; use exatamente os nomes da tabela (minúsculas, sem acento);
`nenhuma` nunca vem junto de outro valor. Linha totalmente em branco = "ainda não rotulada" (será ignorada).
Exemplo de objeção: `preco;adiamento`.

**Não existe coluna `qualificacao`.** O rótulo binário qualificado/não qualificado, que é a tarefa principal do
estudo, é **derivado automaticamente do funil** (seção 4). Isso evita que o mesmo julgamento seja anotado duas
vezes de forma contraditória.

**Coluna `pre_rotulo` (só na planilha do anotador a1):**

- `sugerido:...` — a linha veio com uma sugestão gerada por modelo de linguagem. **A sugestão não é o rótulo.**
  Leia a conversa, corrija o que discordar e apague o que estiver errado; o que fica na planilha é decisão sua.
  As observações que terminam em "revisar" marcam casos em que a própria sugestão é incerta.
- `cega` — a linha está em branco de propósito. Essas conversas formam o subconjunto de concordância e de teste:
  rotule do zero, sem consultar ninguém nem nenhuma ferramenta.
- O anotador a2 recebe apenas o subconjunto cego e nunca vê sugestões.

# 3. Passo 1 — Escopo

Pergunta: **essa pessoa poderia se tornar cliente da clínica?**

| Rótulo | Definição | Exemplos |
|---|---|---|
| `em_escopo` | Pessoa interessada, ou potencialmente interessada, nos serviços da clínica (inclui dúvidas, preços, dor, agendamento). | "Vocês atendem hérnia de disco?" · "Quanto custa a avaliação?" |
| `fora_de_escopo` | Contato que não é cliente potencial: fornecedor, spam, engano, vaga de emprego, cobrança. | "Represento uma empresa de software…" · "Desculpa, errei o número" · "Tem vaga? Segue meu currículo" |

**Casos-limite**

- Paciente atual pedindo remarcação ou tirando dúvida sobre sessão → `em_escopo`.
- Pessoa que procura um serviço de saúde que a clínica **não oferece** (acupuntura, RPG, ventosa, mesa de tração,
  só massagem) → `em_escopo`, com a objeção `servico_nao_oferecido`. Há interesse real em tratamento.
- Pedido totalmente alheio (ex.: indicação de dentista, venda de produto) → `fora_de_escopo`.
- Só a mensagem automática do site, "oi" ou mídia sem continuação → `em_escopo` com `funil = sem_dialogo`.

# 4. Passo 2 — Funil (apenas em escopo)

Pergunta: **qual foi o ponto mais avançado que o cliente alcançou nesta conversa?** A escala é ordinal.

| Rótulo | Definição | Sinais típicos |
|---|---|---|
| `sem_dialogo` | O cliente não desenvolveu diálogo. | Só a mensagem automática do site; só "oi"/"bom dia"; só mídia; sumiu logo após a 1ª resposta do atendente. |
| `informativo` | Houve diálogo e pedido de informação, **sem** sinal explícito de intenção de agendar. | Pergunta preço, como funciona, endereço, convênio, dias e horário **de funcionamento**; agradece e encerra. |
| `intencao` | Sinal explícito de intenção de agendar, **sem** agendamento fechado na conversa. | "Quero agendar"; pede ou aceita um horário específico ("tem horário hoje?", "pode ser às 15h"); pergunta "como marco?"; fornece dados para cadastro. |
| `agendado` | Agendamento **fechado na própria conversa**: dia e horário combinados. | Cliente aceita o horário e o atendente confirma ("está agendado"), ou o cliente confirma o horário proposto ("pode agendar"). |

**Derivação automática da qualificação** (não preencha; o sistema calcula):

| Funil | Qualificação derivada |
|---|---|
| `sem_dialogo`, `informativo` | `nao_qualificado` |
| `intencao`, `agendado` | `qualificado` |

**Regra de ouro:** classifique pelo **ponto mais avançado alcançado nesta conversa**. Se o cliente chegou a
pedir ou aceitar horário → no mínimo `intencao`, mesmo que depois suma ou adie.

**Casos-limite**

| Situação | Funil | Por quê |
|---|---|---|
| "Quanto custa?" → atendente responde → "ok, obrigado" | `informativo` | Só informação de preço. |
| "Vocês atendem sábado? Até que horas?" e nada mais | `informativo` | Pergunta de funcionamento não é pedido de horário. |
| "Tem horário para hoje?" → não há vaga → cliente encerra | `intencao` | Pediu horário (intenção explícita); gargalo `sem_vaga`. |
| "Quero marcar" → atendente oferece horários → cliente some | `intencao` | Pediu agendamento. |
| "Vou ver com meu marido e te retorno", sem ter pedido horário | `informativo` | Não avançou; objeção `adiamento`. |
| "Vou me organizar para agendar semana que vem" | `intencao` | Declara a intenção de agendar, ainda que fraca; objeção `adiamento`. Registre "intenção fraca" em `observacao`. |
| Aceita horário e passa o nome; atendente responde "está agendado" | `agendado` | Fechado na conversa. |
| Aceita horário, mas o atendente não chega a confirmar | `agendado` | Cliente confirmou o horário proposto; registre em `observacao`. |
| Conversa de retorno em que o cliente cancela consulta já marcada | `agendado` + `desfecho = desmarcou` | O cancelamento não rebaixa o funil; é registrado no metadado. |
| Conversa de retorno só com "oi" e sem continuação | `sem_dialogo` | Cada conversa é avaliada por si só. |

# 5. Passo 3 — Objeções (multirrótulo)

Marque **toda objeção levantada pelo cliente**, mesmo que o atendente a tenha resolvido. Só conta o que o
**cliente** diz (fala do atendente não conta). Conversa sem objeção → `nenhuma`.

| Rótulo | Definição | Exemplos de fala do cliente |
|---|---|---|
| `preco` | Questiona o valor, pede desconto, acha caro, adia por falta de dinheiro. | "Achei caro" · "Tem desconto à vista?" · "Assim que tiver o dinheiro eu marco" |
| `horario_disponibilidade` | A **agenda do cliente** é incompatível com o que a clínica oferece. | "Só consigo sábado" · "Saio do trabalho às 19h" · "Amanhã estarei viajando" |
| `localizacao` | Distância, deslocamento, estacionamento, cidade fora da área. | "Fica longe" · "Moro em outra cidade" |
| `confianca_resultado` | Dúvida sobre eficácia, segurança, formação, medo de piorar. | "Funciona mesmo?" · "Será que ajuda no meu caso?" · "Tenho medo de piorar" |
| `convenio_reembolso` | Quer plano de saúde, convênio ou reembolso. | "Atende pela Unimed?" · "Aceita Bradesco?" |
| `adiamento` | Posterga a decisão. | "Vou pensar" · "Depois te falo" · "Vou ver com meu esposo" |
| `concorrencia` | Compara com outro profissional/clínica ou está em tratamento em outro lugar **agora**. | "Estou vendo outra clínica mais barata" · "Já faço fisioterapia" |
| `servico_nao_oferecido` | Procura um serviço que a clínica não oferece. | "Vocês fazem acupuntura?" · "Preciso de RPG" · "Tem mesa de tração?" |

**Regras de decisão**

1. **Pergunta simples não é objeção.** "Onde fica a clínica?" = `nenhuma`. "Atende sábado?" seguido de
   continuação normal = `nenhuma`; se o cliente **desiste ou adia por causa disso**, marque `horario_disponibilidade`.
2. **Parcelamento e cartão** só são `preco` se há sinal de dificuldade ou resistência; como mera forma de
   pagamento = `nenhuma`.
3. **"Vou pensar" com motivo** → marque o motivo **e** `adiamento` (ex.: "vou pensar, está caro" =
   `adiamento;preco`). Sem motivo → só `adiamento`.
4. **Agenda do cliente × agenda da clínica.** Cliente não pode nos horários oferecidos → objeção
   `horario_disponibilidade`. Clínica não tem o horário que o cliente pediu → gargalo `sem_vaga` (seção 6).
   Os dois podem ocorrer na mesma conversa.
5. Ter feito quiropraxia **no passado** em outro lugar não é `concorrencia`.
6. Se a mesma objeção aparece várias vezes, marque uma vez.
7. Em dúvida entre duas categorias, marque a mais específica e registre em `observacao`.

# 6. Passo 4 — Gargalos de atendimento (multirrótulo)

Gargalo = falha **do atendimento** que pode ter custado o lead. Sem falha → `nenhuma`.

| Rótulo | Definição | Como decidir |
|---|---|---|
| `demora_resposta` | O atendente demorou a responder. | Regra objetiva: marque se `primeira_resposta_min` **ou** `mediana_resposta_min` for **> 30 min**, ou se ambas estiverem vazias (o atendente nunca respondeu). Não use julgamento. Se o tempo for claramente um artefato (ex.: conversa só com mensagem do atendente), mantenha a regra e registre em `observacao`. |
| `sem_followup` | O cliente **demonstrou intenção** (`funil = intencao`), a conversa parou sem desfecho e o atendente **não retomou** o contato. | A última palavra ficou com o atendente (oferta de horário ou pergunta sem resposta) e não há nova tentativa depois. **Não** marque se: o atendente retomou ao menos uma vez; a conversa teve encerramento cordial ("fico à disposição" após o cliente dizer que retorna); o funil é `sem_dialogo`, `informativo` ou `agendado`. |
| `info_incompleta` | O cliente fez uma pergunta e ela ficou **sem resposta** até o fim da conversa. | Preço, horário, endereço, forma de pagamento, convênio. |
| `sem_vaga` | A **clínica** não tinha o horário/dia que o cliente pediu. | "Tem horário hoje?" → "hoje não"; pediu sábado para agendar e não há atendimento; agenda do mês ainda não aberta. Vale mesmo que o atendente ofereça outra data. Não marque se o agendamento acabou fechado na conversa em horário próximo ao pedido. |

A regra restrita de `sem_followup` (exige intenção) é uma decisão desta versão: conversas informativas em que o
cliente simplesmente não responde são muito frequentes e não distinguem falha de atendimento de desinteresse.

# 7. Metadados opcionais

- `desfecho = desmarcou`: o agendamento foi cancelado ou desmarcado na própria conversa. Exige `funil = agendado`.
- `para_terceiro = sim`: quem escreve contata em nome de outra pessoa (filho, pai, cônjuge, namorado).

Esses dois campos são descritivos: não entram no cálculo de concordância nem nos classificadores.

# 8. Fluxo de trabalho e controle de qualidade

1. **Piloto (30 conversas abertas):** os dois anotadores (`a1`, `a2`) rotulam as mesmas 30 conversas **não cegas**,
   discutem divergências e **ajustam este guia** (registre mudanças no histórico abaixo). Essas 30 não entram no
   cálculo de concordância.
2. **Subconjunto cego (cerca de 25 % dos leads, sorteado por lead com semente fixa):** `a1` e `a2` rotulam de
   forma independente, sem sugestão de modelo e sem consultar o outro. É desse subconjunto que saem a
   concordância e o **conjunto de teste**.
3. **Demais conversas:** rotuladas por `a1` com apoio das sugestões (seção 2). Vão para treino e validação.
4. **Concordância:** κ de Cohen por dimensão (`python -m tcc.pipeline agreement --db ...`); para o funil,
   κ ponderado linear. Meta: **κ ≥ 0,60**; ≥ 0,80 é excelente. Abaixo de 0,60 em alguma dimensão → reunião e
   revisão da definição.
5. **Adjudicação:** divergências do subconjunto cego são resolvidas por consenso (ou pelo orientador) e gravadas
   como `adjudicado`, que é o **ouro** usado nos experimentos.
6. **Ordem embaralhada:** a planilha já vem embaralhada para evitar viés de sequência. Não reordene.
7. **Cegamento:** não rotule olhando previsões de nenhum modelo, exceto as sugestões marcadas em `pre_rotulo`.
8. **Divisão treino/validação/teste:** feita **depois** da adjudicação, por lead, com semente fixa e o
   subconjunto cego como teste (`make-splits --test-csv`). Ajustes ao guia depois disso exigem re-rotular o que
   mudou **e** registrar no histórico.

# 9. Histórico de versões

| Versão | Data | Mudança |
|---|---|---|
| 1.0 | 2026-11-09 | Versão inicial, antes do piloto. `qualificacao` anotada diretamente. |
| 1.1 | 2026-11-16 | `funil` de quatro estágios substitui a anotação direta de `qualificacao` (agora derivada). Novas classes: objeção `servico_nao_oferecido`, gargalo `sem_vaga`. Metadados `desfecho` e `para_terceiro`. `horario_disponibilidade` restrita à agenda do cliente. `sem_followup` restrito a conversas com intenção. Coluna `pre_rotulo` e subconjunto cego como teste. |
