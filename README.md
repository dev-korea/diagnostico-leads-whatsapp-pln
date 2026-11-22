# Pipeline do TCC — diagnóstico de leads de WhatsApp com PLN

Código para: (1) ler exportações `.txt` do WhatsApp, (2) pseudonimizar (LGPD), (3) guardar em banco,
(4) segmentar em conversas, (5) rotular com concordância entre anotadores, (6) comparar três métodos
(TF-IDF + regressão logística · LLM *few-shot* · BERTimbau) no **mesmo conjunto de teste**, com IC e McNemar.

> **Dados sintéticos servem só para testar o código.** Métricas obtidas neles **não podem** ir para o TCC.

## Instalação
```bash
pip install -r requirements.txt
export PYTHONPATH=src
export TCC_SECRET="uma-chave-longa-e-secreta"   # NUNCA no repositório; guarde num cofre/gerenciador de senhas
make test        # 83 testes
make demo        # pipeline completo com dados sintéticos (sem API, sem GPU)
```
Perca a `TCC_SECRET` e os `lead_id` não podem mais ser recalculados (re-ingestão não bate com o banco existente);
vazá-la permite testar telefones conhecidos contra os hashes. Trate como senha.

## Fluxo com dados reais
```bash
# 1) exporte cada conversa no celular: "Exportar conversa" > "Sem mídia"; coloque os .txt em data/raw/
# 2) crie o banco e importe (cria/atualiza leads, conversas e mensagens; reimportar é seguro)
python -m tcc.cli ingest --db data/processed/tcc.db --input data/raw \
    --attendants "Nome do Atendente 1,Nome do Atendente 2" --names "Outros,Nomes,A,Mascarar" --gap-hours 24

# 3) revise vazamentos de nomes (palavras capitalizadas que sobraram)
python - <<'PY'
import sqlite3; from tcc.pseudonymize import Pseudonymizer
c = sqlite3.connect("data/processed/tcc.db")
print(Pseudonymizer("x"*16).review_candidates(r[0] for r in c.execute("select text from messages")))
PY
#    -> adicione nomes encontrados em --names e reimporte do zero (apague o .db)

# 4) rotulagem (ver docs/guia_rotulagem.md)
python -m tcc.pipeline export-annotation --db data/processed/tcc.db --out rotular.csv
#    cada anotador preenche uma cópia; depois:
python -m tcc.pipeline import-labels --db data/processed/tcc.db --csv rotular_a1.csv --annotator a1
python -m tcc.pipeline import-labels --db data/processed/tcc.db --csv rotular_a2.csv --annotator a2
python -m tcc.pipeline agreement --db data/processed/tcc.db           # kappa + nº de divergências
python -m tcc.pipeline import-labels --db data/processed/tcc.db --csv adjudicado.csv --annotator adjudicado

# 5) divisão treino/val/teste fixa (por lead) — faça UMA vez e versione
python -m tcc.pipeline make-splits --db data/processed/tcc.db --name v1 --seed 42

# 6) experimentos (para cada tarefa: escopo | qualificacao | objecao)
python -m tcc.experiments.tfidf --db data/processed/tcc.db --task qualificacao
python -m tcc.experiments.llm   --db data/processed/tcc.db --task qualificacao \
    --provider anthropic --model claude-sonnet-5-5 --k 3 --limit 5     # teste de custo com 5 conversas
python -m tcc.experiments.llm   --db data/processed/tcc.db --task qualificacao --provider anthropic --model claude-sonnet-5-5 --k 3
python -m tcc.experiments.bert  --db data/processed/tcc.db --task qualificacao   # GPU (Colab/Kaggle)
python -m tcc.experiments.evaluate --db data/processed/tcc.db --task qualificacao  # tabela + IC + McNemar
```
Para a API: `export ANTHROPIC_API_KEY=...` (ou `OPENAI_API_KEY`). Confira antes o contrato de retenção/treino
do provedor, pois, mesmo pseudonimizado, o texto sai da sua máquina (cite isso na Metodologia/LGPD).

## Estrutura
| Caminho | Função |
|---|---|
| `sql/schema.sql` | leads, imports, conversations, messages, labels, splits, predictions |
| `src/tcc/parser.py` | Android/iPhone, 12h/24h, multilinha, sistema/mídia/apagada |
| `src/tcc/pseudonymize.py` | HMAC do identificador; mascara e-mail, link, CPF, CNPJ, telefone, CEP, cartão (Luhn), @, endereço, nomes |
| `src/tcc/segment.py` | janela de 24 h, tempos de resposta |
| `src/tcc/ingest.py` | ingestão idempotente (por arquivo e por mensagem), IDs de conversa estáveis |
| `src/tcc/taxonomy.py` | **fonte única** das classes (guia, gerador, prompts) |
| `src/tcc/labels.py` | exportar/importar rótulos, validação, κ de Cohen, divergências |
| `src/tcc/splits.py` | divisão estratificada **agrupada por lead** |
| `src/tcc/scope.py` | Etapa 1 do pipeline: escopo determinístico por léxico (`catalog_terms`) |
| `src/tcc/rules.py` | gargalos por regra (tempo de resposta, sem follow-up) |
| `src/tcc/experiments/` | `tfidf.py`, `llm.py`, `bert.py`, `evaluate.py` |
| `src/tcc/metrics.py` | F1, bootstrap por lead, diferença pareada, McNemar exato, Holm |
| `src/tcc/scrub.py` | segunda passada de pseudonimização, direto no banco (nomes por contexto) |
| `src/tcc/prelabel.py` | sugestões de rótulo por LLM para a rotulagem semiautomática |
| `src/tcc/supabase_sync.py` | envia/recebe dados pseudonimizados do Supabase (interface de rotulagem) |
| `src/tcc/synthetic.py` | exportações sintéticas para os testes (nunca para métricas do TCC) |
| `sql/schema_postgres.sql` | esquema do Supabase (taxonomia v1.1) |
| `scripts/` | preparo da rotulagem, análises do TCC 1, sensibilidade da segmentação, matrizes de confusão, pacote do Colab |
| `notebooks/colab_tcc1_experimentos.ipynb` | BERTimbau e LLM *few-shot* no Google Colab (GPU T4) |
| `results/` | métricas agregadas (`summary_*`, `pairs_*`, `perclass_*`, `analises_*.json`) e `figuras/` |
| `docs/guia_rotulagem.md` | guia de anotação (v1.1) |
| `docs/MANUAL_passo_a_passo.md` | passo a passo no Windows/PowerShell, dos rótulos aos resultados |

Fora do versionamento (`.gitignore`): `data/` (conversas, banco, planilhas de rotulagem), previsões por conversa
(`results/preds_*.csv`), logs, cache do LLM e o pacote do Colab (`dist/`), que contém o banco.

## Decisões metodológicas (cite na Metodologia)
- **Segmentação:** pausa > 24 h abre nova conversa (`--gap-hours`); recomenda-se análise de sensibilidade (12/24/48 h).
- **Sem vazamento:** a divisão é por *lead*; exemplos *few-shot* vêm só do treino; hiperparâmetros só na validação.
- **Inferência:** F1-macro com IC 95 % por *bootstrap* de leads (2000 reamostragens); McNemar exato pareado com Holm;
  BERTimbau com 5 sementes (média ± desvio; voto majoritário no McNemar).
- **LLM:** temperatura 0, versão fixa registrada, cache em disco, JSON validado, inválidas contabilizadas.
- **Truncamento (BERT):** 256 tokens, cabeça + cauda (25 % / 75 %).
- **Limites conhecidos:** nomes próprios não-listados podem escapar das regex (use `review_candidates` + revisão humana);
  `sem_followup` por regra é heurística; o timestamp do Android tem resolução de minutos (mensagens idênticas no
  mesmo minuto recebem contador).

## Supabase (PostgreSQL) e tabela de rotulagem online
O processamento continua local (SQLite, reprodutível); o Supabase guarda os dados **já pseudonimizados** e
serve para rotular em planilha online.
```bash
# 1) Supabase > SQL Editor: cole e execute sql/schema_postgres.sql (cria tabelas, CHECKs, views e RLS)
# 2) Project Settings > Database > Connection string (Session pooler):
export SUPABASE_DB_URL="postgresql://postgres.<ref>:<senha>@<host>:5432/postgres"   # nunca no repositório
# 3) envie os dados e crie a fila de rotulagem do anotador a1
python -m tcc.supabase_sync push --db data/processed/tcc.db --annotator a1
# 4) rotule no Table Editor: tabela `rotulagem` (leia o texto na view `v_conversas_texto`);
#    ao terminar cada conversa, mude status para 'rotulada'
# 5) traga os rótulos de volta para o banco local e siga o fluxo (agreement, make-splits, experimentos)
python -m tcc.supabase_sync pull-labels --db data/processed/tcc.db
```
