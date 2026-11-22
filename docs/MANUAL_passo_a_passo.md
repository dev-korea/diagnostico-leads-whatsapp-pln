# Manual passo a passo — dos rótulos aos resultados (Windows / PowerShell)

Estado em 16/11/2026: 169 conversas rotuladas (anotação única, sugestões revisadas e aprovadas) no banco como
`a1` e `adjudicado`; divisão `v1` feita (101 treino / 35 validação / 33 teste); TF-IDF executado. Faltam o LLM
few-shot e o BERTimbau. As 54 conversas "cegas" seguem sem rótulo, reservadas para dupla anotação futura.

## 0. Ambiente (em cada terminal novo)

```powershell
cd <pasta-do-projeto>
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
```

Nunca escreva `TCC_SECRET`, chave de API ou senha do banco em arquivo do projeto.

## 1. Privacidade (pendente no Supabase)

O banco local já teve os nomes mascarados em segunda passada. A cópia no Supabase ainda tem o texto antigo:

```powershell
$env:SUPABASE_DB_URL = "<url-do-session-pooler>"
python -m tcc.supabase_sync update-text --db data/processed/tcc.db
```

Depois de conferir, apague `data\processed\tcc.antes_scrub.bak.db` (contém os nomes).

## 2. O que já foi executado (para refazer, se algum rótulo mudar)

```powershell
python -m tcc.pipeline import-labels --db data/processed/tcc.db --csv data/processed/rotular_a1.csv --annotator a1
python -m tcc.pipeline build-gold --db data/processed/tcc.db
python -m tcc.pipeline make-splits --db data/processed/tcc.db --name v1 --seed 42
python -m tcc.experiments.tfidf --db data/processed/tcc.db --task qualificacao
python -m tcc.experiments.tfidf --db data/processed/tcc.db --task objecao
python scripts/descrever_corpus.py
python scripts/sensibilidade_segmentacao.py
```

Atenção: `build-gold` não sobrescreve conversas que já estão no ouro. Para trocar rótulos depois de aprovados,
apague antes a camada antiga:
`python -c "import sqlite3; c=sqlite3.connect('data/processed/tcc.db'); c.execute(\"delete from labels where annotator='adjudicado'\"); c.commit()"`.
A divisão `v1` não muda se o rótulo de qualificação das conversas não mudar.

## 3. LLM few-shot

**Opção A — API da Anthropic (recomendada; segundos por conversa, custo de centavos para 33 conversas):**

```powershell
pip install anthropic
$env:ANTHROPIC_API_KEY = "<sua-chave>"
python -m tcc.experiments.llm --db data/processed/tcc.db --task qualificacao --provider anthropic --model claude-sonnet-5-5 --k 3 --limit 5
python -m tcc.experiments.llm --db data/processed/tcc.db --task qualificacao --provider anthropic --model claude-sonnet-5-5 --k 3
```

A primeira linha testa com 5 conversas; a segunda roda as 33 (as 5 já feitas vêm do cache). O texto enviado é o
pseudonimizado. Registre na Metodologia o modelo exato e a política de retenção do provedor.

**Opção B — modelo aberto local (Ollama; o texto não sai da máquina):**

```powershell
python -m tcc.experiments.llm --db data/processed/tcc.db --task qualificacao --provider ollama --model qwen2.5:1.5b --k 3 --workers 1
```

Neste notebook (8 GB de RAM, sem GPU) levou cerca de 13 minutos por conversa: inviável aqui. Só vale em máquina
com GPU ou mais memória, e com um modelo maior que o `qwen2.5:1.5b`.

## 4. BERTimbau (Google Colab, GPU)

```powershell
python scripts/empacotar_colab.py --com-banco
```

Abra `notebooks\colab_bertimbau.ipynb` no Colab, ative a GPU e execute as células em ordem: enviar
`dist\colab_pacote.zip`, treinar (5 sementes), baixar `preds_bertimbau_qualificacao.csv`. Copie o arquivo baixado
para `results\` e rode a última célula do notebook para apagar o banco do Colab.

## 5. Tabela final

```powershell
python -m tcc.experiments.evaluate --db data/processed/tcc.db --task qualificacao
```

Gera `results\summary_qualificacao.md` (F1 macro com IC 95 %), `perclass_qualificacao.csv` e
`pairs_qualificacao.csv` (diferença pareada, McNemar, Holm).

## 6. Levar para o texto

| Saída | Vai para |
|---|---|
| `results\summary_qualificacao.md` | Tabela de desempenho em `docs\TEXTO-Resultados-esqueleto.tex` (linhas do LLM e do BERTimbau) |
| `results\pairs_qualificacao.csv` | Tabela de comparações pareadas |
| `docs\TEXTO-Metodologia-Pipeline.tex` | substitui a seção "O resto aqui" da Metodologia |
| `docs\TEXTO-Taxonomia-4.2.2.tex` | subseção de taxonomia (guia v1.1) |

O que já está preenchido no capítulo de Resultados: distribuição do corpus, objeções e gargalos, TF-IDF,
regra de `sem_followup` e sensibilidade da segmentação.

## 7. Etapa futura: dupla anotação das 54 conversas reservadas

`data\processed\rotular_a2_cego.csv` e as linhas `cega` de `rotular_a1.csv` estão em branco. Quando houver segundo
anotador: os dois rotulam, `import-labels` para `a1` e `a2`, `agreement` (kappa), `export-disagreements`,
adjudicação e `build-gold`. Esse conjunto serve de teste sem influência de modelo de linguagem.
