# Uso: make demo   (pipeline completo com dados SINTÉTICOS, sem API e sem GPU)
PY=python3
export PYTHONPATH=src
export TCC_SECRET?=segredo-de-teste-123456
DB=data/processed/demo.db
RES=results/demo

demo:
	mkdir -p data/processed results
	rm -rf data/synthetic/demo $(RES) $(DB)
	$(PY) -c "from tcc.synthetic import generate; print(generate('data/synthetic/demo', n_leads=150, seed=7, secret='$(TCC_SECRET)'))"
	$(PY) -m tcc.cli init-db --db $(DB)
	$(PY) -m tcc.cli ingest --db $(DB) --input data/synthetic/demo --attendants "Clinica Vida Plena,Helena Recepcao"
	$(PY) -m tcc.pipeline import-labels --db $(DB) --csv data/synthetic/demo/labels_gold.csv --annotator adjudicado
	$(PY) -m tcc.pipeline make-splits --db $(DB) --name v1 --seed 42
	for t in escopo qualificacao objecao; do \
	  $(PY) -m tcc.experiments.tfidf --db $(DB) --task $$t --out $(RES); \
	  $(PY) -m tcc.experiments.llm --db $(DB) --task $$t --provider mock --k 3 --out $(RES) --cache $(RES)/llm_cache.sqlite; \
	  $(PY) -m tcc.experiments.bert --db $(DB) --task $$t --smoke --seeds 0 1 --epochs 3 --out $(RES); \
	  $(PY) -m tcc.experiments.evaluate --db $(DB) --task $$t --results $(RES) --n-boot 500; \
	done

# Conjunto sintético MAIOR e MAIS DIFÍCIL (falas implícitas, erros de digitação, rajadas, casos limítrofes de escopo)
DBH=data/processed/demo_hard.db
RESH=results/demo_hard
demo-hard:
	mkdir -p data/processed results
	rm -rf data/synthetic/hard $(RESH) $(DBH)
	$(PY) -c "from tcc.synthetic import generate; print(generate('data/synthetic/hard', n_leads=300, seed=11, secret='$(TCC_SECRET)', hard=True))"
	$(PY) -m tcc.cli init-db --db $(DBH)
	$(PY) -m tcc.cli ingest --db $(DBH) --input data/synthetic/hard --attendants "Clinica Vida Plena,Helena Recepcao"
	$(PY) -m tcc.pipeline import-labels --db $(DBH) --csv data/synthetic/hard/labels_gold.csv --annotator adjudicado
	$(PY) -m tcc.pipeline make-splits --db $(DBH) --name v1 --seed 42
	for t in escopo qualificacao objecao; do \
	  $(PY) -m tcc.experiments.tfidf --db $(DBH) --task $$t --out $(RESH); \
	  $(PY) -m tcc.experiments.llm --db $(DBH) --task $$t --provider mock --k 3 --out $(RESH) --cache $(RESH)/llm_cache.sqlite; \
	  $(PY) -m tcc.experiments.evaluate --db $(DBH) --task $$t --results $(RESH) --n-boot 500; \
	done

test:
	$(PY) -m pytest -q
