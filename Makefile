.PHONY: help setup test smoke data eda tune experiments failure all clean

PY ?= python
CFG ?= configs/config.yaml
SYNTH := configs/config_synth.yaml

help:
	@echo "make setup        install dependencies"
	@echo "make test         run the unit tests"
	@echo "make smoke        full pipeline on synthetic data (~2 min, no download needed)"
	@echo "make data         ingest data/raw/ -> data/processed/ + memory benchmark"
	@echo "make eda          exploratory and time-series analysis"
	@echo "make tune         hyperparameter search on the validation week"
	@echo "make experiments  final runs across the three areas"
	@echo "make failure      failure analysis on the saved predictions"
	@echo "make all          data -> eda -> experiments -> failure"

setup:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -q

smoke:
	$(PY) scripts/00_make_synthetic_data.py --days 21 --squares 60
	$(PY) scripts/01_build_dataset.py      --config $(SYNTH)
	$(PY) scripts/02_run_eda.py            --config $(SYNTH)
	$(PY) scripts/03_run_experiments.py    --config $(SYNTH)
	$(PY) scripts/04_failure_analysis.py   --config $(SYNTH)

data:
	$(PY) scripts/01_build_dataset.py --config $(CFG)

eda:
	$(PY) scripts/02_run_eda.py --config $(CFG)

tune:
	$(PY) scripts/03_run_experiments.py --config $(CFG) --tune

experiments:
	$(PY) scripts/03_run_experiments.py --config $(CFG)

failure:
	$(PY) scripts/04_failure_analysis.py --config $(CFG)

all: data eda experiments failure

clean:
	rm -rf results_synth data/processed_synth data/raw_synth
	find . -name __pycache__ -type d -exec rm -rf {} +
