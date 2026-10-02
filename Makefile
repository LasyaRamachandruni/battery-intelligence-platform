.PHONY: install test demo real pipeline dashboard api clean

install:
	pip install -e ".[dev,stream,dashboard]"

test:
	pytest -q

# full pipeline on synthetic cells (no download needed)
demo:
	bip ingest --source synthetic
	$(MAKE) pipeline

# full pipeline on the Severson et al. 2019 cells (see README: put the .mat files in data/raw/)
real:
	bip ingest --source severson
	$(MAKE) pipeline

pipeline:
	bip transform
	bip quality
	bip train
	bip register --promote
	bip drift --split test_primary
	bip drift --batch b3
	bip retrain --batch b3 --registry
	bip stream --cells b2c1,b2c2,b2c3,b2c4,b2c5

dashboard:
	streamlit run dashboard/app.py

api:
	bip serve --registry

clean:
	rm -rf data/lake data/warehouse.duckdb reports artifacts mlruns dbt/target dbt/logs
