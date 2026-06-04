# Makefile for the Corporate RAG assessment.
#
# `make all` chains the full setup end-to-end. Individual targets (`make up`,
# `make migrate`, `make ingest`, ...) are available for cost gating — e.g.
# `make setup-free` verifies the no-API-cost path before you authorize spend.
#
# Override the Python binary with PYTHON=python3.12 etc.

.DEFAULT_GOAL := help

PYTHON ?= python

.PHONY: help install up migrate ingest enrich \
        reclassify-subtypes reclassify-pages contextualize extract-claims \
        back-fill setup-free all app test lint typecheck check-pdfs

help:
	@echo "Setup (run in order, or use 'make all' / 'make setup-free'):"
	@echo "  install            pip install -r requirements.txt (run inside your venv)"
	@echo "  up                 Start Postgres+pgvector (docker compose, waits for healthcheck)"
	@echo "  migrate            Create DB schema"
	@echo "  ingest             Parse PDFs in data/pdfs/ and write chunks"
	@echo "  enrich             Vision-extract charts via Anthropic (~\$$2-5 in API)"
	@echo "  back-fill          reclassify_subtypes + reclassify_page_content +"
	@echo "                     contextualize (Batch API) + extract_claims (Batch API)"
	@echo ""
	@echo "Composed targets:"
	@echo "  setup-free         up + migrate + ingest (no API cost; verifies the cheap path)"
	@echo "  all                setup-free + enrich + back-fill (full setup)"
	@echo ""
	@echo "Runtime:"
	@echo "  app                streamlit run app.py"
	@echo ""
	@echo "Dev:"
	@echo "  test               pytest -q"
	@echo "  lint               ruff check"
	@echo "  typecheck          mypy src/"

install:
	pip install -r requirements.txt

up:
	docker compose up -d --wait

migrate:
	$(PYTHON) scripts/migrate.py

check-pdfs:
	@count=$$(ls data/pdfs/*.pdf 2>/dev/null | wc -l); \
	if [ "$$count" = "0" ]; then \
		echo "ERROR: No PDFs found in data/pdfs/. Place the 10 corpus PDFs there before running 'make ingest' or 'make all'."; \
		exit 1; \
	fi

ingest: check-pdfs
	$(PYTHON) scripts/ingest.py

enrich:
	$(PYTHON) scripts/enrich_charts.py

reclassify-subtypes:
	$(PYTHON) scripts/reclassify_subtypes.py

reclassify-pages:
	$(PYTHON) scripts/reclassify_page_content.py

contextualize:
	$(PYTHON) scripts/contextualize.py

extract-claims:
	$(PYTHON) scripts/extract_claims.py

back-fill: reclassify-subtypes reclassify-pages contextualize extract-claims

setup-free: up migrate ingest

all: setup-free enrich back-fill
	@echo ""
	@echo "Setup complete. Run 'make app' to start the Streamlit UI."

app:
	streamlit run app.py

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	ruff check src/ tests/ scripts/ app.py

typecheck:
	$(PYTHON) -m mypy src/ --ignore-missing-imports
