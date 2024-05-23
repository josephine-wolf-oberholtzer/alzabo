.PHONY: kubeval
.DEFAULT_GOAL := help

DC := docker compose
DCR := $(DC) run --rm
CONTAINER := api
export CORPUS_PATH ?= ~/Dropbox/PraetorMix/ClassicMix
export PRAETOR_CONFIG_PATH ?= praetor.local.yaml

help: ## Print this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-z0-9A-Z_-]+:.*?## / {printf "%-30s%s\n", $$1, $$2}' $(MAKEFILE_LIST)

### DOCKER

build: ## Build Docker images
	$(DC) --progress plain build --build-arg BUILDKIT_INLINE_CACHE=1 $(CONTAINER)

up: up-datastores ## Bring up containers
	PRAETOR_CONFIG_PATH="praetor.server.yaml" $(DC) up -d

up-datastores: ## Bring up Milvus containers
	$(DC) up -d attu etcd milvus minio redis
	make ensure-buckets
	make ensure-database

down: ## Bring down containers
	$(DC) down

clean: down ## Clean out temp data
	rm -Rf ./tmp/etcd/
	rm -Rf ./tmp/redis/
	rm -Rf ./tmp/minio/milvus/
	rm -Rf ./tmp/milvus/

clean-s3: clean  ## Clean out S3 files too
	rm -Rf ./tmp/minio/*

push: ## Push image to GHCR
	$(DC) push $(CONTAINER)

### INGEST

ingest: up ## Run the ingest pipeline
	python -m praetor --api-url http://localhost:8000 audio-upload $(CORPUS_PATH)/*.wav

ingest-test: up ## Run the ingest pipeline
	python -m praetor --api-url http://localhost:8000 audio-upload tests/recordings/*.wav

### CLIENT

client: up ## Run the client
	python -m praetor --api-url http://localhost:8000 run

client-x: ## Run the client
	python -m praetor --api-url http://localhost:8000 run

client-es9: ## Run the client with ES-9
	PRAETOR_CONFIG_PATH=praetor.es9.yaml python -m praetor --api-url http://localhost:8000 run

### FORMATTING

reformat: isort black ## Reformat codebase

black: ## Reformat via black
	$(DCR) $(CONTAINER) black .

isort: ## Reformat via isort
	$(DCR) $(CONTAINER) isort .

### LINTING

lint: black-check isort-check flake8 mypy ## Lint codebase

black-check: ## Check syntax via black
	$(DCR) $(CONTAINER) black --check --diff .

isort-check: ## Check syntax via isort
	$(DCR) $(CONTAINER) isort --check --diff .

flake8: ## Lint via flake8
	$(DCR) $(CONTAINER) flake8 .

mypy: ## Type-check via mypy
	$(DCR) $(CONTAINER) mypy .

kubeval:
	kubectl kustomize kubernetes/praetor/base | kubeval --strict

### TESTING

pytest: up-datastores ## Run pytest
	PRAETOR_CONFIG_PATH="" $(DCR) $(CONTAINER) pytest

pytest-cov: up-datastores ## Run pytest with coverage
	PRAETOR_CONFIG_PATH="" $(DCR) $(CONTAINER) pytest --cov=praetor --cov-report=html --cov-report=term --durations=10

pytest-x: up-datastores ## Run pytest and fail fast
	PRAETOR_CONFIG_PATH="" $(DCR) $(CONTAINER) pytest -x

pytest-sw: up-datastores ## Run pytest and fail fast
	PRAETOR_CONFIG_PATH="" $(DCR) $(CONTAINER) pytest --sw

test: reformat lint pytest-cov

### MISC

ensure-buckets:
	$(DCR) $(CONTAINER) python3 -m praetor ensure-buckets

ensure-database:
	$(DCR) $(CONTAINER) python3 -m praetor ensure-database

pip-compile: ## Rebuild requirements.txt
	python -m piptools compile
	mv requirements.txt requirements.osx.txt
	$(DCR) $(CONTAINER) python -m piptools compile
	mv requirements.txt requirements.unix.txt
