.PHONY: kubeval
.DEFAULT_GOAL := help

DC := docker compose
DCR := $(DC) run --rm --cap-add SYS_NICE
CONTAINER := api
export CORPUS_PATH ?= ~/Dropbox/AlzaboMix/ClassicMix
export ALZABO_CONFIG_PATH ?= alzabo.local.yaml

help: ## Print this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-z0-9A-Z_-]+:.*?## / {printf "%-30s%s\n", $$1, $$2}' $(MAKEFILE_LIST)

### DOCKER

build: ## Build Docker images
	$(DC) --progress plain build --build-arg BUILDKIT_INLINE_CACHE=1 $(CONTAINER)

up: up-datastores ## Bring up containers
	ALZABO_CONFIG_PATH="alzabo.server.yaml" $(DC) up -d

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
	python -m alzabo --api-url http://localhost:8000 audio-upload $(CORPUS_PATH)/*.wav

ingest-test: up ## Run the ingest pipeline
	python -m alzabo --api-url http://localhost:8000 audio-upload tests/recordings/*.wav

### CLIENT

client: up ## Run the client
	python -m alzabo --api-url http://localhost:8000 run

client-x: ## Run the client
	python -m alzabo --api-url http://localhost:8000 run

client-es9: ## Run the client with ES-9
	ALZABO_CONFIG_PATH=alzabo.es9.yaml python -m alzabo --api-url http://localhost:8000 run

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
	kubectl kustomize kubernetes/alzabo/base | kubeval --strict

### TESTING

pytest: up-datastores ## Run pytest
	ALZABO_CONFIG_PATH="" $(DCR) --cap-add SYS_NICE $(CONTAINER) pytest

pytest-cov: up-datastores ## Run pytest with coverage
	ALZABO_CONFIG_PATH="" $(DCR) --cap-add SYS_NICE $(CONTAINER) pytest --cov=alzabo --cov-report=html --cov-report=term --durations=10

pytest-x: up-datastores ## Run pytest and fail fast
	ALZABO_CONFIG_PATH="" $(DCR) --cap-add SYS_NICE $(CONTAINER) pytest -x

pytest-sw: up-datastores ## Run pytest and fail fast
	ALZABO_CONFIG_PATH="" $(DCR) --cap-add SYS_NICE $(CONTAINER) pytest --sw

test: reformat lint pytest-cov

### MISC

ensure-buckets:
	$(DCR) $(CONTAINER) python3 -m alzabo ensure-buckets

ensure-database:
	$(DCR) $(CONTAINER) python3 -m alzabo ensure-database

pip-compile: ## Rebuild requirements.txt
	python -m piptools compile
	mv requirements.txt requirements.osx.txt
	$(DCR) $(CONTAINER) python -m piptools compile
	mv requirements.txt requirements.unix.txt
