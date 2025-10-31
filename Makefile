.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")

.PHONY: help
help:             	## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: venv
venv:			## Create a virtual environment
	@echo "Creating virtualenv ..."
	@rm -rf .venv
	@python3 -m venv .venv
	@./.venv/bin/pip install -U pip
	@echo
	@echo "Run 'source .venv/bin/activate' to enable the environment"

.PHONY: install
install:		## Install dependencies
	pip install -r requirements-dev.txt
	pip install -r requirements-test.txt
	pip install -r requirements.txt

STRESS_URL = https://api-inference-deploy-581710028917.us-central1.run.app
LOCUST_USERS ?= 25
LOCUST_SPAWN_RATE ?= 5
LOCUST_RUNTIME ?= 60s
.PHONY: stress-test
stress-test:
	# change stress url to your deployed app 
	@if ! command -v locust >/dev/null 2>&1; then \
		echo "Locust is not installed. Run 'make install' (or 'pip install -r requirements-test.txt') and retry."; \
		exit 1; \
	fi
	mkdir -p reports
	PYTHONPATH=. locust -f tests/stress/api_stress.py --print-stats --html reports/stress-test.html --run-time $(LOCUST_RUNTIME) --headless --users $(LOCUST_USERS) --spawn-rate $(LOCUST_SPAWN_RATE) -H $(STRESS_URL)

.PHONY: model-test
model-test:			## Run tests and coverage
	mkdir -p reports
	PYTHONWARNINGS="ignore::PendingDeprecationWarning" NPY_DISABLE_MACOS_ACCELERATE=1 pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge.model tests/model

.PHONY: api-test
api-test:			## Run tests and coverage
	mkdir -p reports
	NPY_DISABLE_MACOS_ACCELERATE=1 CHALLENGE_API_DISABLE_GCP=1 CHALLENGE_API_ENABLE_BQ=0 CHALLENGE_API_FAKE_MODEL=1 MODEL_LOCAL_PATH=challenge/xgb_model.pkl pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge tests/api

.PHONY: test
test: model-test api-test	## Run model and API test suites

.PHONY: build
build:			## Build locally the python artifact
	python setup.py bdist_wheel
