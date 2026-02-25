.PHONY: help install test format format-check lint clean run all

help: ## Show this help message
	@echo "Agent project Makefile"
	@echo "Current Python: $(PYTHON)"
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

test: ## Run tests with coverage
	python -m pytest -vv \
		--cov=Agent \
		--cov-report=term-missing

format: ## Format code using Black
	black Agent tests run.py

format-check: ## Check formatting without modifying files
	black --check Agent tests run.py

lint: ## Run Ruff linter
	ruff check Agent tests run.py

run: ## Run the PersonalAgent application
	python run.py

clean: ## Remove caches and coverage files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf coverage.xml

all: install lint format-check test ## Run full CI pipeline locally