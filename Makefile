.PHONY: help install install-dev test test-unit test-integration test-e2e lint format type-check coverage clean build dist ci

# Detect Python/pip command
PYTHON := $(shell which python3 || which python)
PIP := $(PYTHON) -m pip

help:
	@echo "Available targets:"
	@echo "  install          - Install package and dependencies"
	@echo "  install-dev     - Install package with dev dependencies"
	@echo "  test            - Run all tests"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-e2e        - Run end-to-end tests only"
	@echo "  lint            - Run linters (ruff)"
	@echo "  format          - Format code (black)"
	@echo "  type-check      - Run type checker (mypy)"
	@echo "  coverage        - Generate coverage report"
	@echo "  clean           - Clean build artifacts"
	@echo "  build           - Build package"
	@echo "  dist            - Create distribution"
	@echo "  ci              - Run full CI pipeline"
	@echo ""
	@echo "Using Python: $(PYTHON)"
	@echo "Using pip: $(PIP)"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,all]"

test:
	$(PYTHON) -m pytest

test-unit:
	$(PYTHON) -m pytest -m unit

test-integration:
	$(PYTHON) -m pytest -m integration

test-e2e:
	$(PYTHON) -m pytest -m e2e

lint:
	$(PYTHON) -m ruff check tinyrag tests

format:
	$(PYTHON) -m black tinyrag tests
	$(PYTHON) -m ruff check --fix tinyrag tests

type-check:
	$(PYTHON) -m mypy tinyrag

coverage:
	$(PYTHON) -m pytest --cov=tinyrag --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

build: clean
	$(PYTHON) -m build

dist: build
	@echo "Distribution created in dist/"

ci: lint type-check test
	@echo "CI pipeline completed successfully"
