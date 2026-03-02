.PHONY: help install install-dev test test-unit test-integration test-e2e lint format type-check coverage clean build dist ci

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

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,all]"

test:
	pytest

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

test-e2e:
	pytest -m e2e

lint:
	ruff check tinyrag tests

format:
	black tinyrag tests
	ruff check --fix tinyrag tests

type-check:
	mypy tinyrag

coverage:
	pytest --cov=tinyrag --cov-report=html --cov-report=term-missing
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
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

dist: build
	@echo "Distribution created in dist/"

ci: lint type-check test
	@echo "CI pipeline completed successfully"
