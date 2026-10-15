.PHONY: help install install-dev test test-cov lint format typecheck clean build publish docs

PYTHON := python
PIP := pip
PYTEST := pytest
PKG := pygeofetch

help:  ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make \033[36m<target>\033[0m\n\nTargets:\n"} \
	/^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## Install the package
	$(PIP) install -e .

install-dev:  ## Install with all dev dependencies
	$(PIP) install -e ".[dev,all]"

test:  ## Run the test suite
	$(PYTEST) tests/ -v --tb=short

test-cov:  ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=$(PKG) --cov-report=term-missing --cov-report=html

test-fast:  ## Run only unit tests (skips integration)
	$(PYTEST) tests/unit/ -v --tb=short

lint:  ## Lint code with ruff
	ruff check $(PKG)/ tests/

format:  ## Format code with black
	black $(PKG)/ tests/

format-check:  ## Check formatting without modifying files
	black --check $(PKG)/ tests/

typecheck:  ## Run mypy type checker
	mypy $(PKG)/

check: lint format-check typecheck  ## Run all checks (no tests)

clean:  ## Remove build artifacts and cache
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

build: clean  ## Build wheel and source distribution
	$(PYTHON) -m build

publish-test: build  ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish: build  ## Publish to PyPI (requires credentials)
	twine upload dist/*

docs:  ## Build HTML documentation
	cd docs && make html

docs-serve:  ## Serve docs locally
	cd docs/_build/html && python -m http.server 8080

version-patch:  ## Bump patch version
	bump2version patch

version-minor:  ## Bump minor version
	bump2version minor

version-major:  ## Bump major version
	bump2version major

setup-hooks:  ## Install pre-commit hooks
	pre-commit install

run-hooks:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

docker-build:  ## Build Docker image
	docker build -t pygeofetch:latest .

docker-run:  ## Run interactive Docker container
	docker run --rm -it \
	  -v "$$HOME/.pygeofetch:/root/.pygeofetch" \
	  -v "$$(pwd)/data:/data" \
	  pygeofetch:latest bash
