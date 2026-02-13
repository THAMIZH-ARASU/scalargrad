# Makefile for ScalarGrad package management

.PHONY: help clean build check test-upload upload install test all

help:
	@echo "ScalarGrad Package Management"
	@echo "=============================="
	@echo "make clean        - Remove build artifacts"
	@echo "make build        - Build distribution packages"
	@echo "make check        - Validate distributions"
	@echo "make test-upload  - Upload to TestPyPI"
	@echo "make upload       - Upload to PyPI (production)"
	@echo "make install      - Install package locally in editable mode"
	@echo "make test         - Run test suite"
	@echo "make all          - Clean, build, check, and upload to TestPyPI"
	@echo "make release      - Clean, build, check, and upload to PyPI"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✓ Clean complete"

build: clean
	@echo "Building distribution packages..."
	python -m build
	@echo "✓ Build complete"

check: build
	@echo "Validating distributions..."
	twine check dist/*
	@echo "✓ Check complete"

test-upload: check
	@echo "Uploading to TestPyPI..."
	twine upload --repository testpypi dist/*
	@echo "✓ Uploaded to TestPyPI"
	@echo "Install with: pip install --index-url https://test.pypi.org/simple/ scalargrad"

upload: check
	@echo "WARNING: This will upload to production PyPI!"
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ]
	@echo "Uploading to PyPI..."
	twine upload dist/*
	@echo "✓ Uploaded to PyPI"
	@echo "Install with: pip install scalargrad"

install:
	@echo "Installing package in editable mode..."
	pip install -e .
	@echo "✓ Package installed"

test:
	@echo "Running test suite..."
	pytest -v
	@echo "✓ Tests complete"

all: clean build check test-upload
	@echo "✓ All steps complete (uploaded to TestPyPI)"

release: clean build check upload
	@echo "✓ Release complete (uploaded to PyPI)"
