# Makefile for CardioInsight AI

# Variables
PYTHON := python3
PIP := pip
TEST_DIR := tests
SRC_DIR := src
APP := app.py

# Default target
.PHONY: help
help:
	@echo "CardioInsight AI Makefile"
	@echo "========================"
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code with black"
	@echo "  run         - Run the Streamlit app"
	@echo "  docker      - Build and run with Docker"
	@echo "  clean       - Clean up temporary files"
	@echo "  help        - Show this help message"

# Install dependencies
.PHONY: install
install:
	$(PIP) install -r requirements.txt

# Run tests
.PHONY: test
test:
	$(PYTHON) -m pytest $(TEST_DIR) -v

# Run linting
.PHONY: lint
lint:
	$(PYTHON) -m flake8 $(SRC_DIR) $(TEST_DIR) $(APP)
	$(PYTHON) -m mypy $(SRC_DIR) $(APP)

# Format code
.PHONY: format
format:
	$(PYTHON) -m black $(SRC_DIR) $(TEST_DIR) $(APP)

# Run the application
.PHONY: run
run:
	$(PYTHON) -m streamlit run $(APP)

# Build and run with Docker
.PHONY: docker
docker:
	docker-compose up --build

# Clean up temporary files
.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

# Setup development environment
.PHONY: setup
setup:
	$(PYTHON) -m venv venv
	. venv/bin/activate && $(PIP) install -r requirements.txt
	. venv/bin/activate && $(PIP) install -e .[dev]