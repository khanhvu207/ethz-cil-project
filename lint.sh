#!/bin/bash
black --check src/ scripts/ && \
isort --check-only src/ scripts/ && \
flake8 src/ scripts/ && \
mypy src/ scripts/ && \
darglint src/ scripts/ && \
jsonlint-php experiment_configs/*/*.json