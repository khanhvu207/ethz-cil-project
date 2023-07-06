#!/bin/bash
black --check src/ && \
isort --check-only src/ && \
flake8 src/ && \
mypy src/ && \
darglint src/