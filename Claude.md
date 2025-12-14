# Claude Code Context

## Development Environment

- **Package Manager**: uv
- **Development Environment**: GitHub Codespaces
- **Secrets Management**: API keys and secrets are managed through GitHub Codespaces secrets

## Python Version Testing

This project tests across multiple Python versions:
- Python 3.10, 3.11, 3.12, 3.13
- Python 3.14 (upcoming release)
- Python 3.14t (free-threaded/no-GIL build)

## Pre-commit Hooks

The project uses pre-commit with the following hooks:
- **black** - Code formatting
- **isort** - Import sorting (black profile)
- **ruff** - Fast Python linter
- **pylint** - Python linter
- **mypy** - Static type checking
- **zizmor** - GitHub Actions security scanner
- **hadolint** - Dockerfile linter

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # Add uv to PATH

# Install pre-commit
uv pip install pre-commit --system

# Install git hooks
pre-commit install
```

## Project Purpose

Python benchmarking suite for comparing performance across different Python versions and configurations.
