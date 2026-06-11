# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AI engineering portfolio — a `uv` workspace monorepo with Python-first agent/backend projects and optional frontend apps.

```
packages/          # shared internal libraries
  shared/          # common utilities imported by all projects
projects/          # individual AI/agent projects (each is a uv workspace member)
apps/              # frontend applications (React, Next.js, etc.)
```

## Commands

```bash
make install       # uv sync --all-packages (install all deps)
make lint          # ruff check
make fmt           # ruff format
make type-check    # mypy
make test          # pytest
make check         # lint + type-check + test
```

Run a single test file:
```bash
uv run pytest projects/<name>/tests/test_foo.py -v
```

## Adding a new project

1. Create `projects/<name>/pyproject.toml` with `[project]` and any dependencies.
2. Add `shared` as a dependency if needed: `"shared"` in `dependencies`, with `[tool.uv.sources] shared = {workspace = true}`.
3. Run `make install` to update the lockfile.

Each project follows `src/` layout: `projects/<name>/src/<name>/` and `projects/<name>/tests/`.

## Tooling

- **uv** — package/workspace manager (`uv sync`, `uv run`, `uv add`)
- **ruff** — linter + formatter (configured in root `pyproject.toml`)
- **mypy** — strict type checking
- **pytest** — test runner with `asyncio_mode = "auto"`
