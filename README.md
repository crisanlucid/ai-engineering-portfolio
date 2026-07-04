# AI Engineering Portfolio

From fullstack to AI Engineering.  
Autonomous agents, RAG pipelines, and agentic tools built in public.

## Projects

| # | Project | Description | Status |
|---|---|---|---|
| 01 | [rag-pipeline](projects/doc-agent) | RAG + Cache | ✅ |

## Structure

```
ai-engineering-portfolio/
├── pyproject.toml          # uv workspace root + ruff/mypy/pytest config
├── .python-version         # 3.12
├── Makefile                # common commands
├── uv.lock
├── packages/
│   └── shared/             # internal library shared across projects
│       └── src/shared/
├── projects/               # individual AI/agent projects
└── apps/                   # frontend applications
```

## Tooling

This project uses **[uv](https://docs.astral.sh/uv/)** — a fast Python package and project manager written in Rust. It replaces `pip`, `venv`, `pipx`, and `pip-tools` in a single tool. All dependencies are installed into a local `.venv` inside the repo; nothing is installed globally.

`uv.lock` is committed to the repo — cloning and running `make install` gives the exact same dependency versions on any machine.

## Setup

```bash
make install
```

## Commands

```bash
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

1. Create the directory layout:

```bash
mkdir -p projects/<name>/src/<name> projects/<name>/tests
```

2. Create `projects/<name>/pyproject.toml`:

```toml
[project]
name = "<name>"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "shared",
]

[tool.uv.sources]
shared = { workspace = true }
```

3. Add an entry point:

```bash
touch projects/<name>/src/<name>/__init__.py
touch projects/<name>/tests/__init__.py
```

4. Sync the workspace:

```bash
make install
```
