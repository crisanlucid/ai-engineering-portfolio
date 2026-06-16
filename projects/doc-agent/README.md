# doc-agent — RAG Fundamentals

A minimal, fully local **Retrieval-Augmented Generation (RAG)** pipeline built
as a hands-on reference for understanding how RAG works end-to-end.  
No cloud vector DB, no managed API required — runs entirely on your machine.

---

## What This Teaches

| Concept | Where |
|---|---|
| Document loading & chunking | `ingest.py` |
| Embedding with a local HuggingFace model | `ingest.py` |
| Vector storage & similarity search (ChromaDB) | `ingest.py` / `chat.py` |
| L2 distance → cosine similarity conversion | `chat.py` — `retrieve()` |
| Prompt augmentation (context injection) | `chat.py` — `format_context()` |
| LLM adapter pattern (swap providers easily) | `llm.py` |
| LLM-as-judge answer validation | `judge.py` |
| Multilingual support (en / de) | `chat.py` |

---

## Architecture

```
Documents (docs/*.md)
        │
        ▼
  ingest.py
  ├── load_documents()       — reads .md files from docs/
  ├── chunk_documents()      — RecursiveCharacterTextSplitter
  └── embed_and_store()      — intfloat/multilingual-e5-large → ChromaDB
        │
        ▼
  .chroma/  (persisted vector store)
════════════════════════════════════════════════
  User question
        │
        ▼
  chat.py
  ├── retrieve()             — similarity_search_with_score → cosine similarity
  ├── format_context()       — assembles prompt with source citations
  ├── ask()                  — calls LLM adapter, optionally runs judge
  └── main()                 — interactive CLI loop
        │
        ├── llm.py           — LLMAdapter (CLI / Anthropic / Google / agy)
        └── judge.py         — LLM-as-judge → SUPPORTED / PARTIAL / NOT_SUPPORTED
```

---

## Quickstart

### 1. Install dependencies

```bash
make install        # or: uv sync
```

### 2. Add your documents

Drop `.md` files into the `docs/` folder.

### 3. Ingest

```bash
python ingest.py
```

This chunks your docs, embeds them with
[`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large),
and stores them in ChromaDB.

### 4. Chat

```bash
python chat.py
```

---

## Configuration

All options live in `.env` (copy from `.env example`):

```ini
# LLM Provider — cli | anthropic | google | agy
LLM_PROVIDER=cli

# API key — required for anthropic / google; unused for cli and agy
MODEL_API_KEY=

# ChromaDB
CHROMA_PERSIST_DIR=.chroma

# Ingestion
DOCS_DIR=docs
CHUNK_SIZE=1000
CHUNK_OVERLAP=100

# Language — en | de
LANGUAGE=en

# Debug: show cosine retrieval scores per chunk
AGENT_DEBUG=false

# Judge: validate every answer with a second LLM call
AGENT_JUDGE=false
```

---

## LLM Providers

| `LLM_PROVIDER` | Description |
|---|---|
| `cli` | Uses your local **Claude Code** CLI session — no API key needed |
| `agy` | Uses your local **Antigravity (agy)** CLI session — no API key needed |
| `anthropic` | Calls the Anthropic API (`MODEL_API_KEY` required) |
| `google` | Calls the Google Gemini API (`MODEL_API_KEY` required) |

---

## Debug Mode — `AGENT_DEBUG=true`

Shows a cosine similarity score table for every retrieval, printed to `stderr`:

```
┌─ [DEBUG] Retrieval scores ────────────────────────────────┐
│  Chunk 1  [███████████████░░░░░]  0.7986  docs/README.md
│  Chunk 2  [███████████████░░░░░]  0.7955  docs/README.md
│  Chunk 3  [███████████████░░░░░]  0.7925  docs/README.md
└────────────────────────────────────────────────────────────┘
```

Each chunk header in the LLM prompt also includes its score:
```
--- Chunk 1 | source: docs/README.md | cosine: 0.7986 ---
```

> **How scores work:** ChromaDB returns squared L2 distance.
> For normalised embeddings: `cosine_similarity = 1 − (l2_distance / 2)`.
> Score of `1.0` = perfect match, `0.0` = completely unrelated.

---

## LLM-as-Judge — `AGENT_JUDGE=true`

After every answer, a second LLM call evaluates whether the answer is
supported by the retrieved context chunks.

```
Answer:
tesseract.js is an OCR library, supports 100+ languages via WASM SIMD.

Judge:  ✔ SUPPORTED  —  The answer is fully backed by the context.
```

| Verdict | Meaning |
|---|---|
| `✔ SUPPORTED` | Answer is fully grounded in the retrieved chunks |
| `~ PARTIAL` | Some claims are supported; others are missing or unverified |
| `✘ NOT SUPPORTED` | Answer makes claims not found in the chunks |

> A correct *"I don't know"* answer when the topic is absent from the docs
> is also rated **SUPPORTED** — the judge understands absence of evidence.

---

## Project Structure

```
doc-agent/
├── ingest.py          # Ingestion pipeline
├── chat.py            # RAG chat loop
├── llm.py             # LLM adapter layer (CLI / Anthropic / Google / agy)
├── judge.py           # LLM-as-judge module
├── main.py            # Entry point alias
├── docs/              # Your source documents (drop .md files here)
├── tests/
│   ├── test_chat.py   # 27 tests — retrieve, format, ask, judge wiring, main()
│   ├── test_llm.py    # 15 tests — adapter routing, subprocess behaviour
│   └── test_judge.py  # 22 tests — parsing, badge, judge() integration
└── .env example       # Configuration template
```

---

## Running Tests

```bash
uv run pytest projects/doc-agent/tests/ -v
```

64 tests, no network calls, no LLM invocations — all external dependencies are mocked.
