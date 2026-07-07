# rag-pipeline — RAG Fundamentals

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
  ├── load_manifest()        — loads .chroma/manifest.json (doc cache)
  ├── partition_documents()  — level-1 cache: skip unchanged docs by hash
  ├── chunk_documents()      — RecursiveCharacterTextSplitter
  ├── sync_store()           — level-2 cache: embed new (batched) / reuse / delete stale chunks by hash
  └── save_manifest()        — persists updated manifest
        │
        ▼
  .chroma/  (persisted vector store + manifest.json)
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

## Cache Limitation

Chunk reuse depends on where an edit lands in the document.
`RecursiveCharacterTextSplitter` splits by character offset — an edit
near the top shifts all downstream chunk boundaries, causing
re-embedding even for largely unchanged content.

An edit near the end reuses most chunks.

Fix: `SemanticChunker` — but adds embedding cost during splitting.
Only worth it at high document volume with frequent small edits.

---

## Manifest Consistency

`manifest.json` is a denormalized index, not a source of truth — every value
it stores (`source`, `doc_hash`, `chunk_id`) already lives in each chunk's
Chroma metadata. It exists only so a no-op run can skip loading the embedding
model and querying Chroma entirely.

`save_manifest()` runs once, after `sync_store()` finishes. If the process
dies mid-`sync_store` (embedding failure, OOM), Chroma holds partial writes
but the manifest never gets updated to match — the next run then trusts a
stale index.

Alternative: derive hashes from Chroma metadata directly
(`get(where={"source": ...})`), removing the desync risk at the cost of the
fast no-op path.

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
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small),
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
EMBED_BATCH_SIZE=100
EMBEDDING_MODEL=intfloat/multilingual-e5-small

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
rag-pipeline/
├── ingest.py          # Ingestion pipeline
├── chat.py            # RAG chat loop
├── llm.py             # LLM adapter layer (CLI / Anthropic / Google / agy)
├── judge.py           # LLM-as-judge module
├── main.py            # Entry point alias
├── docs/              # Your source documents (drop .md files here)
├── tests/
│   ├── test_chat.py   # 27 tests — retrieve, format, ask, judge wiring, main()
│   ├── test_llm.py    # 15 tests — adapter routing, subprocess behaviour
│   ├── test_judge.py  # 22 tests — parsing, badge, judge() integration
│   └── test_ingest.py # 13 tests — doc/chunk hash caching, batched embedding
└── .env example       # Configuration template
```

---

## Running Tests

```bash
uv run pytest projects/rag-pipeline/tests/ -v
```

77 tests, no network calls, no LLM invocations — all external dependencies are mocked.
