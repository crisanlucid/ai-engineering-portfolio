"""
RAG webhook — exposes the doc-agent pipeline as an HTTP API.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Intranet endpoints:
    POST  /ask       — ask a question, get answer + sources + optional judge verdict
    GET   /health    — liveness check
    GET   /docs      — interactive Swagger UI (auto-generated)
"""
import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chat import retrieve, format_context, SYSTEM_PROMPTS, LANGUAGE
from judge import judge as llm_judge, Verdict
from llm import get_adapter, LLMAdapter

load_dotenv()

AGENT_JUDGE = os.getenv("AGENT_JUDGE", "false").lower() == "true"
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "0.0.0.0")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8000"))

# ---------------------------------------------------------------------------
# App state — vectorstore and adapter loaded once at startup
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once on startup, release on shutdown."""
    from chat import load_vectorstore
    print("Loading vectorstore...")
    _state["vectorstore"] = load_vectorstore()
    _state["adapter"] = get_adapter()
    print("Webhook ready.")
    yield
    _state.clear()


app = FastAPI(
    title="Doc Agent — RAG Webhook",
    description=(
        "Ask questions against your ingested documents. "
        "Powered by ChromaDB + multilingual embeddings + your choice of LLM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer")
    k: int = Field(3, ge=1, le=10, description="Number of chunks to retrieve")
    judge: bool | None = Field(
        None,
        description="Run LLM-as-judge. Defaults to AGENT_JUDGE env var when null.",
    )


class ChunkInfo(BaseModel):
    source: str
    score: float = Field(description="Cosine similarity score (0–1)")
    preview: str = Field(description="First 200 characters of the chunk")


class VerdictInfo(BaseModel):
    verdict: str = Field(description="SUPPORTED | PARTIAL | NOT_SUPPORTED")
    reason: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(description="Unique source files cited")
    chunks: list[ChunkInfo] = Field(description="Retrieved chunks with scores")
    verdict: VerdictInfo | None = Field(
        None, description="LLM-as-judge result (null when judge is disabled)"
    )


class HealthResponse(BaseModel):
    status: str
    provider: str
    language: str
    vectorstore: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check — confirms the vectorstore and adapter are loaded."""
    if "vectorstore" not in _state:
        raise HTTPException(status_code=503, detail="Vectorstore not ready")
    return HealthResponse(
        status="ok",
        provider=os.getenv("LLM_PROVIDER", "cli"),
        language=LANGUAGE,
        vectorstore=os.getenv("CHROMA_PERSIST_DIR", ".chroma"),
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(request: AskRequest):
    """
    Ask a question against the ingested documents.

    - Retrieves the top-k most relevant chunks from ChromaDB
    - Augments a prompt with the chunks and sends it to the configured LLM
    - Optionally validates the answer with an LLM-as-judge
    """
    if "vectorstore" not in _state:
        raise HTTPException(status_code=503, detail="Vectorstore not ready")

    vectorstore: Any = _state["vectorstore"]
    adapter: LLMAdapter = _state["adapter"]

    # Retrieve
    scored_docs = retrieve(vectorstore, request.question, k=request.k)

    # Build LLM prompt
    context = format_context(scored_docs)
    user_prompt = f"Context:\n{context}\n\nQuestion: {request.question}"
    answer = adapter.complete(SYSTEM_PROMPTS[LANGUAGE], user_prompt)

    # Judge
    run_judge = request.judge if request.judge is not None else AGENT_JUDGE
    verdict_info: VerdictInfo | None = None
    if run_judge:
        result = llm_judge(scored_docs, answer, adapter)
        verdict_info = VerdictInfo(
            verdict=result.verdict.value,
            reason=result.reason,
        )

    # Build response
    chunks = [
        ChunkInfo(
            source=doc.metadata.get("source", "unknown"),
            score=score,
            preview=doc.page_content[:200].strip(),
        )
        for doc, score in scored_docs
    ]
    sources = list(dict.fromkeys(c.source for c in chunks))  # unique, ordered

    return AskResponse(
        answer=answer,
        sources=sources,
        chunks=chunks,
        verdict=verdict_info,
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=WEBHOOK_HOST,
        port=WEBHOOK_PORT,
        reload=True,
    )
