import os
import sys
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from llm import LLMAdapter, get_adapter
from judge import JudgeResult, judge as llm_judge

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
LANGUAGE = os.getenv("LANGUAGE", "en")
AGENT_DEBUG = os.getenv("AGENT_DEBUG", "false").lower() == "true"
AGENT_JUDGE = os.getenv("AGENT_JUDGE", "false").lower() == "true"

SYSTEM_PROMPTS = {
    "en": """You are a helpful assistant.
Answer only from the provided context.
If the answer is not in the context, say: "I don't know based on the provided documents."
Always cite the source doc at the end of your answer.""",

    "de": """Du bist ein hilfreicher Assistent.
Antworte nur auf Basis des bereitgestellten Kontexts.
Wenn die Antwort nicht im Kontext enthalten ist, sage: "Das kann ich anhand der bereitgestellten Dokumente nicht beantworten."
Zitiere am Ende deiner Antwort immer das Quelldokument."""
}

CLI_PROMPTS = {
    "en": "Ask a question (or 'exit' to quit): ",
    "de": "Stelle eine Frage (oder 'exit' zum Beenden): "
}

def load_vectorstore():
    print("Loading embedding model... (this takes 20-30s on first run)")
    print("Please wait", end="", flush=True)

    import threading
    import time

    done = False

    def spinner():
        while not done:
            print(".", end="", flush=True)
            time.sleep(1)

    t = threading.Thread(target=spinner)
    t.start()

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"}
    )

    done = True
    t.join()
    print(" done!")

    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    print("Vector store loaded. Ready.\n")
    return vectorstore

def retrieve(vectorstore, question: str, k: int = 3) -> list[tuple[Document, float]]:
    """Return (doc, cosine_similarity) pairs.

    Chroma's similarity_search_with_score returns squared L2 distance.
    For normalised embeddings: cosine_similarity = 1 - (l2_distance^2 / 2).
    """
    results = vectorstore.similarity_search_with_score(question, k=k)
    scored = []
    for doc, l2_dist in results:
        cosine_sim = max(0.0, 1.0 - (l2_dist / 2.0))
        scored.append((doc, round(cosine_sim, 4)))
    return scored


def _print_debug_scores(scored_docs: list[tuple[Document, float]]) -> None:
    """Print a cosine-score table to stderr when AGENT_DEBUG=true."""
    print("\n\033[90m┌─ [DEBUG] Retrieval scores " + "─" * 32 + "┐\033[0m", file=sys.stderr)
    for i, (doc, score) in enumerate(scored_docs):
        src = doc.metadata.get("source", "unknown")
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"\033[90m│  Chunk {i+1}  [{bar}]  {score:.4f}  {src}\033[0m", file=sys.stderr)
    print("\033[90m└" + "─" * 60 + "┘\033[0m\n", file=sys.stderr)


def format_context(scored_docs: list[tuple[Document, float]]) -> str:
    ctx = ""
    for i, (doc, score) in enumerate(scored_docs):
        src = doc.metadata.get("source", "unknown")
        score_tag = f" | cosine: {score:.4f}" if AGENT_DEBUG else ""
        ctx += f"\n--- Chunk {i+1} | source: {src}{score_tag} ---\n"
        ctx += doc.page_content
        ctx += "\n"
    return ctx


def ask(
    vectorstore,
    question: str,
    adapter: LLMAdapter,
) -> tuple[str, JudgeResult | None]:
    scored_docs = retrieve(vectorstore, question)
    if AGENT_DEBUG:
        _print_debug_scores(scored_docs)
    context = format_context(scored_docs)
    user = f"Context:\n{context}\n\nQuestion: {question}"
    print("Thinking...", flush=True)
    answer = adapter.complete(SYSTEM_PROMPTS[LANGUAGE], user)

    result: JudgeResult | None = None
    if AGENT_JUDGE:
        print("Judging...", flush=True)
        result = llm_judge(scored_docs, answer, adapter)

    return answer, result

def main():
    provider = os.getenv("LLM_PROVIDER", "cli")
    print(f"Doc Agent starting — language: {LANGUAGE}, provider: {provider}")

    adapter = get_adapter()
    vectorstore = load_vectorstore()

    prompt = CLI_PROMPTS[LANGUAGE]
    print("Ready. Type your question.\n")

    while True:
        try:
            question = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye.")
            break

        answer, verdict = ask(vectorstore, question, adapter)
        print(f"\nAnswer:\n{answer}\n")
        if verdict is not None:
            print(f"Judge:  {verdict.badge()}\n")

if __name__ == "__main__":
    main()
