import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from llm import LLMAdapter, get_adapter

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
LANGUAGE = os.getenv("LANGUAGE", "en")

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

def retrieve(vectorstore, question: str, k: int = 3):
    return vectorstore.similarity_search(question, k=k)

def format_context(docs: list[Document]) -> str:
    ctx = ""
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "unknown")
        ctx += f"\n--- Chunk {i+1} | source: {src} ---\n"
        ctx += doc.page_content
        ctx += "\n"
    return ctx

def ask(vectorstore, question: str, adapter: LLMAdapter) -> str:
    docs = retrieve(vectorstore, question)
    context = format_context(docs)
    user = f"Context:\n{context}\n\nQuestion: {question}"
    print("Thinking...", flush=True)
    return adapter.complete(SYSTEM_PROMPTS[LANGUAGE], user)

def main():
    provider = os.getenv("LLM_PROVIDER", "cli")
    print(f"Doc Agent starting — language: {LANGUAGE}, provider: {provider}")

    adapter = get_adapter()
    vectorstore = load_vectorstore()

    prompt = CLI_PROMPTS[LANGUAGE]
    print("Ready. Type your question.\n")

    while True:
        question = input(prompt).strip()
        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye.")
            break

        answer = ask(vectorstore, question, adapter)
        print(f"\nAnswer:\n{answer}\n")

if __name__ == "__main__":
    main()
