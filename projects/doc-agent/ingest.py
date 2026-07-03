import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCS_DIR = os.getenv("DOCS_DIR", "docs")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 100))
MANIFEST_PATH = Path(CHROMA_PERSIST_DIR) / "manifest.json"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_documents() -> list[Document]:
    print(f"Loading documents from {DOCS_DIR}...")
    docs = []
    for path in Path(DOCS_DIR).rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(
            page_content=text,
            metadata={"source": str(path)}
        ))
    print(f"Loaded {len(docs)} documents")
    return docs


def load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        return dict(json.loads(MANIFEST_PATH.read_text()))
    return {"docs": {}}


def save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def partition_documents(
    docs: list[Document], manifest: dict[str, Any]
) -> tuple[list[Document], list[str]]:
    """Split loaded docs into ones needing (re)processing vs sources removed from disk.

    Stamps each doc's content hash into its metadata as a side effect, since
    both level-1 comparison here and the later manifest update need it.
    """
    current_sources = {doc.metadata["source"] for doc in docs}
    stale_sources = [s for s in manifest["docs"] if s not in current_sources]

    to_process = []
    for doc in docs:
        source = doc.metadata["source"]
        doc_hash = _hash(doc.page_content)
        doc.metadata["doc_hash"] = doc_hash
        cached = manifest["docs"].get(source)
        if cached is None or cached["doc_hash"] != doc_hash:
            to_process.append(doc)
    return to_process, stale_sources


def chunk_documents(docs: list[Document]) -> list[Document]:
    print(f"Chunking documents with chunk size {CHUNK_SIZE} and overlap {CHUNK_OVERLAP}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["chunk_id"] = _hash(chunk.page_content)
    print(f"Created {len(chunks)} chunks.")
    return chunks


def sync_store(
    chunks: list[Document],
    docs_to_process: list[Document],
    stale_sources: list[str],
    manifest: dict[str, Any],
) -> tuple[int, int, int]:
    """Reconcile the persisted Chroma store with the current chunk set.

    Only chunks whose content hash isn't already in the store get embedded
    (level-2 reuse); chunks left over from a prior version of a changed or
    removed document are deleted so the store doesn't accumulate stale content.
    """
    print(f"Syncing chunks with Chroma at {CHROMA_PERSIST_DIR}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cuda" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu"}
    )
    vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)

    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    existing_ids = set(vector_store.get(ids=chunk_ids, include=[])["ids"]) if chunk_ids else set()
    new_chunks = [c for c, cid in zip(chunks, chunk_ids, strict=True) if cid not in existing_ids]
    new_ids = [cid for cid in chunk_ids if cid not in existing_ids]

    current_ids_by_source: dict[str, list[str]] = {}
    for chunk in chunks:
        current_ids_by_source.setdefault(chunk.metadata["source"], []).append(chunk.metadata["chunk_id"])

    to_delete: list[str] = []
    for doc in docs_to_process:
        source = doc.metadata["source"]
        old_ids = manifest["docs"].get(source, {}).get("chunk_ids", [])
        current_ids = current_ids_by_source.get(source, [])
        to_delete.extend(cid for cid in old_ids if cid not in current_ids)
    for source in stale_sources:
        to_delete.extend(manifest["docs"][source]["chunk_ids"])

    if to_delete:
        vector_store.delete(ids=to_delete)
    for i in range(0, len(new_chunks), EMBED_BATCH_SIZE):
        batch = new_chunks[i : i + EMBED_BATCH_SIZE]
        batch_ids = new_ids[i : i + EMBED_BATCH_SIZE]
        vector_store.add_documents(documents=batch, ids=batch_ids)

    for source in stale_sources:
        del manifest["docs"][source]
    for doc in docs_to_process:
        source = doc.metadata["source"]
        manifest["docs"][source] = {
            "doc_hash": doc.metadata["doc_hash"],
            "chunk_ids": current_ids_by_source.get(source, []),
        }

    return len(new_chunks), len(chunks) - len(new_chunks), len(to_delete)


def main() -> None:
    docs = load_documents()
    if not docs:
        print("No documents found to process. Add .md or .pdf files and retry.")
        return

    manifest = load_manifest()
    to_process, stale_sources = partition_documents(docs, manifest)
    unchanged = len(docs) - len(to_process)

    if not to_process and not stale_sources:
        print(f"{unchanged} documents unchanged. Index already up to date.")
        return

    chunks = chunk_documents(to_process) if to_process else []
    embedded, reused, deleted = sync_store(chunks, to_process, stale_sources, manifest)
    save_manifest(manifest)

    print(
        f"Ingestion complete: {unchanged} docs unchanged, {embedded} chunks embedded, "
        f"{reused} chunks reused, {deleted} chunks removed."
    )


if __name__ == "__main__":
    main()
