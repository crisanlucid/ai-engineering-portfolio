import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DOCS_DIR = os.getenv("DOCS_DIR", "docs")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))


def load_documents():
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

def chunk_documents(docs):
    print(f"Chunking documents with chunk size {CHUNK_SIZE} and overlap {CHUNK_OVERLAP}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def embed_and_store(chunks):
    print(f"Embedding chunks and storing in Chroma at {CHROMA_PERSIST_DIR}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cuda" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu"}
    )
    print(f"Storing vectors in {CHROMA_PERSIST_DIR}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    print(f"Done, {len(chunks)} chunks stored.")
    return vector_store

def main():
   docs = load_documents()
   if not docs:
       print("No documents found to process. Add .md or .pdf files and retry.")
       return
   chunks = chunk_documents(docs)
   embed_and_store(chunks)
   print("Ingestion complete. ready to chat")

if __name__ == "__main__":
    main()   