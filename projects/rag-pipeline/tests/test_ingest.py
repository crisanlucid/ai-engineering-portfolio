"""Unit tests for ingest.py — two-level (doc hash / chunk hash) ingest cache."""

from unittest.mock import MagicMock, patch

from ingest import (
    EMBED_BATCH_SIZE,
    _hash,
    chunk_documents,
    load_documents,
    partition_documents,
    sync_store,
)
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(source: str, text: str) -> Document:
    return Document(page_content=text, metadata={"source": source})


class TestHash:
    def test_deterministic(self):
        assert _hash("hello") == _hash("hello")

    def test_differs_on_content(self):
        assert _hash("hello") != _hash("world")


class TestPartitionDocuments:
    def test_new_document_is_processed(self):
        docs = [_doc("a.md", "content a")]
        manifest = {"docs": {}}
        to_process, stale = partition_documents(docs, manifest)
        assert to_process == docs
        assert stale == []

    def test_unchanged_document_is_skipped(self):
        docs = [_doc("a.md", "content a")]
        manifest = {"docs": {"a.md": {"doc_hash": _hash("content a"), "chunk_ids": ["x"]}}}
        to_process, stale = partition_documents(docs, manifest)
        assert to_process == []
        assert stale == []

    def test_changed_document_is_processed(self):
        docs = [_doc("a.md", "new content")]
        manifest = {"docs": {"a.md": {"doc_hash": _hash("old content"), "chunk_ids": ["x"]}}}
        to_process, stale = partition_documents(docs, manifest)
        assert to_process == docs

    def test_removed_document_is_reported_stale(self):
        docs = [_doc("a.md", "content a")]
        manifest = {
            "docs": {
                "a.md": {"doc_hash": _hash("content a"), "chunk_ids": ["x"]},
                "b.md": {"doc_hash": "whatever", "chunk_ids": ["y", "z"]},
            }
        }
        to_process, stale = partition_documents(docs, manifest)
        assert to_process == []
        assert stale == ["b.md"]

    def test_stamps_doc_hash_metadata(self):
        docs = [_doc("a.md", "content a")]
        partition_documents(docs, {"docs": {}})
        assert docs[0].metadata["doc_hash"] == _hash("content a")


class TestChunkDocuments:
    def test_stamps_chunk_id(self):
        docs = [_doc("a.md", "some short text")]
        chunks = chunk_documents(docs)
        assert all("chunk_id" in c.metadata for c in chunks)
        assert chunks[0].metadata["chunk_id"] == _hash(chunks[0].page_content)


class TestSyncStore:
    def _chunk(self, source: str, text: str, chunk_id: str | None = None) -> Document:
        chunk = Document(page_content=text, metadata={"source": source})
        chunk.metadata["chunk_id"] = chunk_id or _hash(text)
        return chunk

    def test_only_embeds_chunks_not_already_in_store(self):
        chunk_a = self._chunk("a.md", "unchanged text")
        chunk_b = self._chunk("a.md", "new text")
        doc = _doc("a.md", "unchanged text new text")
        doc.metadata["doc_hash"] = _hash("unchanged text new text")
        manifest = {"docs": {}}

        with patch("ingest.HuggingFaceEmbeddings"), patch("ingest.Chroma") as mock_chroma_cls:
            mock_store = MagicMock()
            mock_store.get.return_value = {"ids": [chunk_a.metadata["chunk_id"]]}
            mock_chroma_cls.return_value = mock_store

            embedded, reused, deleted = sync_store([chunk_a, chunk_b], [doc], [], manifest)

            assert embedded == 1
            assert reused == 1
            assert deleted == 0
            added_docs = mock_store.add_documents.call_args.kwargs["documents"]
            assert added_docs == [chunk_b]
            assert mock_store.add_documents.call_args.kwargs["ids"] == [
                chunk_b.metadata["chunk_id"]
            ]

    def test_deletes_stale_chunks_from_changed_document(self):
        old_chunk_id = "old-chunk-hash"
        new_chunk = self._chunk("a.md", "replacement text")
        doc = _doc("a.md", "replacement text")
        doc.metadata["doc_hash"] = _hash("replacement text")
        manifest = {"docs": {"a.md": {"doc_hash": "prior-hash", "chunk_ids": [old_chunk_id]}}}

        with patch("ingest.HuggingFaceEmbeddings"), patch("ingest.Chroma") as mock_chroma_cls:
            mock_store = MagicMock()
            mock_store.get.return_value = {"ids": []}
            mock_chroma_cls.return_value = mock_store

            sync_store([new_chunk], [doc], [], manifest)

            mock_store.delete.assert_called_once_with(ids=[old_chunk_id])
            assert manifest["docs"]["a.md"]["chunk_ids"] == [new_chunk.metadata["chunk_id"]]

    def test_deletes_chunks_for_removed_document(self):
        manifest = {"docs": {"removed.md": {"doc_hash": "h", "chunk_ids": ["r1", "r2"]}}}

        with patch("ingest.HuggingFaceEmbeddings"), patch("ingest.Chroma") as mock_chroma_cls:
            mock_store = MagicMock()
            mock_store.get.return_value = {"ids": []}
            mock_chroma_cls.return_value = mock_store

            embedded, reused, deleted = sync_store([], [], ["removed.md"], manifest)

            mock_store.delete.assert_called_once_with(ids=["r1", "r2"])
            assert deleted == 2
            assert "removed.md" not in manifest["docs"]

    def test_no_new_chunks_skips_add(self):
        manifest = {"docs": {}}
        with patch("ingest.HuggingFaceEmbeddings"), patch("ingest.Chroma") as mock_chroma_cls:
            mock_store = MagicMock()
            mock_store.get.return_value = {"ids": []}
            mock_chroma_cls.return_value = mock_store

            sync_store([], [], [], manifest)

            mock_store.add_documents.assert_not_called()

    def test_batches_add_documents_when_exceeding_batch_size(self):
        n = EMBED_BATCH_SIZE + 5
        chunks = [self._chunk("a.md", f"text {i}") for i in range(n)]
        doc = _doc("a.md", " ".join(f"text {i}" for i in range(n)))
        doc.metadata["doc_hash"] = _hash(doc.page_content)
        manifest = {"docs": {}}

        with patch("ingest.HuggingFaceEmbeddings"), patch("ingest.Chroma") as mock_chroma_cls:
            mock_store = MagicMock()
            mock_store.get.return_value = {"ids": []}
            mock_chroma_cls.return_value = mock_store

            embedded, reused, deleted = sync_store(chunks, [doc], [], manifest)

            assert embedded == n
            assert mock_store.add_documents.call_count == 2
            batch_sizes = [
                len(c.kwargs["documents"]) for c in mock_store.add_documents.call_args_list
            ]
            assert batch_sizes == [EMBED_BATCH_SIZE, 5]
            all_documents = [
                d for c in mock_store.add_documents.call_args_list for d in c.kwargs["documents"]
            ]
            assert all_documents == chunks


class TestLoadDocuments:
    @staticmethod
    def _page(text):
        page = MagicMock()
        page.extract_text.return_value = text
        return page

    def test_loads_markdown_and_pdf(self, tmp_path):
        (tmp_path / "a.md").write_text("hello md", encoding="utf-8")
        (tmp_path / "b.pdf").write_bytes(b"%PDF-fake")

        mock_pdf = MagicMock()
        mock_pdf.pages = [self._page("page one"), self._page("page two")]

        with (
            patch("ingest.DOCS_DIR", str(tmp_path)),
            patch("ingest.pdfplumber.open") as mock_open,
        ):
            mock_open.return_value.__enter__.return_value = mock_pdf
            docs = load_documents()

        by_type = {d.metadata["type"]: d for d in docs}
        assert len(docs) == 2
        assert by_type["md"].page_content == "hello md"
        assert by_type["md"].metadata["source"] == str(tmp_path / "a.md")
        assert by_type["pdf"].page_content == "page one\n\npage two"
        assert by_type["pdf"].metadata["pages"] == 2

    def test_skips_blank_page_but_keeps_rest(self, tmp_path):
        (tmp_path / "b.pdf").write_bytes(b"%PDF-fake")
        mock_pdf = MagicMock()
        mock_pdf.pages = [self._page("real text"), self._page(None)]

        with (
            patch("ingest.DOCS_DIR", str(tmp_path)),
            patch("ingest.pdfplumber.open") as mock_open,
        ):
            mock_open.return_value.__enter__.return_value = mock_pdf
            docs = load_documents()

        assert len(docs) == 1
        assert docs[0].page_content == "real text"
        assert docs[0].metadata["pages"] == 2

    def test_skips_fully_scanned_pdf(self, tmp_path, capsys):
        (tmp_path / "scanned.pdf").write_bytes(b"%PDF-fake")
        mock_pdf = MagicMock()
        mock_pdf.pages = [self._page(None), self._page("")]

        with (
            patch("ingest.DOCS_DIR", str(tmp_path)),
            patch("ingest.pdfplumber.open") as mock_open,
        ):
            mock_open.return_value.__enter__.return_value = mock_pdf
            docs = load_documents()

        assert docs == []
        assert "Skipping" in capsys.readouterr().out
