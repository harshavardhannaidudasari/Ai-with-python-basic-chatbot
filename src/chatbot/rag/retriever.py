"""Ties together document loading, chunking, embedding, and vector search."""

from __future__ import annotations

from pathlib import Path

from ..config import settings
from .chunker import chunk_documents
from .document_loader import load_documents
from .embeddings import Embedder
from .vector_store import SearchResult, VectorStore


class Retriever:
    def __init__(
        self,
        docs_dir: Path | None = None,
        index_dir: Path | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.docs_dir = docs_dir or settings.docs_dir
        self.index_dir = index_dir or settings.index_dir
        self._embedder: Embedder | None = None
        self._embedding_model_name = embedding_model or settings.embedding_model
        self.store = VectorStore(self.index_dir)

    @property
    def embedder(self) -> Embedder:
        # Lazily loaded: importing sentence-transformers is slow, and a plain
        # chat session (RAG disabled) shouldn't pay that cost.
        if self._embedder is None:
            self._embedder = Embedder(self._embedding_model_name)
        return self._embedder

    def ingest(self) -> int:
        """(Re)build the vector index from every document in `docs_dir`.

        Returns the number of chunks indexed.
        """
        documents = load_documents(self.docs_dir)
        if not documents:
            return 0

        chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        sources = [c.source for c in chunks]
        vectors = self.embedder.embed(texts)
        self.store.build(vectors, texts, sources)
        return len(chunks)

    def is_ready(self) -> bool:
        return not self.store.is_empty()

    def chunks_for_source(self, source: str) -> int:
        """Count indexed chunks that came from a given file name (post-`ingest()`)."""
        return sum(1 for m in self.store.metadata if m["source"] == source)

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        if self.store.is_empty():
            return []
        query_vector = self.embedder.embed_one(query)
        return self.store.search(
            query_vector,
            top_k=top_k or settings.top_k,
            min_similarity=settings.min_similarity,
        )

    def retrieve_context(self, query: str, top_k: int | None = None) -> list[str]:
        results = self.retrieve(query, top_k=top_k)
        return [f"[Source: {r.source}]\n{r.text}" for r in results]
