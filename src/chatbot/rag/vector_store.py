"""A minimal, dependency-light vector store: numpy cosine similarity + JSON metadata.

Not meant to scale to millions of vectors — it's here so the RAG pipeline is
transparent and easy to read end-to-end. Swap in Chroma/FAISS/pgvector for
production scale without changing the `Retriever` interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchResult:
    text: str
    source: str
    score: float


class VectorStore:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.vectors_path = index_dir / "vectors.npy"
        self.metadata_path = index_dir / "metadata.json"
        self._vectors: np.ndarray | None = None
        self._metadata: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.vectors_path.exists() and self.metadata_path.exists():
            self._vectors = np.load(self.vectors_path)
            self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        else:
            self._vectors = np.empty((0, 0), dtype=np.float32)
            self._metadata = []

    def is_empty(self) -> bool:
        return self._vectors is None or self._vectors.size == 0

    def build(self, vectors: np.ndarray, texts: list[str], sources: list[str]) -> None:
        self._vectors = vectors
        self._metadata = [{"text": t, "source": s} for t, s in zip(texts, sources)]
        self._save()

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.vectors_path, self._vectors)
        self.metadata_path.write_text(json.dumps(self._metadata), encoding="utf-8")

    def search(self, query_vector: np.ndarray, top_k: int, min_similarity: float) -> list[SearchResult]:
        if self.is_empty():
            return []

        # Vectors are pre-normalized (see Embedder), so dot product == cosine similarity.
        scores = self._vectors @ query_vector
        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_similarity:
                continue
            meta = self._metadata[idx]
            results.append(SearchResult(text=meta["text"], source=meta["source"], score=score))
        return results

    @property
    def size(self) -> int:
        return 0 if self.is_empty() else self._vectors.shape[0]

    @property
    def metadata(self) -> list[dict]:
        return self._metadata
