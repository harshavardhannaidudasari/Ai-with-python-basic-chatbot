"""Splits document text into overlapping chunks suitable for embedding."""

from __future__ import annotations

from dataclasses import dataclass

from .document_loader import Document


@dataclass
class Chunk:
    source: str
    chunk_index: int
    text: str


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Character-based sliding-window chunking with overlap.

    Simple and dependency-free; good enough for a demo knowledge base. Splits
    on paragraph/sentence boundaries where possible to avoid mid-word cuts.
    """
    text = text.strip()
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    chunks: list[str] = []
    start = 0
    length = len(text)
    min_piece = max(chunk_size // 2, 1)

    while start < length:
        raw_end = min(start + chunk_size, length)
        end = raw_end

        # Prefer to break on a paragraph or sentence boundary near `end`, but
        # only if it doesn't shrink the piece below half the target size —
        # otherwise a boundary close to `start` (e.g. right after a heading)
        # would produce tiny chunks and, combined with overlap, stall the
        # sliding window's forward progress.
        if end < length:
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1:
                boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary - start >= min_piece:
                end = boundary + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if raw_end >= length:
            break
        # Advance based on the *unadjusted* window size so the boundary snap
        # above can never shrink the forward step.
        start = max(raw_end - chunk_overlap, start + 1)

    return chunks


def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        for i, piece in enumerate(chunk_text(doc.text, chunk_size, chunk_overlap)):
            chunks.append(Chunk(source=doc.source, chunk_index=i, text=piece))
    return chunks
