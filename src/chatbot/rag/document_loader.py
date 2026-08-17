"""Loads plain-text sources for the knowledge base: .txt, .md, .pdf."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    source: str
    text: str


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pypdf is required to load PDF files. Install it with `pip install pypdf`."
        ) from exc

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _load_pdf(path)
    elif suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return Document(source=path.name, text=text)


def load_documents(directory: Path) -> list[Document]:
    """Load every supported file directly inside `directory`."""
    if not directory.exists():
        return []

    documents = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                documents.append(load_document(path))
            except Exception as exc:  # pragma: no cover - surfaced to caller
                print(f"[warn] skipping {path.name}: {exc}")
    return documents
