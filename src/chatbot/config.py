"""Central configuration, loaded from environment variables / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "claude"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-opus-5"))
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", 2048))
    effort: str = field(default_factory=lambda: os.getenv("CLAUDE_EFFORT", "medium"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_vision_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_VISION_MODEL", "llava-phi3")
    )
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    groq_vision_model: str = field(
        default_factory=lambda: os.getenv(
            "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
        )
    )
    # How long Ollama keeps a model resident in memory after a request, so
    # back-to-back messages don't each pay a full reload-from-disk cost.
    ollama_keep_alive: str = field(default_factory=lambda: os.getenv("OLLAMA_KEEP_ALIVE", "30m"))
    # Caps worst-case generation time on slow/CPU-only hardware. 0 = unlimited.
    ollama_num_predict: int = field(default_factory=lambda: _env_int("OLLAMA_NUM_PREDICT", 512))
    system_prompt: str = field(
        default_factory=lambda: os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful, concise AI assistant. When context from the "
            "knowledge base is provided, ground your answer in it and say so "
            "when the context doesn't contain the answer.",
        )
    )

    # RAG
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 120))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 4))
    min_similarity: float = field(default_factory=lambda: _env_float("MIN_SIMILARITY", 0.2))
    rag_enabled_default: bool = field(default_factory=lambda: _env_bool("RAG_ENABLED", True))

    # Storage
    docs_dir: Path = field(default_factory=lambda: Path(os.getenv("DOCS_DIR", str(PROJECT_ROOT / "data" / "docs"))))
    index_dir: Path = field(default_factory=lambda: Path(os.getenv("INDEX_DIR", str(PROJECT_ROOT / "data" / "index"))))

    # Memory
    max_history_turns: int = field(default_factory=lambda: _env_int("MAX_HISTORY_TURNS", 20))


settings = Settings()
