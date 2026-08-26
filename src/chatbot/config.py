"""Central configuration, loaded from environment variables / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# override=True: this project's .env must win over ambient environment
# variables of the same name set by unrelated tools (e.g. Ollama's own server
# is commonly configured system-wide with OLLAMA_HOST=0.0.0.0:<port> for LAN
# binding, which is not a valid address for *this app* to connect to as a
# client — without override, that pre-existing OS-level value silently wins
# over .env's http://localhost:11434 and breaks every LLM call).
load_dotenv(override=True)

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
    # llama-3.3-70b-versatile was Groq's default here until it was
    # decommissioned 08/16/26; openai/gpt-oss-120b is Groq's recommended
    # replacement (see https://console.groq.com/docs/deprecations).
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    )
    # meta-llama/llama-4-scout-17b-16e-instruct was decommissioned 07/17/26;
    # qwen/qwen3.6-27b is Groq's recommended replacement and, as of this
    # writing, the only current Groq-hosted model that still accepts image
    # input (see https://console.groq.com/docs/vision).
    groq_vision_model: str = field(
        default_factory=lambda: os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
    )
    # Text-to-speech for voice mode — independent of LLM_PROVIDER (works even
    # when chat itself is served by Claude or Ollama) since Groq's Orpheus
    # voice sounds like a real person, unlike any browser's built-in TTS.
    # (playai-tts, the previous default here, was decommissioned 12/31/25 —
    # verified live: Groq now rejects it with model_decommissioned. Orpheus
    # is its replacement.) Full voice list:
    # https://console.groq.com/docs/text-to-speech
    groq_tts_model: str = field(
        default_factory=lambda: os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
    )
    # Female English voices: hannah, autumn, diana. Male: troy, austin, daniel.
    groq_tts_voice: str = field(default_factory=lambda: os.getenv("GROQ_TTS_VOICE", "hannah"))
    # Optional local voice cloning (Coqui XTTS-v2) — when set, voice mode
    # speaks in this specific cloned voice instead of a Groq preset. Path to
    # a short (6s+) reference audio sample. Free (no API), but only works on
    # a machine with requirements-voice-clone.txt installed; leave unset on
    # Render (or any low-memory host) and it silently uses Groq instead.
    voice_clone_reference: str = field(
        default_factory=lambda: os.getenv("VOICE_CLONE_REFERENCE", "")
    )
    voice_clone_language: str = field(
        default_factory=lambda: os.getenv("VOICE_CLONE_LANGUAGE", "en")
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
