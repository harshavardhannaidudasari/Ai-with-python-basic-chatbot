"""Interactive terminal chat client with streaming responses and RAG."""

from __future__ import annotations

import sys

from .config import settings
from .llm_client import ClaudeClient, LLMError
from .memory import ConversationMemory
from .rag import Retriever

HELP_TEXT = """
Commands:
  /help            Show this help message
  /ingest          (Re)build the knowledge base index from data/docs/
  /rag on|off      Toggle retrieval-augmented answers
  /reset           Clear conversation history
  /exit, /quit     Exit the chatbot
"""


def _print_banner(rag_enabled: bool, rag_ready: bool) -> None:
    print("=" * 60)
    print(" Claude AI Chatbot  (RAG-enabled)")
    print("=" * 60)
    print(f" Model: {settings.model}")
    rag_status = "on" if rag_enabled else "off"
    kb_status = "indexed" if rag_ready else "empty — run /ingest"
    print(f" RAG: {rag_status}  |  Knowledge base: {kb_status}")
    print(" Type /help for commands, /exit to quit.")
    print("=" * 60)


def run() -> None:
    if not settings.anthropic_api_key:
        print(
            "[warn] ANTHROPIC_API_KEY is not set. Set it in a .env file or your "
            "environment, or run `ant auth login`.",
            file=sys.stderr,
        )

    try:
        client = ClaudeClient()
    except LLMError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    memory = ConversationMemory(max_turns=settings.max_history_turns)
    retriever = Retriever()
    rag_enabled = settings.rag_enabled_default

    _print_banner(rag_enabled, retriever.is_ready())

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            command, _, arg = user_input.partition(" ")
            command = command.lower()

            if command in {"/exit", "/quit"}:
                print("Goodbye!")
                break
            if command == "/help":
                print(HELP_TEXT)
                continue
            if command == "/reset":
                memory.reset()
                print("[ok] Conversation history cleared.")
                continue
            if command == "/ingest":
                print("Indexing documents in data/docs/ ...")
                count = retriever.ingest()
                if count:
                    print(f"[ok] Indexed {count} chunks from {retriever.docs_dir}")
                else:
                    print(f"[warn] No documents found in {retriever.docs_dir}")
                continue
            if command == "/rag":
                if arg.strip().lower() in {"on", "true", "1"}:
                    rag_enabled = True
                    print("[ok] RAG enabled.")
                elif arg.strip().lower() in {"off", "false", "0"}:
                    rag_enabled = False
                    print("[ok] RAG disabled.")
                else:
                    print("Usage: /rag on|off")
                continue

            print(f"[warn] Unknown command: {command}. Type /help for options.")
            continue

        context_chunks: list[str] | None = None
        if rag_enabled and retriever.is_ready():
            context_chunks = retriever.retrieve_context(user_input)

        memory.add("user", user_input)

        print("Claude: ", end="", flush=True)
        reply_parts: list[str] = []
        try:
            for chunk in client.stream_reply(memory.as_list(), context_chunks):
                print(chunk, end="", flush=True)
                reply_parts.append(chunk)
        except LLMError as exc:
            print(f"\n[error] {exc}")
            memory.messages.pop()  # drop the unanswered user turn
            continue
        print()

        memory.add("assistant", "".join(reply_parts))


if __name__ == "__main__":
    run()
