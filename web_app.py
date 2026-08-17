"""Entry point: run the browser-based chat UI.

    python web_app.py

Serves a single-page chat interface at http://localhost:5000 backed by the
same Claude client + RAG retriever used by the CLI.
"""

from __future__ import annotations

import os
import secrets
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from flask import Flask, jsonify, render_template, request, session  # noqa: E402

from chatbot.config import settings  # noqa: E402
from chatbot.llm_client import ClaudeClient, LLMError  # noqa: E402
from chatbot.memory import ConversationMemory  # noqa: E402
from chatbot.rag import Retriever  # noqa: E402

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)

client: ClaudeClient | None = None
retriever = Retriever()
_sessions: dict[str, ConversationMemory] = {}


def get_client() -> ClaudeClient:
    global client
    if client is None:
        client = ClaudeClient()
    return client


def get_memory() -> ConversationMemory:
    session_id = session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory(max_turns=settings.max_history_turns)
    return _sessions[session_id]


@app.route("/")
def index():
    return render_template(
        "index.html",
        model=settings.model,
        rag_ready=retriever.is_ready(),
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    use_rag = bool(data.get("use_rag", True))

    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    memory = get_memory()

    context_chunks = None
    sources: list[str] = []
    if use_rag and retriever.is_ready():
        results = retriever.retrieve(message)
        context_chunks = [f"[Source: {r.source}]\n{r.text}" for r in results]
        sources = sorted({r.source for r in results})

    memory.add("user", message)

    try:
        reply = get_client().reply(memory.as_list(), context_chunks)
    except LLMError as exc:
        memory.messages.pop()
        return jsonify({"error": str(exc)}), 502

    memory.add("assistant", reply)
    return jsonify({"reply": reply, "sources": sources})


@app.route("/api/ingest", methods=["POST"])
def ingest():
    count = retriever.ingest()
    return jsonify({"chunks_indexed": count, "ready": retriever.is_ready()})


@app.route("/api/reset", methods=["POST"])
def reset():
    get_memory().reset()
    return jsonify({"ok": True})


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(debug=debug_mode, host=host, port=port)
