"""Entry point: run the browser-based chat UI.

    python web_app.py

Serves a single-page chat interface at http://localhost:5000 backed by the
same Claude client + RAG retriever used by the CLI.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import sys
import time
import uuid
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for  # noqa: E402
from werkzeug.utils import secure_filename  # noqa: E402

from chatbot.config import settings  # noqa: E402
from chatbot.llm_client import ClaudeClient, LLMError, OllamaClient, build_llm_client  # noqa: E402
from chatbot.memory import ConversationMemory  # noqa: E402
from chatbot.rag import Retriever  # noqa: E402

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)
app.config["TEMPLATES_AUTO_RELOAD"] = True

ALLOWED_DOC_EXTENSIONS = {".txt", ".md", ".pdf"}
AUTH_USERNAME = os.environ.get("AUTH_USERNAME")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD")
LOGIN_MAX_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 60

client: ClaudeClient | OllamaClient | None = None
vision_client: OllamaClient | None = None
retriever = Retriever()
_sessions: dict[str, ConversationMemory] = {}
_login_failures: dict[str, list[float]] = {}


def get_client() -> ClaudeClient | OllamaClient:
    global client
    if client is None:
        client = build_llm_client()
    return client


def get_vision_client() -> OllamaClient:
    # Image analysis always goes through a local Ollama vision model,
    # regardless of LLM_PROVIDER — Claude vision isn't wired up here.
    global vision_client
    if vision_client is None:
        vision_client = OllamaClient()
    return vision_client


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def get_memory() -> ConversationMemory:
    session_id = session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory(max_turns=settings.max_history_turns)
    return _sessions[session_id]


def _client_ip() -> str:
    return request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()


def _is_locked_out(ip: str) -> bool:
    attempts = [t for t in _login_failures.get(ip, []) if time.time() - t < LOGIN_LOCKOUT_SECONDS]
    _login_failures[ip] = attempts
    return len(attempts) >= LOGIN_MAX_ATTEMPTS


def _record_failure(ip: str) -> None:
    _login_failures.setdefault(ip, []).append(time.time())


def _check_credentials(username: str, password: str) -> bool:
    if not AUTH_USERNAME or not AUTH_PASSWORD:
        return False
    return hmac.compare_digest(username, AUTH_USERNAME) and hmac.compare_digest(password, AUTH_PASSWORD)


@app.before_request
def require_login():
    if not AUTH_USERNAME or not AUTH_PASSWORD:
        # No credentials configured: auth is off (local-only use).
        return None
    if request.endpoint in {"login", "static"}:
        return None
    if not session.get("authenticated"):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not authenticated."}), 401
        return redirect(url_for("login"))
    return None


@app.route("/login", methods=["GET", "POST"])
def login():
    if not AUTH_USERNAME or not AUTH_PASSWORD:
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        ip = _client_ip()
        if _is_locked_out(ip):
            error = "Too many attempts. Try again in a minute."
        else:
            username = (request.form.get("username") or "").strip()
            password = request.form.get("password") or ""
            if _check_credentials(username, password):
                session.clear()
                session["authenticated"] = True
                session.permanent = True
                return redirect(url_for("index"))
            _record_failure(ip)
            error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def index():
    is_ollama = settings.llm_provider.strip().lower() == "ollama"
    return render_template(
        "index.html",
        model=settings.ollama_model if is_ollama else settings.model,
        rag_ready=retriever.is_ready(),
        auth_enabled=bool(AUTH_USERNAME and AUTH_PASSWORD),
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    use_rag = bool(data.get("use_rag", True))
    image_data_url = (data.get("image") or "").strip() or None
    truncate_to = data.get("truncate_to")

    if not message and not image_data_url:
        return jsonify({"error": "Message cannot be empty."}), 400

    memory = get_memory()
    if isinstance(truncate_to, int) and truncate_to >= 0:
        # Rewinding history: the client is editing or retrying an earlier turn.
        memory.truncate_to_turns(truncate_to)

    context_chunks = None
    sources: list[str] = []
    if image_data_url:
        # Image turns are single-shot: no conversation history, no RAG.
        source_mode = "image"
        image_b64 = image_data_url.split(",", 1)[-1]
        stream = get_vision_client().stream_reply_with_image(message, image_b64)
    else:
        if use_rag and retriever.is_ready():
            results = retriever.retrieve(message)
            context_chunks = [f"[Source: {r.source}]\n{r.text}" for r in results]
            sources = sorted({r.source for r in results})
            source_mode = "rag" if sources else "no_match"
        else:
            source_mode = "no_rag"
        memory.add("user", message)
        stream = get_client().stream_reply(memory.as_list(), context_chunks)

    def generate():
        yield _sse("sources", {"sources": sources, "source_mode": source_mode})
        reply_parts: list[str] = []
        try:
            for chunk in stream:
                reply_parts.append(chunk)
                yield _sse("token", {"text": chunk})
        except LLMError as exc:
            if not image_data_url:
                memory.messages.pop()
            yield _sse("error", {"error": str(exc)})
            return
        if not image_data_url:
            memory.add("assistant", "".join(reply_parts))
        yield _sse("done", {})

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/upload_doc", methods=["POST"])
def upload_doc():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file provided."}), 400

    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_DOC_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext or '(none)'}"}), 400

    retriever.docs_dir.mkdir(parents=True, exist_ok=True)
    file.save(retriever.docs_dir / filename)

    count = retriever.ingest()
    return jsonify({"filename": filename, "chunks_indexed": count, "ready": retriever.is_ready()})


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

    if not AUTH_USERNAME or not AUTH_PASSWORD:
        print(
            "[warn] AUTH_USERNAME/AUTH_PASSWORD not set — the web UI has no login. "
            "Fine for localhost-only use; set both in .env before exposing this publicly.",
            file=sys.stderr,
        )

    if debug_mode:
        app.run(debug=True, host=host, port=port)
    else:
        from waitress import serve

        print(f"Serving on http://{host}:{port} (waitress)")
        serve(app, host=host, port=port)
