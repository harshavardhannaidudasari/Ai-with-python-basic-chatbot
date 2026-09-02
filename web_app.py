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
import threading
import time
import uuid
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for  # noqa: E402
from werkzeug.utils import secure_filename  # noqa: E402

from chatbot.config import settings  # noqa: E402
from chatbot.llm_client import (  # noqa: E402
    ClaudeClient,
    LLMError,
    OllamaClient,
    build_llm_client,
    synthesize_speech,
)
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
retriever = Retriever()
_sessions: dict[str, ConversationMemory] = {}
_login_failures: dict[str, list[float]] = {}


def get_client() -> ClaudeClient | OllamaClient:
    global client
    if client is None:
        client = build_llm_client()
    return client


def _warm_up() -> None:
    """Pay cold-start costs (model load, embedder load) at boot instead of on
    the first user's request.

    Only useful for Ollama: Claude has no local model to load, and sending it
    a throwaway request would just cost money for nothing. Only the text
    model is warmed, not vision — on 8GB RAM the two don't comfortably stay
    loaded at once, so warming both would just make one evict the other
    before it's ever used.

    The embedder warms unconditionally — not just `if retriever.is_ready()`
    (an existing index to query) — because loading sentence-transformers is
    also needed for the *first* document a fresh deploy ever ingests, and
    that's exactly the request that used to pay the full cold-load cost
    live: confirmed on Render's free tier, uploading the very first document
    after a deploy timed out with "Unexpected end of JSON input" (the proxy
    cutting off a request that took too long importing torch/
    sentence-transformers and downloading model weights on first use).
    """
    try:
        retriever.embedder  # noqa: B018 - property access triggers the lazy load
    except Exception as exc:
        print(f"[warn] embedder warm-up failed: {exc}", file=sys.stderr)

    if settings.llm_provider.strip().lower() == "ollama":
        try:
            list(get_client().stream_reply([{"role": "user", "content": "Hi"}]))
        except Exception as exc:
            print(f"[warn] Ollama warm-up failed: {exc}", file=sys.stderr)


def _parse_data_url(data_url: str) -> tuple[str, str]:
    """Split a `data:<media_type>;base64,<data>` URL into `(media_type, data)`."""
    header, _, encoded = data_url.partition(",")
    media_type = "image/jpeg"
    if header.startswith("data:") and ";base64" in header:
        media_type = header[len("data:"):].split(";", 1)[0] or media_type
    return media_type, encoded


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


_DISPLAY_MODEL_BY_PROVIDER = {
    "ollama": lambda: settings.ollama_model,
    "groq": lambda: settings.groq_model,
    "claude": lambda: settings.model,
}


@app.route("/")
def index():
    provider = settings.llm_provider.strip().lower()
    display_model = _DISPLAY_MODEL_BY_PROVIDER.get(provider, lambda: settings.model)()
    return render_template(
        "index.html",
        model=display_model,
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

    try:
        llm = get_client()
    except LLMError as exc:
        return jsonify({"error": f"LLM is not configured correctly: {exc}"}), 500

    context_chunks = None
    sources: list[str] = []
    if image_data_url:
        # Image turns are single-shot: no conversation history, no RAG.
        source_mode = "image"
        media_type, image_b64 = _parse_data_url(image_data_url)
        stream = llm.stream_reply_with_image(message, image_b64, media_type)
    else:
        if use_rag and retriever.is_ready():
            try:
                results = retriever.retrieve(message)
            except Exception as exc:
                return jsonify({"error": f"Knowledge base lookup failed: {exc}"}), 500
            context_chunks = [f"[Source: {r.source}]\n{r.text}" for r in results]
            sources = sorted({r.source for r in results})
            source_mode = "rag" if sources else "no_match"
        else:
            source_mode = "no_rag"
        memory.add("user", message)
        stream = llm.stream_reply(memory.as_list(), context_chunks)

    def generate():
        reply_parts: list[str] = []
        error: str | None = None
        try:
            # The disconnect this guards against can happen at *any* yield,
            # including this very first one — a client that vanishes before
            # a single token arrives is the same dangling-message scenario as
            # one that vanishes mid-reply. Both must go through the `finally`
            # below, so both live inside this same try block.
            yield _sse("sources", {"sources": sources, "source_mode": source_mode})
            for chunk in stream:
                reply_parts.append(chunk)
                yield _sse("token", {"text": chunk})
        except LLMError as exc:
            error = str(exc)
        finally:
            # Runs even if the client disconnects mid-stream (e.g. the tab is
            # backgrounded on mobile and the connection drops): without this,
            # the earlier `memory.add("user", message)` is left dangling with
            # no matching assistant turn, and the *next* message sent breaks
            # the API's required user/assistant alternation — the conversation
            # can't continue and retry can't recover it either.
            if not image_data_url:
                if error:
                    memory.messages.pop()
                elif reply_parts:
                    memory.add("assistant", "".join(reply_parts))
                else:
                    memory.messages.pop()
        if error:
            yield _sse("error", {"error": error})
            return
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

    try:
        count = retriever.ingest()
    except Exception as exc:
        return jsonify({"error": f"Failed to index documents: {exc}"}), 500

    file_chunks = retriever.chunks_for_source(filename)
    response = {
        "filename": filename,
        "chunks_indexed": count,
        "file_chunks": file_chunks,
        "ready": retriever.is_ready(),
    }
    if file_chunks == 0:
        response["warning"] = (
            f"No extractable text was found in {filename} — it may be a scanned/"
            "image-based PDF, empty, or in an unsupported encoding. It was not "
            "added to the knowledge base."
        )
    return jsonify(response)


@app.route("/api/speak", methods=["POST"])
def speak():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    # Called once per sentence, so real requests are always short — this
    # just bounds worst-case cost/latency against a malformed/abusive one.
    text = text[:2000]
    # From the frontend's voice picker (voice-select in the voice panel) —
    # both optional, and only meaningful together: `provider` pins which
    # engine to try first, `voice` overrides that engine's own configured
    # voice/speaker for this call. An unrecognized/stale provider value
    # (e.g. a saved choice for an engine no longer configured) just falls
    # through the normal priority order in synthesize_speech(), no
    # validation needed here beyond bounding the length.
    provider = (data.get("provider") or "").strip()[:32] or None
    voice = (data.get("voice") or "").strip()[:64] or None

    try:
        audio_bytes = synthesize_speech(text, provider=provider, voice=voice)
    except LLMError as exc:
        return jsonify({"error": str(exc)}), 503
    return Response(audio_bytes, mimetype="audio/wav")


@app.route("/api/ingest", methods=["POST"])
def ingest():
    try:
        count = retriever.ingest()
    except Exception as exc:
        return jsonify({"error": f"Failed to index documents: {exc}"}), 500
    return jsonify({"chunks_indexed": count, "ready": retriever.is_ready()})


@app.route("/api/reset", methods=["POST"])
def reset():
    get_memory().reset()
    return jsonify({"ok": True})


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    # Render (and most PaaS hosts) assign the listen port via $PORT at runtime
    # rather than letting it be fixed in config, so it takes priority over
    # FLASK_PORT when present.
    port = int(os.getenv("PORT") or os.getenv("FLASK_PORT", "5000"))

    if not AUTH_USERNAME or not AUTH_PASSWORD:
        print(
            "[warn] AUTH_USERNAME/AUTH_PASSWORD not set — the web UI has no login. "
            "Fine for localhost-only use; set both in .env before exposing this publicly.",
            file=sys.stderr,
        )

    # Skip in the debug reloader's watcher process (it re-executes this whole
    # block but never serves requests) so warm-up doesn't run twice.
    if not debug_mode or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(target=_warm_up, daemon=True).start()

    if debug_mode:
        app.run(debug=True, host=host, port=port)
    else:
        from waitress import serve

        print(f"Serving on http://{host}:{port} (waitress)")
        serve(app, host=host, port=port)
