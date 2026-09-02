# OrbitProAI

*Intelligence in Motion*

A Python chatbot powered by Anthropic's Claude API, Groq's free API, or a
local Ollama model, with Retrieval-Augmented Generation (RAG) over your own
documents. Comes with both an interactive terminal client and a Flask-based
browser chat UI.

## Features

- **Claude-powered chat** — streaming responses via the official `anthropic`
  Python SDK, with configurable model, effort, and system prompt.
- **Free API option** — swap in [Groq](https://console.groq.com/keys)
  (no card required) instead of Claude, for deployments without a paid key.
- **Local model support** — swap in a local [Ollama](https://ollama.com) model
  (e.g. `llama3.2:1b`) instead, no API key or network required.
- **Retrieval-Augmented Generation** — index `.txt`, `.md`, and `.pdf` files
  and have the bot ground its answers in them, citing sources.
- **Local embeddings** — uses `sentence-transformers` for embeddings, so
  indexing your knowledge base costs nothing and works offline.
- **Conversation memory** — bounded multi-turn history per session.
- **Two interfaces** — a terminal REPL (`app.py`) and a browser chat UI
  (`web_app.py`).
- **Tested** — unit tests for the chunker, vector store, and memory
  components (no API key required to run them).

## Architecture

```
app.py            terminal chat entry point
web_app.py         Flask web chat entry point
src/chatbot/
├── config.py       settings loaded from environment / .env
├── llm_client.py   Claude Messages API wrapper (streaming)
├── memory.py        bounded conversation history
├── cli.py           interactive terminal client
└── rag/
    ├── document_loader.py   load .txt / .md / .pdf
    ├── chunker.py            overlapping text chunking
    ├── embeddings.py         local sentence-transformers embeddings
    ├── vector_store.py       numpy cosine-similarity vector index
    └── retriever.py          ties it all together (ingest + retrieve)
data/docs/          drop your knowledge-base files here
templates/, static/  web chat UI (HTML/CSS/JS)
tests/               unit tests
```

**RAG flow:** documents in `data/docs/` → chunked → embedded locally →
stored in a numpy-backed vector index → at query time, the question is
embedded and the most similar chunks are retrieved and injected into
Claude's system prompt as grounding context.

## Setup

1. **Clone and enter the project**

   ```bash
   git clone https://github.com/harshavardhannaidudasari/Ai-with-python-basic-chatbot.git
   cd Ai-with-python-basic-chatbot
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS/Linux

   pip install -r requirements.txt
   ```

3. **Configure your API key**

   ```bash
   copy .env.example .env         # Windows
   # cp .env.example .env         # macOS/Linux
   ```

   Then edit `.env` and set `ANTHROPIC_API_KEY` to your key from
   [console.anthropic.com](https://console.anthropic.com/settings/keys).

   **Or use Groq for free** — no card required. Get a key at
   [console.groq.com/keys](https://console.groq.com/keys), then:

   ```env
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_...
   ```

   **Or run fully local with Ollama** — install [Ollama](https://ollama.com),
   pull a small model, and switch the provider:

   ```bash
   ollama pull llama3.2:1b
   ```

   ```env
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama3.2:1b
   OLLAMA_HOST=http://localhost:11434
   ```

   No API key needed — just make sure the Ollama app/service is running.

## Usage

### Terminal chat

```bash
python app.py
```

```
You: /ingest
[ok] Indexed 6 chunks from data/docs

You: What is RAG?
Claude: Retrieval-Augmented Generation combines a language model with...
```

Commands: `/help`, `/ingest`, `/rag on|off`, `/reset`, `/exit`.

### Web chat

```bash
python web_app.py
```

Then open <http://localhost:5000>. Click **Reindex docs** once to build the
knowledge base, then chat — toggle "Use knowledge base" to compare answers
with and without retrieval.

### Adding your own knowledge base

Drop `.txt`, `.md`, or `.pdf` files into `data/docs/`, then run `/ingest`
(terminal) or click **Reindex docs** (web UI). The index is rebuilt from
scratch each time and persisted to `data/index/`.

### Voice cloning (optional, free, local)

Voice mode can speak in a specific cloned voice instead of one of Groq's
presets, using [Coqui XTTS-v2](https://github.com/idiap/coqui-ai-TTS) —
zero-shot cloning from a short reference clip, running entirely on your own
machine. It's free (no API key, no per-request cost) but heavy: it installs
~2GB of ML dependencies, downloads a ~1.9GB model on first use, and takes
several seconds of CPU time per sentence — noticeably slower than Groq's
near-instant API. It's also **not viable on Render's free tier** (not
enough memory) — this is a local-only feature; leave `VOICE_CLONE_REFERENCE`
unset in your Render environment and it uses Groq TTS instead, same as
before.

```bash
pip install -r requirements-voice-clone.txt
```

Then in `.env`:

```
VOICE_CLONE_REFERENCE=data/voice_reference/reference.mp3
VOICE_CLONE_LANGUAGE=en
```

`VOICE_CLONE_REFERENCE` points at a 6-second-or-longer sample of the target
voice (wav or mp3). `data/voice_reference/` is gitignored, so drop your own
sample there — it stays local, never committed. If the clone model fails to
load (dependencies missing, bad reference file, etc.), voice mode falls
back to Groq automatically, then to the browser's built-in voice if that
also fails — it never goes silent.

### Running tests

```bash
pytest
```

Tests cover the chunker, vector store, and conversation memory — no API key
or network access needed.

## Deploying to Render (stable URL)

`scripts/go_live.ps1` gives you a quick public URL via a Cloudflare tunnel,
but that address changes every time the tunnel restarts. For a URL that
stays fixed, deploy the web UI to [Render](https://render.com) — free tier,
no domain required.

1. Push this repo to GitHub (if you haven't already).
2. In the Render dashboard: **New > Blueprint**, connect the repo. Render
   reads `render.yaml` at the project root and configures the service
   automatically.
3. When prompted, set these environment variables (marked `sync: false` in
   `render.yaml` so Render asks for them instead of storing them in the
   blueprint):
   - `GROQ_API_KEY` — required, free (no card), from
     [console.groq.com/keys](https://console.groq.com/keys)
   - `AUTH_USERNAME` / `AUTH_PASSWORD` — required once the app is public;
     without these the web UI has no login

   The blueprint defaults to `LLM_PROVIDER=groq` so no paid API key is
   needed. Swap to `LLM_PROVIDER=claude` (and add `ANTHROPIC_API_KEY`
   instead) in Render's environment settings if you'd rather use Claude.
4. Deploy. Your stable URL is `https://orbitproai.onrender.com` (or
   `orbitproai-<suffix>.onrender.com` if that name is taken — Render shows
   the final URL after the first deploy).

Notes:

- The free plan spins the service down after 15 minutes of inactivity;
  the next request wakes it up, which can take a minute (slower than the
  local Ollama cold-start case `web_app.py`'s warm-up thread already
  handles — that thread only pays off once the process is already running).
- The free plan's disk is ephemeral: anything ingested into
  `data/docs/` / `data/index/` after deploy is lost on the next deploy or
  restart. Commit any knowledge-base files you want to ship as part of the
  repo (see `data/docs/about_this_project.md` for an example) rather than
  uploading them through the UI.
- `LLM_PROVIDER=ollama` won't work on Render — there's no Ollama server to
  point at. The blueprint defaults to `LLM_PROVIDER=groq`.

## Configuration

All settings live in `.env` (see `.env.example` for the full list):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `claude` | `claude` (API), `groq` (free API, no card), or `ollama` (local, no key needed) |
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key (required when `LLM_PROVIDER=claude`) |
| `CLAUDE_MODEL` | `claude-opus-5` | Model to use — try `claude-haiku-4-5-20251001` for a faster/cheaper bot |
| `MAX_TOKENS` | `2048` | Max response length |
| `CLAUDE_EFFORT` | `medium` | Reasoning effort: `low`/`medium`/`high`/`xhigh`/`max` |
| `GROQ_API_KEY` | — | Your Groq API key (required when `LLM_PROVIDER=groq`) — free, from [console.groq.com/keys](https://console.groq.com/keys) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq-hosted model to use when `LLM_PROVIDER=groq` |
| `TTS_PROVIDER` | — | Pins which voice-mode TTS engine to try first, overriding the default priority (`clone` > `gemini` > `sarvam` > `kokoro` > `groq`). One of `clone`/`gemini`/`sarvam`/`kokoro`/`groq` |
| `GEMINI_API_KEY` | — | Your Google AI Studio API key — free, no card, from [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Powers voice mode's spoken replies (Gemini TTS) whenever set, regardless of `LLM_PROVIDER`; preferred by default since its free tier is much harder to exhaust than Groq's |
| `GEMINI_TTS_VOICE` | `Kore` | Voice mode's spoken-reply voice (Gemini TTS). 30 presets — see [ai.google.dev/gemini-api/docs/speech-generation](https://ai.google.dev/gemini-api/docs/speech-generation) |
| `SARVAM_API_KEY` | — | Your Sarvam AI API key — free credits on signup, no card, from [dashboard.sarvam.ai](https://dashboard.sarvam.ai). Best choice for Indian-language voice replies: native Hindi/Tamil/Telugu/etc. and Hinglish code-switching, unlike Gemini/Groq |
| `SARVAM_TTS_LANGUAGE` | `en-IN` | BCP-47 language code for Sarvam TTS (`hi-IN`, `ta-IN`, `te-IN`, ...) |
| `SARVAM_TTS_SPEAKER` | `shubh` | Voice mode's spoken-reply voice (Sarvam TTS) — see [docs.sarvam.ai](https://docs.sarvam.ai) for the full speaker list |
| `KOKORO_TTS_ENABLED` | `false` | Enables local Kokoro TTS (82M-param, open-weight, free/unlimited) — needs `pip install -r requirements-kokoro.txt` first; not viable on Render's free tier |
| `KOKORO_TTS_VOICE` | `af_heart` | Voice mode's spoken-reply voice (Kokoro TTS) — see [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) for the full list and quality grades |
| `GROQ_TTS_VOICE` | `hannah` | Voice mode's spoken-reply voice (Groq Orpheus TTS) — used as a fallback when none of the engines above are configured/enabled. Female: `hannah`/`autumn`/`diana`. Male: `troy`/`austin`/`daniel` |
| `VOICE_CLONE_REFERENCE` | — | Path to a reference audio sample to clone for voice mode instead of a preset. Free, local-only — see [Voice cloning](#voice-cloning-optional-free-local) below |
| `OLLAMA_MODEL` | `llama3.2:1b` | Local model to use when `LLM_PROVIDER=ollama` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers embedding model |
| `TOP_K` | `4` | Number of chunks retrieved per query |
| `MIN_SIMILARITY` | `0.2` | Minimum cosine similarity for a chunk to be used |
| `MAX_HISTORY_TURNS` | `20` | Conversation turns kept in memory |

## Notes

- The vector store is a small, dependency-light numpy implementation chosen
  for readability — swap in Chroma/FAISS/pgvector for production scale
  without changing the `Retriever` interface.
- `sentence-transformers` downloads its model on first use and caches it
  locally.
