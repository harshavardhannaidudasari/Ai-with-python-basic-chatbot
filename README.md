# OrbitPro

*Intelligence in Motion*

A Python chatbot powered by Anthropic's Claude API (or a local Ollama model),
with Retrieval-Augmented Generation (RAG) over your own documents. Comes with
both an interactive terminal client and a Flask-based browser chat UI.

## Features

- **Claude-powered chat** — streaming responses via the official `anthropic`
  Python SDK, with configurable model, effort, and system prompt.
- **Local model support** — swap in a local [Ollama](https://ollama.com) model
  (e.g. `llama3.2:1b`) instead of Claude, no API key or network required.
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

### Running tests

```bash
pytest
```

Tests cover the chunker, vector store, and conversation memory — no API key
or network access needed.

## Configuration

All settings live in `.env` (see `.env.example` for the full list):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `claude` | `claude` (API) or `ollama` (local, no key needed) |
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key (required when `LLM_PROVIDER=claude`) |
| `CLAUDE_MODEL` | `claude-opus-5` | Model to use — try `claude-haiku-4-5-20251001` for a faster/cheaper bot |
| `MAX_TOKENS` | `2048` | Max response length |
| `CLAUDE_EFFORT` | `medium` | Reasoning effort: `low`/`medium`/`high`/`xhigh`/`max` |
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
