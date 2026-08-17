# Architecture Overview

The chatbot is organized into a small number of focused modules under
`src/chatbot/`:

- `config.py` — loads all settings from environment variables / `.env`.
- `llm_client.py` — wraps the Anthropic Claude Messages API, handling
  streaming responses and system-prompt assembly.
- `memory.py` — keeps a bounded conversation history since the Messages API
  is stateless and the full transcript must be resent each turn.
- `rag/` — the retrieval pipeline:
  - `document_loader.py` reads `.txt`, `.md`, and `.pdf` files.
  - `chunker.py` splits text into overlapping chunks.
  - `embeddings.py` wraps a local `sentence-transformers` model.
  - `vector_store.py` is a small numpy-based cosine-similarity index.
  - `retriever.py` ties the pieces together into `ingest()` and
    `retrieve()`.
- `cli.py` — an interactive terminal chat client.

`app.py` and `web_app.py` at the project root are the two entry points: a
terminal REPL and a Flask-based browser chat UI, respectively. Both share the
same underlying `ClaudeClient`, `ConversationMemory`, and `Retriever`
components.
