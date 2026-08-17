# About This Project

This is a sample document used to demonstrate the chatbot's Retrieval-Augmented
Generation (RAG) capability.

## What is RAG?

Retrieval-Augmented Generation combines a language model with a search step
over a private knowledge base. Instead of relying only on what the model
learned during training, the application:

1. Splits your documents into small overlapping chunks.
2. Converts each chunk into a vector embedding using a local sentence
   embedding model.
3. Stores those vectors in a vector index.
4. At query time, embeds the user's question and finds the most similar
   chunks (cosine similarity).
5. Passes the retrieved chunks to Claude as context, so answers are grounded
   in your actual documents instead of the model's general training data.

## Why this matters

RAG lets a chatbot answer questions about content it was never trained on —
internal wikis, product docs, PDFs, support tickets, and so on — without
retraining or fine-tuning the underlying model.

## How to try it

Drop your own `.txt`, `.md`, or `.pdf` files into `data/docs/`, then run
`/ingest` in the CLI (or click "Reindex docs" in the web UI) to rebuild the
index. Ask a question related to those documents and the chatbot will cite
the source file it used.
