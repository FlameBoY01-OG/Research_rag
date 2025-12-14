# Research Paper Chatbot (RAG)

A local Retrieval-Augmented Generation (RAG) system that allows users to
chat with research papers (PDFs) using a fully offline LLM via Ollama.

## Goals

- Ask natural language questions about research papers
- Ground answers in document content (no hallucinations)
- Work fully offline
- Modular, debuggable pipeline

## Architecture (High Level)

PDF → Chunking → Embeddings → Vector DB → Retrieval → LLM Answer

## Status

🚧 Under active development
