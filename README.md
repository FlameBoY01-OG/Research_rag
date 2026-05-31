# Research Paper RAG (Retrieval-Augmented Generation) System

Welcome! If you are reviewing this project (perhaps for an interview), this README serves as a comprehensive guide to understanding what this project is, how it works under the hood, and its underlying architecture. 

## 🌟 What is this Project?

This project is a fully offline **Retrieval-Augmented Generation (RAG)** system designed specifically for interacting with research papers (PDFs). It allows users to:
1. **Upload and Index** research papers dynamically.
2. **Chat/Q&A** with the papers to get grounded answers with exact page-level citations.
3. **Summarize** papers with a single click (extracting problem statements, methods, and contributions).
4. **Compare** two different papers side-by-side to contrast their goals, methods, and results.

It achieves this by combining a **FastAPI** backend for robust API endpoints, a **Streamlit** frontend for an interactive UI, and **Ollama** for running LLMs (Large Language Models) locally, ensuring complete privacy and offline capability.

---

## 🏗️ Architecture & How It Works

The system is broken down into two main pipelines: **Ingestion** (processing PDFs) and **Retrieval/Generation** (answering questions).

### 1. The Ingestion Pipeline (Data Preparation)
Before the system can answer questions, it needs to understand the documents. This happens in the following steps:
- **PDF Loading (`ingest/load_pdf.py`)**: Uses `pypdf` to parse uploaded PDF files and extract text page by page.
- **Chunking (`ingest/chunk.py`)**: The extracted text is split into smaller, overlapping segments (default 500 characters with 100 character overlap). Overlapping ensures that context is not lost at the boundaries of chunks.
- **Embedding (`ingest/embed.py`)**: Each text chunk is converted into a 768-dimensional numerical vector using the `nomic-embed-text` model via Ollama. Embeddings capture the semantic meaning of the text.
- **Vector Storage (`rag/vectorstore.py`)**: The embeddings are stored in a **FAISS** (Facebook AI Similarity Search) index using Inner Product (Cosine Similarity). FAISS allows for lightning-fast similarity searches across thousands of chunks. Metadata (source file, page number, raw text) is stored alongside it in a `.pkl` file.

### 2. The Retrieval & Generation Pipeline (Q&A)
When a user asks a question via the UI, the backend processes it as follows:
- **Question Classification (`rag/question_type.py`)**: The system determines if the user is asking for a summary, an explanation, or a specific fact based on the input text.
- **Query Embedding**: The user's question is embedded into a vector using the same `nomic-embed-text` model.
- **Advanced Retrieval (MMR)**: The system searches the FAISS index for chunks that are semantically similar to the question's embedding. Instead of just picking the top results, it uses **MMR (Maximal Marginal Relevance)**. MMR balances *relevance* (how well it answers the question) with *diversity* (ensuring we don't just pull 3 chunks that say the exact same thing).
- **Context Assembly**: The retrieved chunks are formatted into a context block, appending the source document name and page number. If it's a chat, the recent conversation history is also prepended to maintain context.
- **Generation (`rag/generator.py`)**: A prompt is constructed containing the System Role (Student, Researcher, or Reviewer), the assembled context, and the user's question. This is sent to the local `llama3.2:latest` model via Ollama to generate a grounded, natural language response.

---

## 💻 Tech Stack Deep Dive

- **Backend (FastAPI)**: Found in `api/app.py`. Exposes endpoints like `/ask`, `/upload`, `/summarize`, and `/compare`. Chosen for its speed, asynchronous capabilities, and automatic documentation generation (Swagger UI).
- **Frontend (Streamlit)**: Found in `ui/app.py`. Provides a chat interface, sidebar for file uploads, and specific modes for Q&A, Summarization, and Comparison.
- **Local LLM Engine (Ollama)**: Handles both text generation (`llama3.2:latest`) and embeddings (`nomic-embed-text`) entirely locally.
- **Vector Database (FAISS)**: An efficient, CPU-friendly library for dense vector similarity search, enabling quick retrieval even on machines without a GPU.

---

## 🚀 Getting Started

### Prerequisites
1. Python 3.10+
2. [Ollama](https://ollama.ai) installed and running locally.
3. Pull required models via terminal: 
   ```bash
   ollama pull llama3.2:latest
   ollama pull nomic-embed-text
   ```

### Installation
```bash
# Clone the repository and setup virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
1. **Start the API (Backend)**: 
   Open a terminal and run:
   ```bash
   uvicorn api.app:app --reload
   ```
   *The API will be available at `http://localhost:8000`*

2. **Start the UI (Frontend)**:
   Open a second terminal and run:
   ```bash
   streamlit run ui/app.py
   ```
   *The UI will open in your browser at `http://localhost:8501`*

*(Optional) You can place PDFs in `data/papers/` and run `python -m rag.ingest_index` to build the index manually via CLI.*

---

## 💡 Proposed Improvements (Future Roadmap)

If we were to scale this project or push it to production, here are several architectural and feature improvements that could be made:

### 1. Architectural & Scaling Improvements
- **Persistent Vector Database**: Migrate from a local FAISS index (pickle/binary files) to a robust, persistent vector database like **ChromaDB**, **Pinecone**, or **Milvus**. This would allow concurrent writes, better scaling, easier metadata filtering, and eliminate the need to manually manage index files.
- **Asynchronous Ingestion (Message Queue)**: Currently, uploading and indexing a PDF blocks the FastAPI thread. For large PDFs, this could cause timeouts. Implementing a task queue (like **Celery** or **Redis Queue**) would allow PDFs to be processed in the background, returning an immediate "processing" response to the UI.
- **Better Chunking Strategy**: Move from naive character-based chunking to **Semantic Chunking** or **Recursive Character Text Splitting** (e.g., using LangChain or LlamaIndex). This ensures we don't break sentences or paragraphs in half, preserving semantic meaning and improving context quality.

### 2. Retrieval Enhancements
- **Hybrid Search**: Combine dense vector search (FAISS) with sparse keyword search (like BM25). This is highly effective because dense search is great for semantic meaning, but sparse search is better for exact keyword matches (like specific acronyms, variables, or author names).
- **Re-ranking**: Introduce a cross-encoder model (e.g., Cohere Re-rank or a local `sentence-transformers` cross-encoder) after the initial FAISS retrieval. The vector DB could fetch the top 20 results quickly, and the cross-encoder precisely ranks the top 5 to pass to the LLM, drastically improving accuracy.

### 3. User Experience & Application Features
- **Advanced Document Parsing**: Replace `pypdf` with a more advanced OCR/Parsing tool (like `Nougat`, `Grobid`, or `Unstructured.io`) that can correctly parse tables, math formulas, and multi-column layouts typical in academic papers.
- **Source Highlighting**: Pass the exact bounding boxes of the text back to the frontend so the Streamlit UI can render the PDF visually and highlight the exact paragraph the LLM used to answer the question.
- **Session & User Management**: Add a relational database (like PostgreSQL/SQLite) to store user profiles, maintain persistent multi-turn chat histories across sessions, and allow users to manage their personal library of papers.
