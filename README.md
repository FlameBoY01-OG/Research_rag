# Research Paper Chatbot (RAG)

A complete Retrieval-Augmented Generation (RAG) system for chatting with research papers using a fully offline LLM served by Ollama. Features FastAPI backend, Streamlit UI, and automated PDF ingestion with FAISS vector indexing.

## ✨ Features

- **Conversational Q&A**: Ask natural language questions and get grounded answers with page-level citations
- **One-Click Summaries**: Generate structured summaries of papers (problem, methods, contributions, conclusions)
- **Paper Comparison**: Side-by-side analysis comparing goals, methods, strengths, and differences
- **Dynamic PDF Upload**: Upload PDFs from the UI; automatically chunks, embeds, and indexes them
- **Role-Aware Responses**: Tailored answers for students (simple explanations), researchers (technical detail), and reviewers (critical analysis)
- **Advanced Retrieval**: Uses MMR (Maximal Marginal Relevance) to balance relevance and diversity
- **Fully Offline**: All embeddings and generation happen locally via Ollama
- **Conversational Memory**: Maintains chat context across questions in a session

## 🛠 Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Frontend**: Streamlit
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **LLM & Embeddings**: Ollama (`llama3.2:latest`, `nomic-embed-text`)
- **PDF Processing**: PyPDF
- **Core Libraries**: NumPy, Requests, Pydantic

## 📋 Prerequisites

1. **Python 3.10+** installed
2. **Ollama** running locally ([Install Ollama](https://ollama.ai))
3. Pull required models:
   ```bash
   ollama pull llama3.2:latest
   ollama pull nomic-embed-text
   ```
4. **(Optional)** Environment variables:
   - `OLLAMA_BASE`: Custom Ollama endpoint (default: `http://localhost:11434`)
   - `OLLAMA_GEN_MODEL`: Alternative generation model (default: `llama3.2:latest`)

## 🚀 Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Research_rag

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 📁 Project Structure

```
Research_rag/
├── api/
│   └── app.py                 # FastAPI server (ask, upload, summarize, compare)
├── ui/
│   └── app.py                 # Streamlit interface with chat/summary/compare modes
├── ingest/
│   ├── load_pdf.py           # PDF loader with page extraction
│   ├── chunk.py              # Text chunking with overlap
│   └── embed.py              # Ollama embedding wrapper
├── rag/
│   ├── vectorstore.py        # FAISS wrapper with MMR search
│   ├── generator.py          # LLM generation with role/mode prompting
│   ├── question_type.py      # Question classification (summary/explanation/fact)
│   ├── ingest_index.py       # CLI: Build FAISS index from data/papers/
│   └── query.py              # CLI: Query the index directly
├── data/
│   └── papers/               # Place your PDFs here
└── rag/index/
    ├── faiss.index           # Persisted FAISS index
    └── metadata.pkl          # Chunk metadata (source, page, text)
```

## ⚡ Quickstart

### 1. Prepare Your Papers

Place PDF research papers in `data/papers/`:

```bash
mkdir -p data/papers
cp your_paper.pdf data/papers/
```

### 2. Build the Index

```bash
python -m rag.ingest_index
```

This will:

- Load all PDFs from `data/papers/`
- Chunk documents with overlap (500 chars, 100 overlap)
- Generate 768-dim embeddings via Ollama
- Save FAISS index to `rag/index/`

### 3. Start the Backend

```bash
uvicorn api.app:app --reload
```

API runs at `http://localhost:8000` • Auto-reloads on code changes

### 4. Launch the UI

In a separate terminal:

```bash
streamlit run ui/app.py
```

UI opens at `http://localhost:8501`

### 5. Start Chatting!

- Upload more PDFs from the sidebar
- Switch between **Chat**, **Summarize**, and **Compare** modes
- Select your role (**Student**/**Researcher**/**Reviewer**) for tailored responses

## 🔧 API Reference

Base URL: `http://localhost:8000`

### Endpoints

#### Health Check

```bash
GET /
```

Returns `{"status": "running"}`

#### Ask Question

```bash
POST /ask
Content-Type: application/json

{
  "question": "What is the main contribution of this paper?",
  "session_id": "uuid-string",
  "role": "student"  # student | researcher | reviewer
}
```

Returns: `{"answer": "...", "sources": ["paper.pdf (page 3) — 95.2%", ...]}`

#### Upload PDF

```bash
POST /upload
Content-Type: multipart/form-data

file: <pdf-file>
```

Returns: `{"status": "success", "file": "paper.pdf", "chunks_added": 42}`

#### Summarize Papers

```bash
POST /summarize?role=student
```

Returns: `{"summary": "..."}`

#### Compare Papers

```bash
POST /compare
Content-Type: application/json

{
  "paper_a": "transformer.pdf",
  "paper_b": "bert.pdf",
  "role": "researcher"
}
```

Returns: `{"comparison": "..."}`

#### List Papers

```bash
GET /papers
```

Returns: `{"papers": ["paper1.pdf", "paper2.pdf"]}`

#### Get Paper Text

```bash
GET /paper_text?name=paper.pdf
```

Returns: `{"text": "concatenated chunk text..."}`

## 🔍 How It Works

### Ingestion Pipeline

```
1. PDF Loading (load_pdf.py)
   ↓
2. Text Chunking (chunk.py)
   • 500 characters per chunk
   • 100 character overlap
   ↓
3. Embeddings (embed.py)
   • Ollama nomic-embed-text
   • 768-dimensional vectors
   ↓
4. FAISS Indexing (vectorstore.py)
   • Cosine similarity
   • MMR search
```

### Query Pipeline

```
User Question
   ↓
1. Question Classification (question_type.py)
   • summary | explanation | fact
   ↓
2. Embedding (embed.py)
   ↓
3. MMR Retrieval (vectorstore.py)
   • Balance relevance & diversity
   • Top-k chunks with scores
   ↓
4. Context Building
   • Combine chunks with citations
   • Add conversation history
   ↓
5. Generation (generator.py)
   • Role-aware prompting
   • Grounded in context
   ↓
Answer + Citations
```

## 🎯 Advanced Features

### Maximal Marginal Relevance (MMR)

Retrieval uses MMR to balance:

- **Relevance**: Similarity to query
- **Diversity**: Dissimilarity to already-selected chunks

This prevents redundant results and provides broader context.

### Role-Aware Generation

- **Student**: Simple explanations, minimal jargon, short examples
- **Researcher**: Technical precision, domain terminology, complete details
- **Reviewer**: Critical analysis, identifies strengths/weaknesses/assumptions

### Question Classification

Automatically detects question type to optimize prompting:

- **Summary**: Structured overview (problem, methods, contributions, conclusions)
- **Explanation**: Detailed "how" and "why" answers
- **Fact**: Direct answers from context

### Conversational Memory

Maintains last 6 exchanges per session for context-aware follow-ups.

## ⚠️ Troubleshooting

### Backend Returns 500 on Ask/Upload

**Cause**: Ollama not running or models not pulled  
**Solution**:

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull required models
ollama pull llama3.2:latest
ollama pull nomic-embed-text
```

### Empty or Irrelevant Retrieval

**Cause**: Index not built or outdated  
**Solution**: Rebuild the index after adding PDFs

```bash
python -m rag.ingest_index
```

### Streamlit Cannot Reach Backend

**Cause**: API not running on port 8000  
**Solution**: Start Uvicorn or update `API_BASE` in [ui/app.py](ui/app.py)

```bash
uvicorn api.app:app --reload
```

### FAISS Import Error

**Cause**: Platform-specific installation issue  
**Solution**:

```bash
# For CPU-only systems
pip install faiss-cpu

# For GPU systems (optional)
pip install faiss-gpu
```

### "RuntimeError: Run ingest_index first"

**Cause**: No FAISS index found  
**Solution**: Create initial index

```bash
mkdir -p data/papers
cp your_paper.pdf data/papers/
python -m rag.ingest_index
```

### Slow Response Times

**Cause**: Large context or slow model  
**Solution**:

- Reduce `top_k` in retrieval (default: 3)
- Use smaller/faster Ollama model (e.g., `llama3.2:1b`)
- Adjust `max_tokens` in [rag/generator.py](rag/generator.py)

## 🛠️ Development

### CLI Tools

#### Build Index

```bash
python -m rag.ingest_index
```

Rebuilds FAISS index from all PDFs in `data/papers/`

#### Query Index Directly

```bash
python -m rag.query
```

Interactive CLI for testing retrieval and generation without the UI

### Configuration

#### Change Embedding Model

Edit [ingest/embed.py](ingest/embed.py):

```python
EMBED_MODEL = "nomic-embed-text"  # or mxbai-embed-large, etc.
```

#### Change Generation Model

Set environment variable:

```bash
export OLLAMA_GEN_MODEL="llama3.2:3b"  # or mixtral, etc.
```

#### Adjust Chunking

Edit [ingest/chunk.py](ingest/chunk.py):

```python
chunk_size = 500   # characters per chunk
overlap = 100      # overlap between chunks
```

#### Tune Retrieval

Edit [api/app.py](api/app.py) in the ask endpoint:

```python
top_chunks = store.search_mmr(
    query_embedding,
    top_k=3,           # number of chunks
    fetch_k=10,        # candidate pool
    lambda_mult=0.5    # relevance vs diversity (0-1)
)
```

### Testing

Run individual modules:

```bash
# Test embedding
python ingest/embed.py

# Test chunking
python ingest/chunk.py

# Test retrieval pipeline
python rag/retriever.py
```
## 📝 License

MIT License - feel free to use this project for research or commercial purposes.

