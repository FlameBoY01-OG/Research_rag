import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ingest.embed import get_embedding
from rag.vectorstore import FaissVectorStore
from rag.generator import generate_answer
from rag.question_type import classify_question


# -----------------------------
# Paths
# -----------------------------
INDEX_DIR = "rag/index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "metadata.pkl")


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Research Paper RAG API",
    description="Ask questions about research papers using a FAISS-backed RAG system",
    version="1.0.0"
)


# -----------------------------
# Request / Response Models
# -----------------------------
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


# -----------------------------
# Load Vector DB at Startup
# -----------------------------
@app.on_event("startup")
def load_vector_db():
    global store

    if not os.path.exists(FAISS_PATH) or not os.path.exists(META_PATH):
        raise RuntimeError(
            "FAISS index not found. Run `python -m rag.ingest_index` first."
        )

    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    store = FaissVectorStore(dim=768)
    store.load(FAISS_PATH, metadata)


# -----------------------------
# Health Check (Optional but Pro)
# -----------------------------
@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "Research Paper RAG API is live. Visit /docs to use the API."
    }


# -----------------------------
# Main RAG Endpoint
# -----------------------------
@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    question = req.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # Embed question
    query_embedding = get_embedding(question)

    # Retrieve relevant chunks
    top_chunks = store.search(query_embedding, top_k=3)

    if not top_chunks:
        return AnswerResponse(
            answer="I could not find the answer in the provided documents.",
            sources=[]
        )

    # Build context with citations
    context = "\n\n".join(
        f"[{c['source']} | page {c['page']}]\n{c['text']}"
        for c in top_chunks
    )

    # Classify question intent
    mode = classify_question(question)

    # Generate answer
    answer = generate_answer(context, question, mode=mode)

    # Prepare sources
    sources = [
        f"{c['source']} (page {c['page']})"
        for c in top_chunks
    ]

    return AnswerResponse(
        answer=answer,
        sources=sources
    )
