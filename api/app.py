import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ingest.embed import get_embedding
from rag.vectorstore import FaissVectorStore
from rag.generator import generate_answer
from rag.question_type import classify_question

from fastapi import UploadFile, File
import shutil
from ingest.load_pdf import load_pdfs
from ingest.chunk import chunk_pdf_documents


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

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    # 1. Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    # 2. Save PDF to disk
    pdf_path = f"data/papers/{file.filename}"

    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. Load only this PDF
    documents = load_pdfs("data/papers")
    new_docs = [d for d in documents if d["source"] == file.filename]

    if not new_docs:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from PDF"
        )

    # 4. Chunk only new document
    new_chunks = chunk_pdf_documents(new_docs)

    # 5. Add to FAISS index
    for c in new_chunks:
        emb = get_embedding(c["text"])
        store.add(emb, c)

    # 6. Persist updated index
    store.save(FAISS_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(store.metadata, f)

    return {
        "status": "success",
        "file": file.filename,
        "chunks_added": len(new_chunks)
    }
