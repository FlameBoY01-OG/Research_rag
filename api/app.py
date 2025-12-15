# api/app.py
import os
import pickle
import shutil
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from ingest.embed import get_embedding
from ingest.load_pdf import load_pdfs
from ingest.chunk import chunk_pdf_documents

from rag.vectorstore import FaissVectorStore
from rag.generator import generate_answer
from rag.question_type import classify_question

# =============================
# Paths
# =============================
INDEX_DIR = "rag/index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "metadata.pkl")

# =============================
# App
# =============================
app = FastAPI(
    title="Research Paper RAG API",
    description="Ask, summarize, and compare research papers using RAG",
    version="1.0.0"
)

chat_memory = defaultdict(list)


# =============================
# Models
# =============================
class QuestionRequest(BaseModel):
    question: str
    session_id: str
    role: str = "student"


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


class CompareRequest(BaseModel):
    paper_a: str
    paper_b: str
    role: str = "student"


# =============================
# Startup: load vector DB
# =============================
@app.on_event("startup")
def load_vector_db():
    global store

    if not os.path.exists(FAISS_PATH) or not os.path.exists(META_PATH):
        raise RuntimeError("Run `python -m rag.ingest_index` first")

    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    store = FaissVectorStore(dim=768)
    store.load(FAISS_PATH, metadata)


# =============================
# Helpers
# =============================
def normalize_scores(chunks):
    scores = [c["score"] for c in chunks]
    max_s, min_s = max(scores), min(scores)

    for c in chunks:
        if max_s == min_s:
            c["confidence"] = 100.0
        else:
            c["confidence"] = round(
                100 * (c["score"] - min_s) / (max_s - min_s), 1
            )
    return chunks


def safe_generate_answer(context: str, question: str, mode: Optional[str] = None, role: Optional[str] = None) -> str:
    """
    Call generate_answer safely:
    - Try calling with role if provided
    - Fall back gracefully to older signatures if needed
    """
    try:
        if role is None:
            return generate_answer(context=context, question=question, mode=mode)
        else:
            return generate_answer(context=context, question=question, mode=mode, role=role)
    except TypeError:
        # signature might not accept role or mode keyword names; try fewer args
        try:
            return generate_answer(context=context, question=question)
        except Exception as e:
            # as a last resort, re-raise to surface the underlying problem
            raise RuntimeError(f"generate_answer failed: {e}") from e


# =============================
# Routes
# =============================
@app.get("/")
def health():
    return {"status": "running"}


# -----------------------------
# Ask
# -----------------------------
@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    question = req.question.strip()
    session_id = req.session_id
    role = req.role.lower()

    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    history = chat_memory[session_id]

    # Embed query
    try:
        query_embedding = get_embedding(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed question: {e}")

    # Retrieve using MMR
    try:
        top_chunks = store.search_mmr(query_embedding, top_k=3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    if not top_chunks:
        return AnswerResponse(
            answer="I could not find the answer in the documents.",
            sources=[]
        )

    top_chunks = normalize_scores(top_chunks)

    # Conversation memory
    conversation = "\n".join(history[-6:])

    context = conversation + "\n\n" + "\n\n".join(
        f"[{c['source']} | page {c['page']}]\n{c['text']}"
        for c in top_chunks
    )

    mode = classify_question(question)

    # Generate answer using safe wrapper
    try:
        answer = safe_generate_answer(context=context, question=question, mode=mode, role=role)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

    history.append(f"User: {question}")
    history.append(f"Assistant: {answer}")

    # Deduplicate sources
    seen = {}
    for c in top_chunks:
        key = (c["source"], c["page"])
        if key not in seen or c["confidence"] > seen[key]["confidence"]:
            seen[key] = c

    sources = [
        f"{c['source']} (page {c['page']}) — {c['confidence']}%"
        for c in seen.values()
    ]

    return AnswerResponse(answer=answer, sources=sources)


# -----------------------------
# Upload PDF
# -----------------------------
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs allowed")

    os.makedirs("data/papers", exist_ok=True)
    pdf_path = f"data/papers/{file.filename}"

    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    documents = load_pdfs("data/papers")
    new_docs = [d for d in documents if d["source"] == file.filename]

    if not new_docs:
        raise HTTPException(status_code=400, detail="Failed to read PDF")

    new_chunks = chunk_pdf_documents(new_docs)

    for c in new_chunks:
        emb = get_embedding(c["text"])
        store.add(emb, c)

    store.save(FAISS_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(store.metadata, f)

    return {
        "status": "success",
        "file": file.filename,
        "chunks_added": len(new_chunks)
    }


# -----------------------------
# Summarize
# -----------------------------
@app.post("/summarize")
def summarize_papers(role: str = "student"):
    if not store.metadata:
        raise HTTPException(status_code=400, detail="No papers indexed")

    context = "\n\n".join(
        f"[{c['source']} | page {c['page']}]\n{c['text']}"
        for c in store.metadata[:8]
    )

    summary_prompt = (
        "Provide a structured summary including:\n"
        "- Problem statement\n"
        "- Methods\n"
        "- Contributions\n"
        "- Conclusions"
    )

    try:
        summary = safe_generate_answer(context=context, question=summary_prompt, mode="summarize", role=role)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {e}")

    return {"summary": summary}


# -----------------------------
# Compare
# -----------------------------
@app.post("/compare")
def compare_papers(req: CompareRequest):
    role = req.role.lower()

    chunks_a = [c for c in store.metadata if c["source"] == req.paper_a]
    chunks_b = [c for c in store.metadata if c["source"] == req.paper_b]

    if not chunks_a or not chunks_b:
        raise HTTPException(status_code=400, detail="One or both papers not found")

    context = (
        "PAPER A:\n"
        + "\n\n".join(c["text"] for c in chunks_a[:5])
        + "\n\nPAPER B:\n"
        + "\n\n".join(c["text"] for c in chunks_b[:5])
    )

    compare_prompt = (
        "Compare the two papers in terms of:\n"
        "- Goals\n"
        "- Methods\n"
        "- Strengths\n"
        "- Weaknesses\n"
        "- Key differences"
    )

    try:
        answer = safe_generate_answer(context=context, question=compare_prompt, mode="compare", role=role)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison generation error: {e}")

    return {"comparison": answer}


# -----------------------------
# List Papers
# -----------------------------
@app.get("/papers")
def list_papers():
    if not store.metadata:
        return {"papers": []}

    papers = sorted({c["source"] for c in store.metadata})
    return {"papers": papers}


# -----------------------------
# Paper text helper (for UI diff)
# -----------------------------
@app.get("/paper_text")
def get_paper_text(name: str):
    """
    Return the concatenated text of all chunks for a given paper name.
    Used by the UI to compute diffs / local summaries. Safe and read-only.
    """
    if not store.metadata:
        return {"text": ""}

    chunks = [c for c in store.metadata if c["source"] == name]
    if not chunks:
        return {"text": ""}

    text = "\n\n".join(c["text"] for c in chunks)
    return {"text": text}
