import os
import pickle
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from collections import defaultdict
import shutil

from ingest.embed import get_embedding
from ingest.load_pdf import load_pdfs
from ingest.chunk import chunk_pdf_documents

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

chat_memory = defaultdict(list)


# -----------------------------
# Models
# -----------------------------
class QuestionRequest(BaseModel):
    question: str
    session_id: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
def load_vector_db():
    global store

    if not os.path.exists(FAISS_PATH) or not os.path.exists(META_PATH):
        raise RuntimeError("Run `python -m rag.ingest_index` first")

    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    store = FaissVectorStore(dim=768)
    store.load(FAISS_PATH, metadata)


# -----------------------------
# Helpers
# -----------------------------
def normalize_scores(chunks):
    scores = [c["score"] for c in chunks]
    max_s, min_s = max(scores), min(scores)

    for c in chunks:
        if max_s == min_s:
            c["confidence"] = 100
        else:
            c["confidence"] = round(
                100 * (c["score"] - min_s) / (max_s - min_s), 1
            )
    return chunks


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health():
    return {"status": "running"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    question = req.question.strip()
    session_id = req.session_id

    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    history = chat_memory[session_id]

    # Embed question
    query_embedding = get_embedding(question)

    # Retrieve
    top_chunks = store.search(query_embedding, top_k=3)
    if not top_chunks:
        return AnswerResponse(
            answer="I could not find the answer in the documents.",
            sources=[]
        )

    top_chunks = normalize_scores(top_chunks)

    # Conversation context
    conversation = "\n".join(history[-6:])

    context = conversation + "\n\n" + "\n\n".join(
        f"[{c['source']} | page {c['page']}]\n{c['text']}"
        for c in top_chunks
    )

    mode = classify_question(question)
    answer = generate_answer(context, question, mode=mode)

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


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs allowed")

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
