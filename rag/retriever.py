import numpy as np
from ingest.embed import get_embedding
from ingest.load_pdf import load_pdfs
from ingest.chunk import chunk_pdf_documents
from rag.generator import generate_answer
from rag.vectorstore import FaissVectorStore


if __name__ == "__main__":
    # 1. Load PDFs
    documents = load_pdfs("data/papers")

    if not documents:
        raise RuntimeError("No PDFs found in data/papers")

    # 2. Chunk PDFs with page metadata
    chunks = chunk_pdf_documents(documents)

    # 3. Build FAISS vector store
    DIM = 768
    store = FaissVectorStore(dim=DIM)

    for c in chunks:
        emb = get_embedding(c["text"])
        store.add(emb, c)

    # 4. User question
    question = input("Enter your question: ").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    query_embedding = get_embedding(question)

    # 5. Retrieve top-K relevant chunks
    top_chunks = store.search(query_embedding, top_k=3)

    # 6. Combine context with citations
    combined_context = "\n\n".join(
        [
            f"[{c['source']} | page {c['page']}]\n{c['text']}"
            for c in top_chunks
        ]
    )

    # 7. Generate grounded answer
    answer = generate_answer(combined_context, question)

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")
    for c in top_chunks:
        print(f"- {c['source']} (page {c['page']})")
