import numpy as np
from ingest.embed import get_embedding
from ingest.load_pdf import load_pdfs
from ingest.chunk import chunk_pdf_documents
from rag.generator import generate_answer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    # 1. Load PDFs (real research papers)
    documents = load_pdfs("data/papers")

    if len(documents) == 0:
        raise RuntimeError("No PDFs found in data/papers")

    # 2. Chunk PDFs with page info
    chunks = chunk_pdf_documents(documents)

    # 3. Embed chunks
    embeddings = [get_embedding(c["text"]) for c in chunks]

    # 4. User question (interactive)
    question = input("Enter your question: ").strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    question_embedding = get_embedding(question)

    # 5. Retrieve top-K relevant chunks
    TOP_K = 3

    similarities = [
        cosine_similarity(question_embedding, emb)
        for emb in embeddings
    ]

    top_k_indices = np.argsort(similarities)[-TOP_K:][::-1]
    top_chunks = [chunks[i] for i in top_k_indices]

    # 6. Combine context with document + page citations
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
