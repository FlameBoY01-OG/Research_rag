import numpy as np
from ingest.embed import get_embedding
from ingest.chunk import chunk_text
from rag.generator import generate_answer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    # 1. Load multiple text files (simulate multiple documents)
    files = [
        "data/sample.txt",
        "data/sample2.txt",
    ]

    chunks = []

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        file_chunks = chunk_text(text)

        for chunk in file_chunks:
            chunks.append({
                "source": file,
                "text": chunk
            })

    # 2. Embed chunks
    embeddings = [get_embedding(c["text"]) for c in chunks]

    # 3. User question
    question = "What are CNNs used for?"
    question_embedding = get_embedding(question)

    # 4. Retrieve top-K chunks
    TOP_K = 3

    similarities = [
        cosine_similarity(question_embedding, emb)
        for emb in embeddings
    ]

    top_k_indices = np.argsort(similarities)[-TOP_K:][::-1]
    top_chunks = [chunks[i] for i in top_k_indices]

    # 5. Combine context with document citations
    combined_context = "\n\n".join(
        [
            f"[{c['source']}]\n{c['text']}"
            for c in top_chunks
        ]
    )

    # 6. Generate grounded answer
    answer = generate_answer(combined_context, question)

    print("QUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")
    for c in top_chunks:
        print(f"- {c['source']}")
