from rag.generator import generate_answer
import numpy as np
from ingest.embed import get_embedding
from ingest.chunk import chunk_text


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    # 1. Load document
    with open("data/sample.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # 2. Chunk document
    chunks = chunk_text(text)

    # 3. Embed chunks
    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

    # 4. User question
    question = "What are transformers good at?"
    question_embedding = get_embedding(question)

    # 5. Retrieve best chunk
    TOP_K = 3

    similarities = [
        cosine_similarity(question_embedding, emb)
        for emb in chunk_embeddings
    ]

    top_k_indices = np.argsort(similarities)[-TOP_K:][::-1]
    top_chunks = [chunks[i] for i in top_k_indices]


    # 6. Generate grounded answer
    combined_context = "\n\n".join(
    [f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(top_chunks)]
    )

    answer = generate_answer(combined_context, question)


    print("QUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")
    for idx in top_k_indices:
        print(f"- Chunk {idx + 1}")

