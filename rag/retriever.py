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
    question = "What optimizer do transformers use?"
    question_embedding = get_embedding(question)

    # 5. Retrieve best chunk
    similarities = [
        cosine_similarity(question_embedding, emb)
        for emb in chunk_embeddings
    ]
    best_idx = int(np.argmax(similarities))
    best_chunk = chunks[best_idx]

    # 6. Generate grounded answer
    answer = generate_answer(best_chunk, question)

    print("QUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)
