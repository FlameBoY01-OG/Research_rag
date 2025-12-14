import numpy as np
from ingest.embed import get_embedding
from ingest.chunk import chunk_text


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    # 1. Load text
    with open("data/sample.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # 2. Chunk text
    chunks = chunk_text(text)

    # 3. Embed chunks
    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

    # 4. User question
    question = "What are transformers good at?"
    question_embedding = get_embedding(question)

    # 5. Find most similar chunk
    similarities = [
        cosine_similarity(question_embedding, emb)
        for emb in chunk_embeddings
    ]

    best_index = int(np.argmax(similarities))

    print("QUESTION:")
    print(question)

    print("\nMOST RELEVANT CHUNK:")
    print(chunks[best_index])
