import requests
import numpy as np

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


def get_embedding(text: str) -> np.ndarray:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBED_MODEL,
            "prompt": text
        }
    )
    response.raise_for_status()
    return np.array(response.json()["embedding"])


if __name__ == "__main__":
    test_text = "Transformers use self-attention to model long-range dependencies."
    emb = get_embedding(test_text)

    print("Embedding length:", len(emb))
    print("First 8 values:", emb[:8])
