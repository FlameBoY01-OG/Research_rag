import faiss
import numpy as np


class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def add(self, embedding: np.ndarray, meta: dict):
        embedding = embedding / np.linalg.norm(embedding)
        self.index.add(embedding.reshape(1, -1))
        self.metadata.append(meta)

    def search(self, query_embedding, top_k=3):
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), top_k
        )

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            chunk = self.metadata[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "score": float(score)
            })

        return results

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str, metadata: list):
        self.index = faiss.read_index(path)
        self.metadata = metadata
