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

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1), top_k
        )

        return [self.metadata[i] for i in indices[0] if i != -1]

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str, metadata: list):
        self.index = faiss.read_index(path)
        self.metadata = metadata
