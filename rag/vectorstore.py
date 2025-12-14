import faiss
import numpy as np


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


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

    # 🔥 MMR SEARCH (ADD THIS HERE)
    def search_mmr(self, query_embedding, top_k=3, fetch_k=10, lambda_mult=0.5):
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # 1. Fetch more candidates
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), fetch_k
        )

        candidates = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            meta = self.metadata[idx]
            candidates.append({
                "embedding": self.index.reconstruct(idx),
                "text": meta["text"],
                "source": meta["source"],
                "page": meta["page"],
                "score": float(score)
            })

        selected = []

        # 2. MMR selection
        while len(selected) < top_k and candidates:
            best = None
            best_score = -1

            for c in candidates:
                relevance = cosine_sim(query_embedding, c["embedding"])

                diversity = 0
                if selected:
                    diversity = max(
                        cosine_sim(c["embedding"], s["embedding"])
                        for s in selected
                    )

                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best = c

            selected.append(best)
            candidates.remove(best)

        return selected

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str, metadata: list):
        self.index = faiss.read_index(path)
        self.metadata = metadata
