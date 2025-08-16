"""Vector store retriever using FAISS."""
import numpy as np
from ragcraft.types import Chunk, RetrievalResult


class Retriever:
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.chunks: list[Chunk] = []
        self._index = None

    def _ensure_index(self):
        if self._index is None:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)

    def add(self, chunks: list[Chunk]):
        self._ensure_index()
        vectors = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"chunk {chunk.id} has no embedding")
            vectors.append(chunk.embedding)
            self.chunks.append(chunk)
        arr = np.array(vectors, dtype="float32")
        # normalize for cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        arr = arr / norms
        self._index.add(arr)

    def search(self, query_embedding: list[float], k: int = 5) -> RetrievalResult:
        self._ensure_index()
        q = np.array([query_embedding], dtype="float32")
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        scores, indices = self._index.search(q, min(k, len(self.chunks)))
        result_chunks = []
        result_scores = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            result_chunks.append(self.chunks[idx])
            result_scores.append(float(scores[0][i]))
        return RetrievalResult(query="", chunks=result_chunks, scores=result_scores)

    @property
    def size(self) -> int:
        return len(self.chunks)

