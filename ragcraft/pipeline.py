"""End-to-end RAG pipeline."""
from ragcraft.types import Document, Chunk, RetrievalResult, GenerationResult
from ragcraft.retriever import Retriever
from ragcraft.loaders import chunk_text
from ragcraft.embedder import get_embeddings
import uuid
import os


class Pipeline:
    def __init__(self, embedding_model: str = "text-embedding-3-small", dimension: int = 1536):
        self.embedding_model = embedding_model
        self.retriever = Retriever(dimension)
        self._docs: list[Document] = []

    def ingest(self, documents: list[Document], chunk_size: int = 500, overlap: int = 50):
        """Ingest documents: chunk, embed, index."""
        all_chunks = []
        all_texts = []
        for doc in documents:
            self._docs.append(doc)
            texts = chunk_text(doc.text, chunk_size, overlap)
            for t in texts:
                chunk = Chunk(id=str(uuid.uuid4()), doc_id=doc.id, text=t)
                all_chunks.append(chunk)
                all_texts.append(t)

        # batch embed
        if all_texts:
            embeddings = get_embeddings(all_texts, self.embedding_model)
            for chunk, emb in zip(all_chunks, embeddings):
                chunk.embedding = emb
            self.retriever.add(all_chunks)

        return len(all_chunks)

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """Retrieve relevant chunks for a query."""
        q_emb = get_embeddings([query], self.embedding_model)[0]
        result = self.retriever.search(q_emb, k)
        result.query = query
        return result

    def generate(self, query: str, k: int = 5, model: str = "gpt-4o-mini") -> GenerationResult:
        """Retrieve and generate an answer."""
        import openai
        retrieval = self.retrieve(query, k)
        context = "\n\n".join(c.text for c in retrieval.chunks)
        prompt = f"Answer the question based on the following context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        return GenerationResult(
            query=query,
            answer=answer,
            source_chunks=retrieval.chunks,
            tokens_used=tokens,
        )
