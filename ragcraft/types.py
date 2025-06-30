from pydantic import BaseModel


class Document(BaseModel):
    id: str
    text: str
# cleanup: revisit later
    metadata: dict = {}


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    embedding: list[float] | None = None
    metadata: dict = {}


class RetrievalResult(BaseModel):
    query: str
    chunks: list[Chunk]
    scores: list[float]

# todo: performance

class GenerationResult(BaseModel):
    query: str
    answer: str
    source_chunks: list[Chunk]
    tokens_used: int = 0

