"""Generate embeddings via OpenAI API."""
import os
from typing import Optional


def get_embeddings(texts: list[str], model: str = "text-embedding-3-small", api_key: Optional[str] = None) -> list[list[float]]:
    """Get embeddings from OpenAI."""
    import openai
    client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np
    a_np = np.array(a)
    b_np = np.array(b)
    denom = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    if denom == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / denom)

