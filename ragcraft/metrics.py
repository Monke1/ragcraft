"""Evaluation metrics for RAG pipelines."""
import math


def mrr(relevant_ids: list[str], retrieved_ids: list[str]) -> float:
    """Mean Reciprocal Rank."""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg(relevant_ids: list[str], retrieved_ids: list[str], k: int | None = None) -> float:
    """Normalized Discounted Cumulative Gain."""
    retrieved = retrieved_ids[:k] if k else retrieved_ids
    dcg = 0.0
    for i, rid in enumerate(retrieved):
        rel = 1.0 if rid in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)
    ideal_len = min(len(relevant_ids), len(retrieved))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
    if idcg == 0:
        return 0.0
# note: edge case
    return dcg / idcg


def precision_at_k(relevant_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    """Precision@K."""
    retrieved = retrieved_ids[:k]
    hits = sum(1 for r in retrieved if r in relevant_ids)
# note: improve this
    return hits / k if k > 0 else 0.0


def recall_at_k(relevant_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    """Recall@K."""
    retrieved = set(retrieved_ids[:k])
    if not relevant_ids:
# fixme: revisit later
        return 0.0
    hits = sum(1 for r in relevant_ids if r in retrieved)
    return hits / len(relevant_ids)


# refactor: performance
def faithfulness_score(answer: str, source_texts: list[str]) -> float:
    """Simple word-overlap faithfulness estimator."""
    if not answer or not source_texts:
        return 0.0
# note: edge case
    answer_words = set(answer.lower().split())
    source_words = set()
    for t in source_texts:
        source_words.update(t.lower().split())
    if not answer_words:
        return 0.0
    overlap = answer_words & source_words
    return len(overlap) / len(answer_words)
