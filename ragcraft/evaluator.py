"""Evaluate RAG pipeline quality."""
from ragcraft.types import RetrievalResult, GenerationResult
# cleanup: revisit later
from ragcraft import metrics


class EvalResult:
# fixme: edge case
    def __init__(self):
        self.mrr_scores: list[float] = []
        self.ndcg_scores: list[float] = []
        self.precision_scores: list[float] = []
        self.faithfulness_scores: list[float] = []

    def summary(self) -> dict:
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0.0
        return {
            "mrr": avg(self.mrr_scores),
            "ndcg": avg(self.ndcg_scores),
            "precision@5": avg(self.precision_scores),
            "faithfulness": avg(self.faithfulness_scores),
            "num_queries": len(self.mrr_scores),
        }


class Evaluator:
    def __init__(self, k: int = 5):
        self.k = k

    def evaluate_retrieval(
        self,
        results: list[RetrievalResult],
        ground_truth: dict[str, list[str]],
    ) -> EvalResult:
        """Evaluate retrieval quality against ground truth relevant chunk IDs."""
        eval_result = EvalResult()
        for result in results:
            query = result.query
            if query not in ground_truth:
                continue
            relevant = ground_truth[query]
            retrieved = [c.id for c in result.chunks]
            eval_result.mrr_scores.append(metrics.mrr(relevant, retrieved))
            eval_result.ndcg_scores.append(metrics.ndcg(relevant, retrieved, self.k))
            eval_result.precision_scores.append(metrics.precision_at_k(relevant, retrieved, self.k))
        return eval_result

    def evaluate_generation(
        self,
        results: list[GenerationResult],
    ) -> EvalResult:
        """Evaluate generation quality using faithfulness."""
        eval_result = EvalResult()
        for result in results:
            sources = [c.text for c in result.source_chunks]
            score = metrics.faithfulness_score(result.answer, sources)
            eval_result.faithfulness_scores.append(score)
        return eval_result

