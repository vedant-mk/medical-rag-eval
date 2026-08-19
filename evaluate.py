"""Evaluation module: accuracy and retrieval hit-rate metrics."""


def compute_accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["predicted"] == r["gold_answer"])
    return correct / len(results)


def compute_hit_rate(results: list[dict]) -> float:
    """Fraction of questions where at least one gold context appeared in top-k."""
    if not results:
        return 0.0
    hits = sum(1 for r in results if r.get("retrieval_hit", False))
    return hits / len(results)


def check_retrieval_hit(retrieved_docs: list[dict], gold_pubid: str) -> bool:
    """Check if any retrieved doc comes from the same PubMedQA entry as the question."""
    return any(doc["pubid"] == gold_pubid for doc in retrieved_docs)


def summarize_results(results: list[dict], label: str) -> dict:
    return {
        "label": label,
        "n_questions": len(results),
        "accuracy": round(compute_accuracy(results), 4),
        "hit_rate": round(compute_hit_rate(results), 4),
    }
