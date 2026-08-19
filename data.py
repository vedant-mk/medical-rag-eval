"""Load and sample PubMedQA (pqa_labeled split)."""

from datasets import load_dataset


def load_pubmedqa(sample_size: int = 200, seed: int = 42) -> list[dict]:
    """Load PubMedQA pqa_labeled and return a deterministic sample.

    Each record contains:
        - pubid: the PubMedQA ID
        - question: the yes/no/maybe question
        - contexts: list of context strings (the passage segments)
        - gold_answer: one of "yes", "no", "maybe"
        - long_answer: the free-text gold explanation
    """
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(sample_size, len(ds))))

    records = []
    for row in ds:
        contexts = row["context"]["contexts"]
        records.append({
            "pubid": row["pubid"],
            "question": row["question"],
            "contexts": contexts,
            "gold_answer": row["final_decision"],
            "long_answer": row["long_answer"],
        })
    return records


def build_corpus(records: list[dict]) -> list[dict]:
    """Flatten all context passages into a corpus for indexing.

    Each corpus entry contains:
        - text: the passage text
        - pubid: source question ID
        - ctx_idx: index within that question's contexts
    """
    corpus = []
    for rec in records:
        for i, ctx in enumerate(rec["contexts"]):
            corpus.append({
                "text": ctx,
                "pubid": rec["pubid"],
                "ctx_idx": i,
            })
    return corpus
