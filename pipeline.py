"""End-to-end RAG pipeline orchestrator."""

import json
import os
from pathlib import Path
from tqdm import tqdm

from src.data import load_pubmedqa, build_corpus
from src.retrieve import DenseRetriever, BM25Retriever, HybridRetriever
from src.generate import build_rag_prompt, build_blind_prompt, call_groq, parse_answer
from src.evaluate import check_retrieval_hit, summarize_results

RESULTS_DIR = Path("results")
RAW_DIR = RESULTS_DIR / "raw_outputs"


def run_baseline(records: list[dict]) -> list[dict]:
    """Phase 1a: no-retrieval baseline — LLM answers blind."""
    results = []
    for rec in tqdm(records, desc="Baseline (no retrieval)"):
        messages = build_blind_prompt(rec["question"])
        raw = call_groq(messages)
        predicted = parse_answer(raw)
        results.append({
            "pubid": rec["pubid"],
            "question": rec["question"],
            "gold_answer": rec["gold_answer"],
            "predicted": predicted,
            "raw_response": raw,
            "retrieved_docs": [],
            "retrieval_hit": False,
        })
    return results


def run_rag(records: list[dict], retriever, label: str = "rag") -> list[dict]:
    """Run RAG with a given retriever (dense or hybrid)."""
    results = []
    for rec in tqdm(records, desc=f"RAG ({label})"):
        retrieved = retriever.search(rec["question"], top_k=5)
        contexts = [doc["text"] for doc in retrieved]
        messages = build_rag_prompt(rec["question"], contexts)
        raw = call_groq(messages)
        predicted = parse_answer(raw)
        hit = check_retrieval_hit(retrieved, rec["pubid"])
        results.append({
            "pubid": rec["pubid"],
            "question": rec["question"],
            "gold_answer": rec["gold_answer"],
            "predicted": predicted,
            "raw_response": raw,
            "retrieved_docs": retrieved,
            "retrieval_hit": hit,
        })
    return results


def save_results(all_results: dict, path: Path = RESULTS_DIR / "results.json"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


def save_raw_outputs(results: list[dict], name: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / f"{name}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


def run_full_pipeline(sample_size: int = 200):
    """Run all phases and return structured results."""
    print("Loading PubMedQA...")
    records = load_pubmedqa(sample_size=sample_size)
    print(f"Loaded {len(records)} questions.")

    print("\nBuilding corpus...")
    corpus = build_corpus(records)
    print(f"Corpus size: {len(corpus)} passages.")

    print("\nBuilding dense index...")
    dense = DenseRetriever(corpus)

    print("\nBuilding BM25 index...")
    bm25 = BM25Retriever(corpus)

    print("\nBuilding hybrid retriever...")
    hybrid = HybridRetriever(dense, bm25)

    # Phase 1a: Baseline
    print("\n--- Phase 1a: No-retrieval baseline ---")
    baseline_results = run_baseline(records)
    save_raw_outputs(baseline_results, "baseline")

    # Phase 1b: Dense RAG
    print("\n--- Phase 1b: Dense RAG ---")
    dense_results = run_rag(records, dense, label="dense")
    save_raw_outputs(dense_results, "dense_rag")

    # Phase 2: Hybrid RAG
    print("\n--- Phase 2: Hybrid RAG ---")
    hybrid_results = run_rag(records, hybrid, label="hybrid")
    save_raw_outputs(hybrid_results, "hybrid_rag")

    # Summarize
    summary = {
        "sample_size": sample_size,
        "corpus_size": len(corpus),
        "baseline": summarize_results(baseline_results, "No retrieval"),
        "dense_rag": summarize_results(dense_results, "Dense RAG"),
        "hybrid_rag": summarize_results(hybrid_results, "Hybrid RAG"),
    }
    save_results(summary)
    print("\n=== Results ===")
    for key in ["baseline", "dense_rag", "hybrid_rag"]:
        s = summary[key]
        print(f"  {s['label']:20s}  Accuracy: {s['accuracy']:.2%}  Hit-rate: {s.get('hit_rate', 'N/A')}")

    return summary, baseline_results, dense_results, hybrid_results
