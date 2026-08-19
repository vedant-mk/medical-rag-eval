"""Retrieval module: dense, BM25, and hybrid retrievers."""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


MODEL_NAME = "all-MiniLM-L6-v2"


class DenseRetriever:
    def __init__(self, corpus: list[dict], model_name: str = MODEL_NAME, batch_size: int = 64):
        self.corpus = corpus
        self.model = SentenceTransformer(model_name)
        texts = [doc["text"] for doc in corpus]
        self.embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.corpus[idx])
            entry["score"] = float(score)
            entry["rank"] = len(results)
            results.append(entry)
        return results


class BM25Retriever:
    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        tokenized = [doc["text"].lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            entry = dict(self.corpus[idx])
            entry["score"] = float(scores[idx])
            entry["rank"] = rank
            results.append(entry)
        return results


class HybridRetriever:
    """Fuses dense and BM25 results via reciprocal rank fusion (RRF)."""

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, k: int = 60):
        self.dense = dense
        self.bm25 = bm25
        self.k = k

    def search(self, query: str, top_k: int = 5, dense_top: int = 20, bm25_top: int = 20) -> list[dict]:
        dense_results = self.dense.search(query, top_k=dense_top)
        bm25_results = self.bm25.search(query, top_k=bm25_top)

        rrf_scores: dict[int, float] = {}
        corpus_idx_map: dict[int, dict] = {}

        for result in dense_results:
            idx = self._corpus_index(result)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self.k + result["rank"] + 1)
            corpus_idx_map[idx] = result

        for result in bm25_results:
            idx = self._corpus_index(result)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self.k + result["rank"] + 1)
            if idx not in corpus_idx_map:
                corpus_idx_map[idx] = result

        sorted_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]
        results = []
        for rank, idx in enumerate(sorted_indices):
            entry = dict(corpus_idx_map[idx])
            entry["score"] = rrf_scores[idx]
            entry["rank"] = rank
            results.append(entry)
        return results

    def _corpus_index(self, result: dict) -> int:
        for i, doc in enumerate(self.dense.corpus):
            if doc["pubid"] == result["pubid"] and doc["ctx_idx"] == result["ctx_idx"]:
                return i
        return -1
