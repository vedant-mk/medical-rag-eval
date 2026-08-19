# Medical RAG Evaluation

An empirical study of retrieval-augmented generation on medical question-answering, comparing vanilla dense retrieval against a hybrid BM25+dense variant on PubMedQA, with explicit failure-mode analysis.

## Problem

A patient with COPD and heart failure asks their doctor about bronchodilator therapy. A naive retrieval system finds passages about COPD treatment and confidently recommends a standard bronchodilator — missing that the same drug class can worsen heart failure. The system *found* the right topic but failed to *synthesise* across the comorbidity. This is not a retrieval failure; it is an architecture failure. The system was never designed to reason across multiple evidence threads.

Medical RAG matters because clinical decisions require both finding the right evidence and integrating it correctly. This project measures where a standard RAG pipeline succeeds and where — predictably — it breaks.

## Analytical Frame

This study applies the **local/global** distinction as its analytical lens: *finding* a relevant passage (local retrieval) versus *synthesising* across multiple passages or reasoning about relationships between them (global reasoning). A RAG system is the product of three factors — the **engine** (the LLM), the **recipe** (how retrieval and generation are wired together), and the **corpus** (what is being searched). This project changes the recipe (dense vs hybrid retrieval) while holding the engine (Llama 3.1 8B) and corpus (PubMedQA) fixed. When results change between configurations, we can name exactly which knob turned.

This framing comes from my Master's seminar term paper on Medical Graph RAG, where I analyse how different systems (GraphRAG, MedGraphRAG, MedRAG) encode different things in their graphs — themes, evidence chains, or diagnostic differences — each driven by what failure mode the system most fears.

## Methodology

**Corpus:** PubMedQA `pqa_labeled` — 1,000 expert-annotated biomedical questions with yes/no/maybe gold answers. We sample 200 questions (fixed seed=42) as a deliberate trade-off: large enough to argue from, small enough to run on a free API tier.

**Three phases:**

1. **Baseline + Dense RAG.** Embed all 676 context passages with `all-MiniLM-L6-v2`, build a FAISS index, and for each question retrieve top-5 by cosine similarity. Assemble a prompt with the retrieved contexts and call Llama 3.1 8B (via Groq free tier) to produce a yes/no/maybe answer. Also run a no-retrieval baseline where the LLM answers blind. The recipe here is the simplest possible: embed, retrieve, generate. No reranking, no query rewriting, no chain-of-thought — deliberately, so that failures are attributable to the architecture, not to missing optimisations.

2. **Hybrid retrieval.** Add BM25 scoring alongside dense embeddings. Fuse via reciprocal rank fusion (RRF, k=60). Re-run the same 200 questions. The hypothesis: BM25 catches lexical matches that dense embeddings miss (medical acronyms, drug names, exact phrases). We are turning one knob — the recipe's retrieval component — while holding engine and corpus fixed.

3. **Failure-mode analysis.** Take 15 questions the system got wrong and categorise each into one of four failure modes:
   - **Retrieval miss** — gold context not in top-5
   - **Retrieval hit, wrong generation** — context was there, the LLM ignored or misread it
   - **Synthesis-requiring** — the question structurally needs integration across passages; local retrieval is insufficient
   - **Ambiguous gold label** — the answer key is debatable; we acknowledge this honestly rather than pretending it away

Note that traceability (being able to point to which passages the answer came from) is a separate property from correctness. Our system is fully auditable — every answer links back to its retrieved passages — but *auditable does not mean correct*. The failure analysis demonstrates this distinction concretely.

## Results

| Configuration | Accuracy | Retrieval Hit-Rate |
|---|---|---|
| No retrieval (baseline) | 51.50% | — |
| Dense RAG (FAISS, top-5) | **68.50%** | **99.50%** |
| Hybrid RAG (BM25 + dense, RRF) | 67.50% | 98.00% |

**Key findings:**

- RAG improves accuracy by 17 percentage points over the blind baseline. The engine is the same; the recipe (adding retrieval) is what changed.
- Dense retrieval already saturates hit-rate at 99.5% on PubMedQA. This dataset's contexts are short, pre-chunked passages that embed well. BM25 fusion *hurts* slightly — it introduces noise from lexical false positives without catching meaningful misses.
- The bottleneck is not retrieval. It is generation. Of 63 errors in the dense system, 62 had the gold context in the top-5. The model had the answer in front of it and still got it wrong.
- Accuracy by gold answer type reveals a severe calibration problem: 87.4% on "yes" questions, 63.9% on "no" questions, and **4.0% on "maybe" questions**. The model almost never predicts "maybe."

## Failure-Mode Breakdown

Of 15 analysed errors (full details in [`failures.md`](failures.md)):

| Failure Mode | Count | Key Pattern |
|---|---|---|
| Retrieval miss | 1 | Very short query carries too little signal to match |
| Retrieval hit, wrong generation | 7 | Affirmation bias; parametric knowledge overriding context |
| Synthesis-requiring | 3 | Multi-evidence integration needed; local retrieval insufficient |
| Ambiguous gold label | 4 | "Maybe" label on questions where "yes" is defensible |

**Example: Retrieval hit, wrong generation** (PubID 23719685). "Does high-dose radiotherapy benefit palliative lung cancer patients?" Gold: no. Predicted: yes. The retrieved context explicitly describes no survival benefit, but the model's affirmation bias — primed by the word "benefit" in the question — overrode the evidence. The system is auditable (we can see it retrieved the right passage) but incorrect (it ignored what the passage said).

**Example: Synthesis-requiring** (PubID 10430303). "Does laparoscopic cholecystectomy influence peri-sinusoidal cell activity?" Gold: yes. Predicted: maybe. No single sentence states the conclusion; it requires synthesising across multiple experimental observations. This is the local/global failure in the wild: our retriever finds relevant passages (local), but the question demands reasoning across them (global). This is exactly the failure mode that graph-based indexing architectures like MedGraphRAG are designed to address — by moving synthesis to indexing time rather than leaving it to the generator.

**Example: Ambiguous gold** (PubID 25636371). "Is it possible to stop treatment with nucleos(t)ide analogs?" Gold: maybe. Predicted: yes. The evidence supports conditional cessation. "Maybe" and "yes, with caveats" are arguably equivalent. This is label noise, not a system failure — and acknowledging it honestly is part of rigorous evaluation.

## What I'd Do Next

Three specific improvements, not implemented here:

1. **Cross-encoder reranking.** Replace the first-stage retriever's cosine scores with a cross-encoder (e.g., `ms-marco-MiniLM-L-6-v2`) that scores (query, passage) pairs jointly. This directly addresses the one retrieval miss and may improve borderline retrievals. Cost: ~2x latency, no additional API calls.

2. **Chain-of-thought prompting with evidence extraction.** Instead of asking for a one-word answer, prompt the model to first quote the key evidence from the retrieved context, then reason about it, then answer. This forces the model to attend to the context rather than defaulting to parametric priors. It directly addresses the "retrieval hit, wrong generation" category (62/63 errors).

3. **Graph-based indexing (GraphRAG-style).** At indexing time, build a graph of entity relationships across the corpus. At query time, traverse the graph to retrieve not just relevant passages but the *relationships between them*. This addresses the synthesis-requiring failure mode — the system's architecture would structurally support multi-hop reasoning rather than relying on the generator to do it from flat context. I analysed this approach in my seminar term paper but did not implement it here; the conceptual depth is in the paper, the empirical measurement is in this project.

## Companion Term Paper

This project is the empirical companion to my Master's seminar term paper on Medical Graph RAG; the analytical framework used here (local/global, engine/recipe/corpus, failure-mode taxonomy) is developed there.

**Paper:** *What Should the Graph Encode? An Analysis of Medical Graph RAG and Its Neighbours*
Vedant Kulkarni, University of Augsburg, 2026.

The PDF will be added to this repository as `paper.pdf` once reviewed.

## Reproduce

```bash
git clone https://github.com/vedant-mk/medical-rag-eval.git
cd medical-rag-eval
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
```

Then either run the notebook:
```bash
jupyter notebook 01_full_eval.ipynb
```

Or run programmatically:
```python
from pipeline import run_full_pipeline
summary, baseline, dense, hybrid = run_full_pipeline(sample_size=200)
```

**Requirements:** Python 3.11+, ~2GB disk for sentence-transformers model, Groq API key (free tier). Runs on M1 MacBook Air 8GB. The full pipeline (600 API calls) takes ~45 minutes on Groq free tier due to rate limits.

## Stack

- `sentence-transformers` / `all-MiniLM-L6-v2` — embeddings (local, free)
- `faiss-cpu` — dense vector index
- `rank-bm25` — lexical retrieval for hybrid variant
- Groq API / Llama 3.1 8B Instant — generation (free tier)
- `datasets` — PubMedQA from HuggingFace

## License

MIT
