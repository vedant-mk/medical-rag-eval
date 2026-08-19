# Failure-Mode Analysis

Analysis of 15 representative errors from the dense RAG system (68.50% accuracy on 200 PubMedQA questions). Dense RAG outperformed hybrid on this dataset, so we analyse its failures.

## Error Distribution

Of 63 total errors (200 questions), we categorise 15 below into four failure modes:

| Category | Count (in 15) | Description |
|---|---|---|
| 1. Retrieval miss | 1 | Gold context not in top-5 |
| 2. Retrieval hit, wrong generation | 7 | Context present, LLM misread or ignored it |
| 3. Synthesis-requiring question | 3 | Local retrieval structurally insufficient |
| 4. Ambiguous gold / label noise | 4 | "Maybe" label that reasonable models disagree on |

The system's retrieval hit-rate is 99.5% — almost perfect. The bottleneck is overwhelmingly in generation, not retrieval. The engine (Llama 3.1 8B) struggles most with "no" answers (63.9% accuracy) and "maybe" answers (4.0% accuracy), while handling "yes" answers well (87.4%). This confidence bias — the model defaults to affirmative answers — is the single largest source of error.

---

## Category 1: Retrieval Miss

### Error 1 — PubID 11570976
**Question:** Is eligibility for a chemotherapy protocol a good prognostic factor for invasive bladder cancer after radical cystectomy?
**Gold:** yes | **Predicted:** no | **Retrieval hit:** False

The gold context for this question was not retrieved in the top-5, despite the corpus containing it. The dense embedder likely failed to associate "eligibility for a protocol" with the specific prognostic outcome described in the passage. This is a semantic gap: the question asks about a proxy variable (protocol eligibility), while the context discusses survival outcomes. A lexical retriever might have caught the keyword overlap, but the hybrid system's RRF fusion did not rescue this case either.

**What would fix it:** Query expansion or a cross-encoder reranker that scores (query, passage) pairs jointly rather than relying on independent embeddings.

---

## Category 2: Retrieval Hit, Wrong Generation

### Error 2 — PubID 9582182
**Question:** Does the SCL 90-R obsessive-compulsive dimension identify cognitive impairments?
**Gold:** yes | **Predicted:** no

The relevant context was retrieved (hit = True), but the model answered "no." The passage describes a correlation between SCL 90-R scores and cognitive test performance, which supports "yes." The LLM likely focused on hedging language in the abstract ("moderate correlation," "further research needed") and interpreted it as insufficient evidence. This is the hallucination-over-context failure: the model's prior (cautious = no) overrode what the context actually said.

**What would fix it:** A stronger instruction-following model, or chain-of-thought prompting that forces the model to quote evidence before answering.

### Error 3 — PubID 12970636
**Question:** Does early discharge with nurse home visits affect adequacy of newborn metabolic screening?
**Gold:** no | **Predicted:** yes

Context was present and describes a study finding no significant difference in screening adequacy. The model answered "yes" — likely because the question asks whether something "affects" adequacy, and the model's prior is that interventions have effects. This is the affirmation bias: the model defaults to "yes" when a question asks "does X affect Y?"

**What would fix it:** Better calibration of the generation model, or a prompt that explicitly instructs: "Answer 'no' if the evidence shows no significant effect."

### Error 4 — PubID 23719685
**Question:** Does high-dose radiotherapy benefit palliative lung cancer patients?
**Gold:** no | **Predicted:** yes

The context discusses a study finding no survival benefit for high-dose palliative radiotherapy. The model answered "yes," again showing affirmation bias. The word "benefit" in the question may have primed the model toward an affirmative response despite the negative evidence in the context.

**What would fix it:** Same as above — the model needs to attend to the actual evidence direction, not the question framing.

### Error 5 — PubID 15112004
**Question:** Are WHO/UNAIDS/UNICEF-recommended replacement milks for infants of HIV-infected mothers appropriate in the South African context?
**Gold:** no | **Predicted:** yes

The context discusses cost and nutritional barriers making replacement milks inappropriate in the South African context. The model likely anchored on the authority of WHO/UNAIDS/UNICEF recommendations and defaulted to "yes." This is a case where the model's parametric knowledge (these organisations make good recommendations) overrode the passage-specific evidence.

**What would fix it:** A retrieval-augmented system that explicitly deprioritises parametric knowledge when context is provided. Instruction tuning or system prompts emphasising "answer based only on the provided context."

### Error 6 — PubID 27643961
**Question:** Major depression and alcohol use disorder in adolescence: Does comorbidity lead to poorer outcomes of depression?
**Gold:** no | **Predicted:** yes

Intuitively, comorbidity leading to worse outcomes seems obvious — but the study found no additional impact. The model's medical prior (comorbidity = worse) overrode the specific finding. This is parametric-vs-contextual conflict.

**What would fix it:** Stronger grounding in the provided context. Retrieval-augmented generation works only when the model actually defers to the retrieved evidence.

### Error 7 — PubID 18926458
**Question:** Are octogenarians at high risk for carotid endarterectomy?
**Gold:** no | **Predicted:** yes

The study found acceptable complication rates in octogenarians. The model's prior that elderly patients are high-risk overrode the evidence. Another parametric-vs-contextual failure.

**What would fix it:** Same pattern — the engine needs better context-grounding.

### Error 8 — PubID 11977907
**Question:** Subclavian steal syndrome: can the blood pressure difference between arms predict the severity of steal?
**Gold:** yes | **Predicted:** no

Here the model erred toward the negative. The context supports the prediction, but the model may have been confused by the clinical specificity of the question. This is less a systematic bias and more a comprehension failure on a technically dense passage.

**What would fix it:** A larger or domain-specialised model (e.g., Med-PaLM) that better handles clinical reasoning.

---

## Category 3: Synthesis-Requiring Question

### Error 9 — PubID 10430303
**Question:** Does laparoscopic cholecystectomy influence peri-sinusoidal cell activity?
**Gold:** yes | **Predicted:** maybe

This question requires synthesising information across multiple experimental observations in the passage to conclude that the procedure does affect cell activity. The model hedged with "maybe" because no single sentence states the conclusion directly. This is the local/global failure: our retriever finds relevant passages (a local operation), but the question requires synthesising across them (a global operation).

**What would fix it:** Graph-based indexing (as in MedGraphRAG) that pre-computes relationships across passages at index time, or a multi-hop retrieval strategy.

### Error 10 — PubID 23571528
**Question:** Sternal skin conductance: a reasonable surrogate for hot flash measurement?
**Gold:** no | **Predicted:** maybe

The passage presents mixed evidence requiring synthesis of correlation data, practical limitations, and comparison with established methods. The model's "maybe" reflects genuine uncertainty, but the gold label is "no" — the study concluded it is not a reasonable surrogate. Answering correctly requires weighing multiple pieces of evidence, not just finding a single supporting passage.

**What would fix it:** Chain-of-thought reasoning or a system that explicitly aggregates evidence pro and con before deciding.

### Error 11 — PubID 24139705
**Question:** Telemedicine and type 1 diabetes: is technology per se sufficient to improve glycaemic control?
**Gold:** yes | **Predicted:** no

This question requires understanding a nuanced argument: the study found that technology alone (without additional clinical contact) was sufficient. The model, perhaps expecting that technology needs human support, answered "no." Correctly answering requires synthesising the study design with its outcome — the finding is counterintuitive.

**What would fix it:** A system that can handle counterintuitive findings, perhaps through explicit evidence extraction before answer generation.

---

## Category 4: Ambiguous Gold / Label Noise

### Error 12 — PubID 25636371
**Question:** Is it possible to stop treatment with nucleos(t)ide analogs in patients with e-antigen negative chronic hepatitis B?
**Gold:** maybe | **Predicted:** yes

The question asks about possibility, and some evidence supports stopping treatment under certain conditions. "Maybe" and "yes, conditionally" are arguably equivalent here. The PubMedQA label reflects the paper's cautious framing, but a model that answers "yes" is not unreasonable. This is label noise, not a system failure.

### Error 13 — PubID 25987398
**Question:** The influence of atmospheric pressure on aortic aneurysm rupture — is the diameter of the aneurysm important?
**Gold:** maybe | **Predicted:** yes

The study found some evidence that diameter interacts with atmospheric pressure effects. "Maybe" reflects the paper's equivocal conclusion, but "yes" is a defensible reading of the data. Again, the model and gold disagree on the strength of evidence, not its direction.

### Error 14 — PubID 20101129
**Question:** Is prophylactic fixation a cost-effective method to prevent a future contralateral fragility hip fracture?
**Gold:** maybe | **Predicted:** yes

The cost-effectiveness analysis had mixed results depending on assumptions. "Maybe" is the honest answer; "yes" reflects one reading of the data. The model systematically fails to output "maybe" (only 4% accuracy on maybe-gold questions), which is its core calibration weakness.

### Error 15 — PubID 16816043
**Question:** Do French lay people and health professionals find it acceptable to breach confidentiality to protect a patient's wife from a sexually transmitted disease?
**Gold:** maybe | **Predicted:** yes

The study found mixed attitudes across groups. "Maybe" captures this heterogeneity. The model, forced to commit, chose "yes" — reasonable but not what the label says.

---

## Summary

The dominant failure mode is **generation error with context present** (62/63 errors had retrieval hits). The retriever works. The problem is downstream:

1. **Affirmation bias** — the model defaults to "yes" (36/63 error predictions are "yes"), particularly failing on "no" and "maybe" gold answers.
2. **"Maybe" blindness** — 96% of "maybe"-gold questions are answered incorrectly. The model almost never predicts "maybe," treating it as not a real answer category.
3. **Parametric override** — when the retrieved context contradicts the model's medical priors, the model often trusts its training data over the passage.
4. **Synthesis failure** — questions requiring integration across multiple evidence points in a passage expose the local-retrieval, single-shot-generation architecture.

In the local/global framing from the companion term paper: this system's recipe (embed → retrieve → generate) handles the *find* operation well (99.5% hit-rate) but fails at *synthesise*. Improving the engine (larger model, domain-specific fine-tuning) would address categories 1-3. Category 4 requires changing the recipe — moving from local retrieval to graph-based or multi-hop architectures that pre-compute relationships at indexing time, as Medical Graph RAG does.
