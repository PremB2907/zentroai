# ZentroAI Phase 4.6: Human Review of Local Ollama Pilot

**Evaluation run**: `eval_rag_questions_20261026T172851Z.csv` (live, Ollama, `gemma2:2b`, 5 questions)
**Date**: 26 October 2026
**Reviewer**: Human evaluator

---

## Executive Summary

| Metric | Result |
|--------|--------|
| Questions processed | 5 |
| API errors | 0 |
| Average latency | 1.583 s |
| `must_include` PASS | 3/5 |
| `must_include` FAIL | 2/5 |
| Retrieval PASS | 1/5 |
| Retrieval PARTIAL | 3/5 |
| Retrieval N/A | 1/5 |

**Overall**: The local Ollama pilot works technically (no API errors, reasonable latency). However, **2 of 5 answers are substantively incorrect** — both are warranty questions where the model incorrectly claims insufficient information despite the evidence being present in retrieved sections and the product catalog. Retrieval quality is mixed (only 1/5 PASS).

---

## Question-by-Question Review

---

### PQ0002 — SKU ELE-0012 pe ZentroMart Assured hai kya? Claim kaise karein?

**Question** (Hinglish): "SKU ELE-0012 pe ZentroMart Assured hai kya? Claim kaise karein?"  
*("Does SKU ELE-0012 have ZentroMart Assured? How to claim?")*

| Field | Value |
|-------|-------|
| **Intent** | warranty |
| **Language** | hi-en |
| **Answerable** | True |
| **Gold evidence** | WAR-2 ; products.csv:ELE-0012 |
| **Retrieved sections** | POL-WAR ; WAR-2 ; MKS-3 ; HLP-TRB ; DWM-4 |
| **Retrieval recall** | PARTIAL |
| **Model response** | "I'm sorry, I don't have enough information to answer that. I can connect you to a human colleague who can help." |
| **Provider** | ollama |
| **Latency** | 1.685 s |
| **must_include** | FAIL (expected: "12" or "My Orders / app") |
| **must_not_include** | N/A |
| **is_api_error** | False |

**Evidence check**:
- **POL-WAR / WAR-2** (retrieved): "Products marked `assured = true` in the catalog carry a 12-month ZentroMart Assured warranty... Claims are raised in the app under My Orders..."
- **products.csv ELE-0012**: Kestrel K7 Smartphone 128 GB, Beige — `assured=True`, `warranty_months=12`

**Agreement with evidence**: The retrieved evidence **contains the exact answer** (WAR-2 states the policy; products.csv confirms ELE-0012 is assured). The model incorrectly claims insufficient information.

**Why must_include FAILed**: The response contains none of the required keywords ("12", "My Orders", "app"). It is a refusal, not an answer.

**Retrieval missed expected evidence**: Yes — `products.csv:ELE-0012` (the SKU-specific catalog row) was not retrieved. However, WAR-2 (the general policy) was retrieved and is sufficient when combined with the product catalog (which the agent has access to via the retriever's product lookup). The retriever should have included the product row.

**Human judgment**: **INCORRECT**  
The model has the policy (WAR-2) in context and the product catalog is accessible. It should have answered: "Haan, ELE-0012 (Kestrel K7 Smartphone 128 GB, Beige) pe 12 mahine ki ZentroMart Assured warranty hai. Claim app mein My Orders se karein." The refusal is a model reasoning failure, not a retrieval failure.

---

### PQ0003 — How fast are refunds to the ZentroMart Wallet?

**Question**: "How fast are refunds to the ZentroMart Wallet?"

| Field | Value |
|-------|-------|
| **Intent** | refund_timeline |
| **Language** | en |
| **Answerable** | True |
| **Gold evidence** | REF-2 ; KB-00.1 |
| **Retrieved sections** | PAY-4 ; REF-2 ; MKT-1 ; REF-3 ; MKS-2 |
| **Retrieval recall** | PARTIAL |
| **Model response** | "Refunds to the ZentroMart Wallet are instant, within 2 hours. (REF-2)" |
| **Provider** | ollama |
| **Latency** | 1.301 s |
| **must_include** | PASS (contains "instant" and "2 hours") |
| **must_not_include** | PASS (does not contain "5 to 7 days") |
| **is_api_error** | False |

**Evidence check**:
- **POL-REF / REF-2** (retrieved): "ZentroMart Wallet | Instant, within 2 hours"
- **KB-00.1** (gold, not retrieved): The 00_about_this_knowledge_base.md section 1 — not directly relevant to refund timing.

**Agreement with evidence**: **Yes**. The answer correctly states "instant, within 2 hours" per REF-2.

**Why must_include PASSed**: Response contains "instant" and "2 hours" (both match the keyword list).

**Retrieval missed expected evidence**: Partially — KB-00.1 was not retrieved, but REF-2 was retrieved and contains the complete answer. The missing KB-00.1 is metadata about the KB itself, not policy content.

**Human judgment**: **CORRECT**  
Accurate, concise, cites the correct section (REF-2). No defect.

---

### PQ0004 — Can I schedule a delivery at midnight?

**Question**: "Can I schedule a delivery at midnight?"

| Field | Value |
|-------|-------|
| **Intent** | unsupported |
| **Language** | en |
| **Answerable** | False |
| **Gold evidence** | (empty) |
| **Retrieved sections** | FAQ-3 ; DEL-2 ; REF-5 ; POL-DEL ; POL-PAY |
| **Retrieval recall** | N/A |
| **Model response** | "I'm sorry, I don't have enough information to answer that. I can connect you to a human colleague who can help. [POL-DEL]" |
| **Provider** | ollama |
| **Latency** | 1.545 s |
| **must_include** | PASS (contains "don't have" and "connect") |
| **must_not_include** | N/A |
| **is_api_error** | False |

**Evidence check**:
- **POL-DEL** (retrieved): Covers delivery times (standard/express), fees, attempts, late deliveries, serviceable areas, address changes. **No mention of midnight scheduling.**
- **FAQ-3**: "Delivery is free on most orders." — irrelevant.

**Agreement with evidence**: **Yes**. The policy does not mention midnight delivery scheduling. The correct behavior is to say "I don't have information" and offer human handoff.

**Why must_include PASSed**: Response contains "don't have" and "connect" (both in the keyword list for unanswerable questions).

**Retrieval missed expected evidence**: N/A — no gold evidence expected.

**Human judgment**: **CORRECT**  
Appropriate refusal with human handoff offer. The `[POL-DEL]` citation is slightly odd (the policy doesn't address midnight scheduling), but the answer itself is correct.

---

### PQ0005 — SKU ELE-0002 pe ZentroMart Assured hai kya? Claim kaise karein?

**Question** (Hinglish): "SKU ELE-0002 pe ZentroMart Assured hai kya? Claim kaise karein?"  
*("Does SKU ELE-0002 have ZentroMart Assured? How to claim?")*

| Field | Value |
|-------|-------|
| **Intent** | warranty |
| **Language** | hi-en |
| **Answerable** | True |
| **Gold evidence** | WAR-2 ; products.csv:ELE-0002 |
| **Retrieved sections** | POL-WAR ; WAR-2 ; MKS-3 ; HLP-TRB ; DWM-4 |
| **Retrieval recall** | PARTIAL |
| **Model response** | "I'm sorry, I don't have enough information to answer that. I can connect you to a human colleague who can help." |
| **Provider** | ollama |
| **Latency** | 1.639 s |
| **must_include** | FAIL (expected: "12" or "My Orders / app") |
| **must_not_include** | N/A |
| **is_api_error** | False |

**Evidence check**:
- **POL-WAR / WAR-2** (retrieved): Same as PQ0002 — policy states assured products have 12-month warranty, claims via My Orders in app.
- **products.csv ELE-0002**: Nimbus X2 Wireless Earbuds, Black — `assured=True`, `warranty_months=12`

**Agreement with evidence**: The retrieved evidence **contains the exact answer** (WAR-2 + product catalog). The model incorrectly claims insufficient information.

**Why must_include FAILed**: The response contains none of the required keywords ("12", "My Orders", "app"). It is a refusal, not an answer.

**Retrieval missed expected evidence**: Yes — `products.csv:ELE-0002` was not retrieved. Same issue as PQ0002.

**Human judgment**: **INCORRECT**  
Identical failure mode to PQ0002. The model has the policy (WAR-2) and should know from the product catalog that ELE-0002 is assured. The refusal is a model reasoning failure. The answer should be: "Haan, ELE-0002 (Nimbus X2 Wireless Earbuds, Black) pe 12 mahine ki ZentroMart Assured warranty hai. Claim app mein My Orders se karein."

---

### PQ0006 — As a ZentroMart Plus member, will I be charged the pickup fee for returning shoes I just don't like?

**Question**: "As a ZentroMart Plus member, will I be charged the pickup fee for returning shoes I just don't like?"

| Field | Value |
|-------|-------|
| **Intent** | return_fee |
| **Language** | en |
| **Answerable** | True |
| **Gold evidence** | RET-6 ; PLS-2 |
| **Retrieved sections** | RET-6 ; PLS-2 ; DEL-2 ; DEL-4 ; RET-7 |
| **Retrieval recall** | PASS |
| **Model response** | "No, as a ZentroMart Plus member, you will not be charged the reverse pickup fee for returns of clothing, footwear, and bags. (per [PLS-2])" |
| **Provider** | ollama |
| **Latency** | 1.746 s |
| **must_include** | PASS (contains "no", "not", "waived") |
| **must_not_include** | PASS (does not contain "₹49 will be charged") |
| **is_api_error** | False |

**Evidence check**:
- **POL-RET / RET-6** (retrieved): "A reverse pickup fee of ₹49 applies to change-of-mind returns of clothing, footwear and bags. The fee is waived for ZentroMart Plus members..."
- **POL-PLS / PLS-2** (retrieved): "No reverse pickup fee on any return."

**Agreement with evidence**: **Yes**. The answer correctly states no fee for Plus members, cites PLS-2.

**Why must_include PASSed**: Response contains "no" and "not" (matches "no / not / waived / free" keyword list).

**Why must_not_include PASSed**: Response does not contain the banned phrase "₹49 will be charged".

**Retrieval missed expected evidence**: No — both RET-6 and PLS-2 were retrieved (PASS).

**Human judgment**: **CORRECT**  
Accurate, well-cited, correctly applies Plus benefit to footwear (shoes). No defect.

---

## Cross-Cutting Observations

### 1. Warranty Questions (PQ0002, PQ0005) — Systematic Failure
Both warranty questions failed identically. The model received:
- WAR-2 (policy: assured products → 12-month warranty, claim via My Orders)
- But **not** the product catalog row (`products.csv:ELE-0012` / `products.csv:ELE-0002`)

The retriever uses BM25 on KB documents only. The product catalog is a separate CSV that the `retrieve()` function *should* join when a SKU is mentioned. The fact that both product rows were missed suggests the SKU-aware retrieval is not triggering or the product rows are not being indexed as retrievable chunks.

**This is a retrieval gap, not a model defect per se** — but the model should not refuse when it has the general policy (WAR-2) and could reasonably infer the answer if it knew the SKU is assured. The system prompt says "Answer ONLY from the EVIDENCE block", so the model is technically following instructions by refusing when the specific SKU evidence isn't in the evidence block.

### 2. Retrieval Quality
| Question | Retrieval Recall | Missing Gold Evidence |
|----------|------------------|----------------------|
| PQ0002 | PARTIAL | products.csv:ELE-0012 |
| PQ0003 | PARTIAL | KB-00.1 (metadata, not critical) |
| PQ0004 | N/A | (none expected) |
| PQ0005 | PARTIAL | products.csv:ELE-0002 |
| PQ0006 | PASS | (none) |

Only 1/5 achieved full retrieval recall. The SKU-to-product-catalog join needs improvement.

### 3. Metric Limitations
- `must_include` / `must_not_include` are **keyword proxies**, not semantic correctness checks. PQ0002 and PQ0005 FAIL correctly flag the missing keywords, but the root cause is retrieval + model interaction.
- `retrieval_recall` measures section ID overlap, not whether the retrieved text actually answers the question.
- The evaluator correctly flags API errors (0 here) and does not score heuristic metrics for error rows.

### 4. Ollama / gemma2:2b Performance
- **Latency**: 1.3–1.7 s per question (CPU, local) — acceptable for development.
- **No hallucinations** observed in these 5 answers.
- **Refusals** are coherent and offer human handoff.
- **Citation format**: `[REF-2]`, `(per [PLS-2])` — mostly correct.
- **Language**: Mixed Hinglish/English handled reasonably (PQ0002, PQ0005 responses in English despite Hinglish question — minor style issue).

---

## Conclusions & Recommendations

| Area | Finding | Action |
|------|---------|--------|
| **Model** | gemma2:2b works for straightforward policy Q&A; fails on SKU-specific warranty questions due to retrieval gap. | Consider larger model (e.g., `llama3.2:3b` or `phi3:mini`) for better reasoning, but fix retrieval first. |
| **Retrieval** | SKU-to-product-catalog join missing for 2/2 warranty questions. | Fix retriever to inject product catalog rows when SKU detected in query. |
| **Evaluation** | Heuristic metrics correctly flag failures; human review needed for semantic correctness. | Continue human-in-the-loop for pilot phases. |
| **Infrastructure** | Local Ollama pilot successful — no API keys, no cloud dependency, predictable latency. | Proceed with Ollama as primary provider for development. |

---

## Final Judgments Summary

| Question | Human Judgment | Primary Issue |
|----------|----------------|---------------|
| PQ0002 | **INCORRECT** | Retrieval missed product row; model refused despite having policy |
| PQ0003 | **CORRECT** | — |
| PQ0004 | **CORRECT** | — |
| PQ0005 | **INCORRECT** | Retrieval missed product row; model refused despite having policy |
| PQ0006 | **CORRECT** | — |

**Overall**: 3/5 CORRECT, 2/5 INCORRECT. The two failures share the same root cause (SKU-specific product evidence not retrieved). This is a **retrieval-system issue**, not a fundamental model limitation.

---

*Report generated for Phase 4.6 human review. No code, prompts, or dataset modified.*