# Phase 4.10 – Live Ollama RAG Review (Post‑Phase‑4.9)

**Evaluation run**: `eval_rag_questions_20261026T180201Z.csv` (live, Ollama `gemma2:2b`, 5 questions)  
**Date**: 26 Oct 2026  
**Reviewer**: Human evaluator  

---  

## 1. Summary of Metrics (from harness)

| Metric | Value |
|--------|-------|
| Questions | 5 |
| API errors | 0 |
| Avg latency | 3.21 s |
| `must_include` PASS | 5 / 5 |
| `must_not_include` PASS | 2 / 2 |
| Retrieval recall | 3 PASS, 1 PARTIAL, 1 N/A |

All proxy metrics are green, but proxy metrics do **not** guarantee semantic correctness, language fidelity, or citation hygiene.

---  

## 2. Question‑by‑question Verdicts  

| # | Question (ID) | Verdict | Key Findings |
|---|----------------|---------|--------------|
| **PQ0002** | “SKU ELE‑0012 pe ZentroMart Assured hai kya? Claim kaise karein?” (Hinglish) | **PARTIALLY CORRECT** | • Response is **English**, not Hinglish → language‑matching rule not honoured.<br>• Correctly states product is ZentroMart Assured, claim via *My Orders*, pickup & repair/replacement.<br>• **Missing** explicit “12‑month” warranty duration that is present in `products.csv:ELE‑0012` and `WAR‑2`.<br>• Citations: only the product chunk (`products.csv:ELE‑0012`) and `WAR‑2` are relevant – no spurious citations. |
| **PQ0003** | “How fast are refunds to the ZentroMart Wallet?” (English) | **CORRECT** | • Answer: “instant, within 2 hours” – matches `REF‑2` exactly.<br>• Cites `(REF‑2)` which is in the retrieved set.<br>• No extra or missing details. |
| **PQ0004** | “Can I schedule a delivery at midnight?” (English) | **PARTIALLY CORRECT** | • Correctly refuses to invent a policy and offers human hand‑off.<br>• **Citation** `(POL‑DEL)` is **misleading** – `POL‑DEL` (delivery policy) contains no mention of midnight scheduling; the evidence simply does not support the claim.<br>• Better to cite nothing or note “no evidence found”. |
| **PQ0005** | “SKU ELE‑0002 pe ZentroMart Assured hai kya? Claim kaise karein?” (Hinglish) | **INCORRECT** | • Response again in **English** (no Hinglish mirroring).<br>• Omits explicit “12‑month” warranty duration despite catalog (`products.csv:ELE‑0002` → `warranty_months=12`) and `WAR‑2`.<br>• **Hallucinated** clause: “*ZentroMart Assured does not apply to marketplace seller items, even if the listing says assured. Seller items carry only the brand warranty.*” – this statement is **absent** from `WAR‑2`, `POL‑WAR`, or the product row. It invents a policy restriction.<br>• Citation set includes only relevant sections (`products.csv:ELE‑0002`, `WAR‑2`, `POL‑WAR`) but the added sentence has no backing evidence. |
| **PQ0006** | “As a ZentroMart Plus member, will I be charged the pickup fee for returning shoes I just don’t like?” (English) | **CORRECT** | • States fee is waived for Plus members – matches `RET‑6` (“fee is waived for ZentroMart Plus members”) and `PLS‑2` (“No reverse pickup fee on any return”).<br>• Cites `[RET‑6]` which is in retrieved set and directly supports the claim.<br>• No extra or missing details. |

---  

## 3. Citation Audit  

| Question | Citations used | Supported by retrieved evidence? |
|----------|----------------|----------------------------------|
| PQ0002 | `products.csv:ELE‑0012`, `WAR‑2` (implicit via evidence) | **Yes** |
| PQ0003 | `(REF‑2)` | **Yes** |
| PQ0004 | `(POL‑DEL)` | **No** – `POL‑DEL` does not discuss midnight delivery. |
| PQ0005 | `products.csv:ELE‑0002`, `WAR‑2`, `POL‑WAR` (implicit) | **Partially** – the extra marketplace‑seller sentence has **no** citation. |
| PQ0006 | `[RET‑6]` | **Yes** |

---  

## 4. Comparison with Phase 4.8 Review  

| Issue | Phase 4.8 | Phase 4.10 (this run) | Progress |
|-------|-----------|-----------------------|----------|
| Hinglish language mirroring | Not done (both warranty answers English) | Still English for PQ0002 & PQ0005 | **No improvement** |
| Explicit 12‑month warranty mention | Missing in both warranty answers | Still missing in both | **No improvement** |
| Citation hygiene (midnight delivery) | `(POL‑DEL)` used then, still used now | Same misleading citation persists | **No improvement** |
| Hallucinated marketplace‑seller clause | Not present in Phase 4.8 | **New** hallucination appears in PQ0005 | **Regression** |
| Plus‑member fee answer | Correct | Correct | **Stable** |

---  

## 5. Remaining Issues  

1. **Language mirroring** – the prompt addition (Rule 6) did not translate into Hinglish output for the two Hinglish queries.  
2. **Numeric‑detail rule** – Rule 7 (“include warranty months”) not honoured; both warranty answers omit “12‑month”.  
3. **Citation restriction** – Rule 4 amendment (“cite only directly supporting evidence”) not respected for PQ0004 and the hallucinated sentence in PQ0005.  
4. **Hallucination** – model introduced a marketplace‑seller limitation not present in any retrieved document.  

---  

## 6. Recommendations  

* Strengthen the system prompt with few‑shot examples showing Hinglish ↔ Hinglish and English ↔ English exchanges.  
* Add a post‑generation check that verifies every numeric fact (e.g., “12‑month”) appears when the evidence contains it.  
* Implement a citation validator that strips or flags citations whose source text does not contain the claimed fact.  
* Consider a smaller‑temperature or constrained‑decoding setting for higher factual fidelity.  

---  

*Report generated per Phase 4.10 instructions. No code, prompts, dataset, or evaluation scores were altered.*