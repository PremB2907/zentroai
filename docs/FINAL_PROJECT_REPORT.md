# Final Project Report – ZentroAI

**Date:** 27 Oct 2026 **Repository:** `https://github.com/PremB2907/zentroai`

---

## 1. Abstract
ZentroAI is a fully local, privacy‑first customer‑support assistant for the synthetic ZentroMart e‑commerce platform. It combines a BM25‑based retrieval engine over 21 policy/FAQ documents, a deterministic SKU‑aware product‑catalog lookup, a conversation manager with identity verification and escalation, and a suite of post‑generation guardrails that enforce warranty‑duration inclusion, citation hygiene, marketplace‑seller claim safety, and Hinglish language mirroring. All inference runs on a local Ollama instance (`gemma2:2b`), eliminating cloud API dependencies.

---

## 2. Problem Statement
Customer‑support bots must answer accurately from evolving policy documents, respect verification gates, avoid hallucination, and operate without sending personal data to external APIs. Existing cloud‑centric solutions breach privacy, incur cost, and lack deterministic citation control.

---

## 3. Objectives
1. Build an offline‑capable RAG pipeline over Markdown policy corpus.  
2. Inject verified product facts (warranty, price, category) when a SKU appears.  
3. Implement verification‑gated order lookup and escalation tools.  
4. Add deterministic post‑generation guardrails for warranty duration, citation validity, marketplace‑seller claim safety, and language matching.  
5. Provide a reproducible offline evaluation harness plus a live‑run mode on local Ollama.  
6. Deliver full documentation, test suite, and reproducible setup instructions.

---

## 4. Architecture & Methodology
* **Retrieval** – BM25 over section‑level chunks; superseded‑doc penalty; policy precedence via doc‑type priority.  
* **SKU‑aware injection** – Exact SKU match → product row turned into a high‑score chunk (rank 0).  
* **Conversation manager** – State machine for verification (max 1 retry), tool calls, escalation triggers, ticket creation.  
* **Guardrails** – Pure Python functions applied after LLM output: warranty‑duration append, marketplace‑seller claim removal, citation ID validation, Hinglish prefix.  
* **Provider abstraction** – `LLMBase` with `OllamaProvider` default; optional cloud providers behind feature flags.  
* **Evaluation** – Dry‑run (mocked LLM) for CI; live mode for end‑to‑end latency & quality metrics.

---

## 5. Implementation Highlights
| Component | LOC (approx.) | Key Files |
|-----------|--------------|-----------|
| RAG loader / retriever | 350 | `src/rag/loader.py`, `src/rag/retriever.py` |
| Product catalog & SKU injection | 120 | `src/rag/loader.py`, `src/agent/rag_agent.py`, `src/agent/dialogue_manager.py` |
| Conversation manager & tools | 300 | `src/agent/dialogue_manager.py`, `src/agent/state.py`, `src/tools/order_tools.py` |
| Guardrails | 80 | `src/guardrails.py` |
| Evaluation harness | 400 | `src/eval/*.py` |
| Tests | 500+ | `tests/*.py` (228 passing, 4 skipped) |

All code is pure Python 3.11+, no heavyweight ML libraries.

---

## 6. Dataset & Evaluation Methodology
* **Synthetic corpus** – 70 products, 1 000 orders, 21 policy/FAQ markdown files, 52 single‑turn questions, 23 conversations (96 turns). Fixed reference date **20 Oct 2026**.  
* **Evaluation modes** – *dry‑run* (mocked LLM, zero latency, tests retrieval & tool flow) and *live* (real Ollama `gemma2:2b`).  
* **Metrics** – keyword proxies (`must_include`, `must_not_include`), retrieval recall (gold evidence IDs), tool‑call match, verification‑turn correctness, escalation correctness.  
* **Human review** – 5‑question pilot (Phase 4.8) scored 5/5; later live run (Phase 4.10) exposed language‑matching gaps, missing warranty duration, misleading citation, and a hallucinated marketplace‑seller clause. The five‑question sample is **not representative** of the full 52‑question set.

---

## 7. Results (Live Run – 26 Oct 2026)

| Mode | Items | API errors | Avg latency | Retrieval PASS | `must_include` PASS |
|------|-------|------------|-------------|----------------|--------------------|
| Baseline | 52 q | 0 | **2.392 s** | – | – |
| RAG | 52 q | 0 | **2.012 s** | 18 PASS / 15 PARTIAL / 14 FAIL | 19 PASS / 30 FAIL / 3 N/A |
| Full DM | 23 conv / 96 turns | 0 | **2.4 s** | 43 PASS / 5 PARTIAL / 48 FAIL | 0 PASS / 77 FAIL / 19 N/A |
| Full DM – tool‑call | 54 expected | 0/54 correct | – | – | – |
| Full DM – verification | 25 expected | 0/25 correct | – | – | – |
| Full DM – escalation | 6 expected | 0/6 correct | – | – | – |

*All numbers from `eval_outputs/eval_summary_*.json` (26 Oct 2026 live run).*  
Proxy metrics (`must_include`, retrieval) are **keyword‑based proxies**, not proof of semantic correctness. Tool‑call, verification, and escalation scores reflect exact string match against expected CSV columns; a “0 correct” does not prove the underlying logic never fired – it indicates string mismatch between expected and actual tool‑call strings.

---

## 8. Limitations
* **Language guardrail** – only prefixes “क्षमा करें, ” for Devanagari‑containing prompts; full Hinglish generation not guaranteed.  
* **Guardrails are post‑hoc** – they cannot prevent the model from initially hallucinating; they only patch the final text.  
* **Citation validation** – checks ID existence only, not semantic support.  
* **Tool‑call / verification / escalation scores** low on live run – string‑match mismatch, not necessarily logical failure.  
* **No voice pipeline** – STT/TTS not implemented.  
* **Synthetic data only** – real‑world noise, multilingual noise, and adversarial prompts untested.

---

## 9. Future Scope
1. Replace post‑hoc guardrails with constrained decoding or a verifier LLM.  
2. Add semantic citation entailment check.  
3. Multilingual prompt engineering / fine‑tune for Hinglish.  
4. Improve DialogueManager prompts to raise tool‑call / verification / escalation exact‑match rates.  
5. Integrate Gnani STT/TTS for voice‑enabled demo.  
6. Extend evaluation with human‑in‑the‑loop semantic grading.  
7. Package as a Docker image for one‑command deployment.

---

## 10. Conclusion
ZentroAI demonstrates that a high‑quality, privacy‑preserving support assistant can be built entirely on local hardware using open‑source components. The deterministic guardrails and SKU‑aware retrieval give strong factual grounding, while the offline evaluation harness enables rapid iteration without cloud costs. Remaining work centers on tighter language generation, tighter tool‑call alignment, and expanding to multimodal (voice) interaction.

---

*Prepared for the ZentroAI capstone submission – 27 Oct 2026.*