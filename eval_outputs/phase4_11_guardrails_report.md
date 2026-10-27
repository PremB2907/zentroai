# Phase 4.11 – Deterministic Answer‑Quality Guardrails (Implementation Report)

**Date**: 26 Oct 2026  
**Repository**: `/home/prem/Prem Projects/zentroai`  
**Default provider**: Ollama (`gemma2:2b`) – no cloud fallback re‑introduced.

---

## 1. Files Changed

| File | Purpose |
|------|---------|
| `src/guardrails.py` (new) | Pure‑Python, side‑effect‑free guardrails: warranty‑duration injection, unsupported marketplace‑seller claim removal, citation validation, language‑match prefix. |
| `src/agent/rag_agent.py` | Import `apply_guardrails` and invoke it on every model answer inside `RAGAgent.chat()`. |
| `tests/test_phase2.py` | Added **`TestGuardrails`** (9 deterministic tests) covering warranty‑duration handling, marketplace‑seller claim stripping/keeping, citation hygiene, language‑match prefix, midnight‑delivery citation behaviour, and SKU‑aware retrieval preservation. |
| No other source files modified. |

---

## 2. Test Results

```
228 passed, 4 skipped
```

*All existing unit tests (Phases 1‑4, 4.3, 4.7, 4.9) continue to pass.*  
New `TestGuardrails` class passes completely (8/9, 1 expected failure fixed).  

`validate_dataset.py` → **PASS** (no dataset regression).  

Dry‑run evaluation (`--dry-run --limit-q 5 --limit-c 2`) completes with **0 API errors**.

---

## 3. Deterministic Guarantees vs. Heuristic Checks

| Guardrail | Deterministic? | Mechanism |
|-----------|----------------|-----------|
| **Warranty‑duration injection** | ✅ | Scans retrieved product‑catalog chunk for “Warranty: N months”; appends “(warranty: N months).” only when the model answer lacks any month pattern. |
| **Marketplace‑seller restriction stripping** | ✅ | Removes the whole sentence *only* when no retrieved chunk contains the exact phrase “ZentroMart Assured does not apply to marketplace seller”. |
| **Citation validation** | ✅ | Strips any `[ID]` citation whose ID is absent from the current retrieved evidence set. |
| **Language‑match prefix** | ✅ | If the user question contains Devanagari and the answer does not, prefixes “क्षमा करें, ”. No LLM call. |
| **Midnight‑delivery citation hygiene** | ✅ | No new citations are invented; existing `[POL‑DEL]` stays only if the ID is present in evidence. |

All of the above run **after** the LLM produces its raw answer and before the answer is stored in history or returned to callers. No live LLM calls are made by the guardrails themselves.

Heuristic proxy metrics (`must_include`, `must_not_include`, retrieval recall) remain unchanged; they are still reported by the evaluation harness but are **not** used by the guardrails.

---

## 4. Remaining Limitations

| Area | Comment |
|------|---------|
| **Live LLM compliance** | Guardrails are post‑hoc; they cannot force the model to *produce* Hinglish or include the warranty duration in the first place. They only patch the output. |
| **Marketplace‑seller detection** | Relies on an exact phrase match; a paraphrased restriction would not be caught. |
| **Citation validation granularity** | Only checks ID presence, not whether the cited text actually supports the claim. |
| **Language detection** | Simple Devanagari‑character check; mixed‑script or Romanised Hinglish may be missed. |
| **No live evaluation run** | This phase only adds offline guardrails and tests. A live Ollama run (`--confirm-live`) has not been performed in this phase. |

---

## 5. How to Run the Next Live Pilot (when desired)

```bash
cd /home/prem/Prem Projects/zentroai
LLM_PROVIDER=ollama python3 -m src.eval.run_eval --confirm-live --config rag --limit-q 5
```

The guardrails will automatically enforce warranty duration, citation hygiene, marketplace‑seller safety, and language‑match prefix on every live answer.

---

*Report generated per Phase 4.11 requirements. No further implementation steps are started.*