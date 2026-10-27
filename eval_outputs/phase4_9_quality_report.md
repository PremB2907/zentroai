# ZentroAI Phase 4.9: Answer-Quality Improvement Report

**Date**: 26 October 2026  
**Scope**: Targeted improvements to language matching, numeric detail inclusion, citation hygiene, and unsupported‑claim handling for the local Ollama RAG pipeline.  
**Provider**: Ollama (`gemma2:2b`) — default, no cloud fallback.

---

## 1. Changes Made

| File | Change Summary |
|------|----------------|
| `src/agent/rag_agent.py` | Expanded `_SYSTEM_PROMPT` with three new rules:<br>• **Rule 6** – *Match the language of the customer’s question (English, Hindi, or Hinglish). If the question is in Hinglish, respond in Hinglish.*<br>• **Rule 7** – *When evidence contains specific numeric details (e.g., warranty months, fees, timelines), include them in your answer.*<br>• **Rule 4 (amended)** – *Only cite sections that directly support your answer; do not cite irrelevant or unused evidence.* |
| `tests/test_phase2.py` | Added **`TestAnswerQualityImprovements`** (7 new tests) covering:<br>• Presence of language‑matching, numeric‑detail, and citation‑restriction rules in the system prompt.<br>• Product chunk contains the catalog’s warranty‑months field (`Warranty: 12 months`).<br>• `format_evidence` embeds each chunk’s citation ID.<br>• Retrieval for an unsupported midnight‑delivery query returns no chunk that mentions “midnight”. |

No other source files were modified. The dialogue manager, baseline agent, and evaluation harness remain unchanged.

---

## 2. Test Results

```
219 passed, 4 skipped
```

*All existing tests (Phases 1‑4, 4.3, 4.7) continue to pass.*  
The new `TestAnswerQualityImprovements` class passes completely.

`validate_dataset.py` → **PASS** (no dataset regressions).

---

## 3. Behavioural Impact (offline verification)

| Issue | Before | After (prompt‑level) |
|-------|--------|----------------------|
| **Hinglish language mirroring** | Model answered in English. | Prompt now instructs the model to reply in the same language (English / Hindi / Hinglish). |
| **Explicit warranty duration** | Answer omitted “12‑month” (PQ0005). | Rule 7 forces inclusion of numeric details such as warranty months when present in evidence. |
| **Citation hygiene** | Model cited `[POL-DEL]` for a midnight‑delivery question that the policy does not address. | Rule 4 now restricts citations to sections that *directly support* the answer. |
| **SKU‑aware retrieval** | Unchanged – product rows still injected at rank 0. | Preserved; regression tests confirm unchanged behaviour. |

Because the evaluation harness runs in **dry‑run** mode (mocked provider returns `"[DRY-RUN]"`), live model output was not generated during this phase. The prompt changes are deterministic and will take effect on the next live Ollama run.

---

## 4. Remaining Limitations

1. **Live verification pending** – Actual Hinglish responses, warranty‑month inclusion, and citation restraint have not been observed on a real `gemma2:2b` call. A live run (`LLM_PROVIDER=ollama python -m src.eval.run_eval --confirm-live --config rag --limit-q 5`) is required to confirm end‑to‑end behaviour.
2. **Language detection is model‑driven** – The prompt asks the model to mirror the user language; no explicit language‑detection pre‑processor is added.
3. **Citation enforcement is soft** – The model is instructed not to cite irrelevant evidence, but compliance is not programmatically verified.
4. **No new metrics** – Existing proxy metrics (`must_include`, `retrieval_recall`) remain unchanged; they do not yet measure language match or numeric‑detail coverage.

---

## 5. Commands for Next Live Pilot

```bash
cd /home/prem/Prem Projects/zentroai
# Full five‑question live RAG evaluation
LLM_PROVIDER=ollama python3 -m src.eval.run_eval --confirm-live --config rag --limit-q 5
```

The output CSVs will show whether:
* PQ0002 / PQ0005 answers appear in Hinglish,
* “12‑month” (or “12 months”) is present in the warranty answers,
* Midnight‑delivery answer cites only evidence that actually discusses delivery scheduling (or none).

---

## 6. Summary

* **Files changed**: `src/agent/rag_agent.py`, `tests/test_phase2.py`.
* **Tests**: 219 passed, 4 skipped (cloud‑provider tests skipped due to missing `google-genai`).
* **Validation**: `validate_dataset.py` PASS.
* **No live LLM calls** were made during this phase.

The targeted prompt enhancements and regression tests lay the groundwork for higher‑quality, language‑aware, and citation‑clean answers in the next live Ollama evaluation.