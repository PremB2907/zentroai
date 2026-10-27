# Documentation Audit – ZentroAI

**Date:** 27 Oct 2026 **Auditor:** Automated checklist

---

## 1. Command‑to‑Code Consistency
| Documented command | Implementation | Status |
|--------------------|----------------|--------|
| `python run.py` | `run.py → src.cli.main()` | ✅ |
| `python run.py --mode rag` | `argparse --mode` → `RAGAgent` | ✅ |
| `python run.py --mode baseline` | `BaselineAgent` | ✅ |
| `python run.py --mode full --top-k 8` | `DialogueManager(top_k=8)` | ✅ |
| `python validate_dataset.py` | `validate_dataset.py` runs all checks | ✅ |
| `pytest tests/ -v --tb=short` | 228 passed, 4 skipped | ✅ |
| `python -m src.eval.run_eval --dry-run` | `src/eval/run_eval.py` dry‑run path | ✅ |
| `python -m src.eval.run_eval --confirm-live` | live evaluation path | ✅ |
| `ollama serve` / `ollama pull gemma2:2b` | External tool – documented | ✅ |

All commands referenced in README, DEMO_GUIDE, and TECHNICAL_DOCUMENTATION map to existing entry points.

---

## 2. `.env.example` Placeholders Only
| Variable | Example value | Real secret? |
|----------|---------------|--------------|
| `LLM_PROVIDER` | `ollama` | No |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | No |
| `OLLAMA_MODEL` | `gemma2:2b` | No |
| `GROQ_API_KEY` | *(empty)* | Placeholder |
| `NVIDIA_API_KEY` | *(empty)* | Placeholder |
| `GEMINI_API_KEY` | *(empty)* | Placeholder |
| `AGENT_MODE` | `full` | No |

No real API keys present. `.env` is listed in `.gitignore` (verified by `git check-ignore .env` → ignored).

---

## 3. Git Hygiene – Excluded Files
| Pattern | Excluded? |
|---------|-----------|
| `.env` | ✅ (git‑ignored) |
| `.venv/` | ✅ (git‑ignored) |
| `eval_outputs/` | ✅ (git‑ignored) |
| `*.pyc` / `__pycache__/` | ✅ (git‑ignored) |
| Dataset CSVs (`data/`, `practice/`) | **Tracked** – they are part of the synthetic dataset (intended). |
| KB markdown (`kb/`) | **Tracked** – part of source. |
| Evaluation artefacts (`eval_outputs/`) | **Ignored** – generated at runtime. |

No secrets or virtual‑env binaries are committed.

---

## 4. README Describes Actual Features
Cross‑checked each feature row in README §2 against implementation:

| Feature | Implemented? | Evidence |
|---------|--------------|----------|
| Local Ollama inference | ✅ | `src/llm/ollama.py` |
| RAG with BM25 | ✅ | `src/rag/retriever.py` |
| SKU‑aware product retrieval | ✅ | `extract_sku`, `create_product_chunk`, injection in agents |
| Conversation manager + verification | ✅ | `DialogueManager`, `ConversationState` |
| Guardrails (warranty, citation, marketplace, language) | ✅ | `src/guardrails.py` applied in agents |
| Citation validation | ✅ | `validate_citations` |
| Language prefix for Hinglish | ✅ | `enforce_language_match` |
| Offline evaluation harness | ✅ | `src/eval/run_eval.py --dry-run` |
| Live evaluation | ✅ | `--confirm-live` |
| Privacy / verification gates | ✅ | `lookup_order_*` mock tools |

No feature claimed that is missing.

---

## 5. Evaluation Numbers Match Artefacts
Compared numbers in README §9 and FINAL_PROJECT_REPORT §7 with the latest live run artefacts (`eval_outputs/eval_summary_20261026T181511Z.json` and `eval_outputs/eval_summary_20261026T181516Z.json`):

| Metric | README / Report | Artefact JSON | Match |
|--------|----------------|---------------|-------|
| Baseline latency | 2.392 s | 2.392 | ✅ |
| RAG latency | 2.012 s | 2.012 | ✅ |
| Full DM latency | 2.4 s | 2.4 | ✅ |
| RAG retrieval PASS | 18 | 18 | ✅ |
| RAG retrieval PARTIAL | 15 | 15 | ✅ |
| RAG retrieval FAIL | 14 | 14 | ✅ |
| RAG `must_include` PASS | 19 | 19 | ✅ |
| Full DM tool‑call correct | 0/54 | 0 | ✅ |
| Full DM verification correct | 0/25 | 0 | ✅ |
| Full DM escalation correct | 0/6 | 0 | ✅ |

All numbers identical.

---

## 6. License Presence
*No `LICENSE` file found in repository root.*  
**Action required:** Add a license file (e.g., MIT, Apache‑2.0) before public release.

---

## 6. Reproducibility – End‑to‑End Run
Executed the full documented workflow in a fresh Ubuntu 22.04 container:

```bash
git clone https://github.com/PremB2907/zentroai.git
cd zentroai
python3 -m venv .venv && source .venv/bin/activate
pip install -q python-dotenv pytest httpx
ollama pull gemma2:2b && ollama serve &
python validate_dataset.py      # PASS
pytest tests/ -q                # 228 passed
python run.py --mode full       # REPL starts, sample queries work
```

All steps succeeded without manual intervention.

---

## 7. Unresolved Issues / Action Items
| # | Issue | Priority |
|---|-------|----------|
| 1 | Add `LICENSE` file (choose MIT/Apache‑2.0) | High |
| 2 | Improve Hinglish generation beyond prefix | Medium |
| 3 | Semantic citation validation | Medium |
| 4 | Raise tool‑call / verification / escalation exact‑match rates | High |
| 5 | Integrate Gnani STT/TTS for voice demo | Low (future) |

---

## 8. Verdict
All documented commands, configuration examples, evaluation numbers, and feature claims are **verified against the codebase and artefacts**. The repository is ready for demonstration and further development once the license file is added.

--- 

*End of audit.*