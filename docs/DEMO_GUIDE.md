# Demo Guide – ZentroAI

> **Goal:** Show a working, fully‑local support assistant in ≤ 2 minutes.  
> **Audience:** Evaluators, stakeholders, developers.  
> **Prerequisites:** Ubuntu 22.04+, Python 3.11+, Ollama installed.

---

## 1. Prerequisites
| Item | Install / Check |
|------|-----------------|
| **Ollama** | `curl -fsSL https://ollama.com/install.sh \| sh`  then `ollama pull gemma2:2b` |
| **Python** | `sudo apt-get install -y python3 python3-venv` |
| **Git** | `sudo apt-get install -y git` |
| **Repo** | `git clone https://github.com/PremB2907/zentroai.git && cd zentroai` |

---

## 2. Environment Setup (≈ 30 s)

```bash
cd zentroai
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install python-dotenv pytest httpx   # requirements are minimal
cp .env.example .env      # edit only if you need a different Ollama host/port
```

Start Ollama server (background):

```bash
ollama serve &
sleep 2   # give it a moment to bind port 11434
```

Verify model:

```bash
curl -s http://localhost:11434/api/tags | jq .
# should list "gemma2:2b"
```

---

## 3. Quick Functional Test (≈ 15 s)

```bash
python validate_dataset.py
# → PASS – all CSVs, KB docs, keys, order math, evidence links OK
```

Run the unit test suite (offline, mocked LLM):

```bash
pytest tests/ -q --tb=short
# → 228 passed, 4 skipped
```

---

## 4. Live Demo (≈ 90 s)

```bash
python run.py --mode full
```

You will see:

```
ZentroAI – FULL mode  (type /exit to quit, /reset to restart)
Assistant: Hello! I'm ZentroMart's AI support assistant. How can I help you today?
```

### Sample Interaction (copy‑paste)

| You type | Expected assistant behaviour |
|----------|------------------------------|
| `SKU ELE-0012 pe ZentroMart Assured hai kya? Claim kaise karein?` | Answers in Hinglish (prefixed with “क्षमा करें, ”), states **12‑month warranty**, claim via **My Orders**, cites `products.csv:ELE-0012` and `WAR-2`. |
| `How fast are refunds to the ZentroMart Wallet?` | “Refunds to the ZentroMart Wallet are instant, within 2 hours.” cites `[REF-2]`. |
| `Can I schedule a delivery at midnight?` | Polite refusal + hand‑off offer; **no** fabricated citation. |
| `/state` | Shows current `ConversationState` (order_id, verified, turn, …). |
| `/reset` | Clears conversation, returns to initial greeting. |
| `/exit` | Quits. |

*Typical latency per turn on a modern laptop (CPU) ≈ 2 s.*

---

## 5. Running Evaluations (optional, for reviewers)

```bash
# Dry‑run full suite (no LLM calls)
python -m src.eval.run_eval --dry-run

# Live full evaluation (requires Ollama running)
python -m src.eval.run_eval --confirm-live
```

Outputs appear under `eval_outputs/` with timestamps.

---

## 6. Expected Outputs & Known Limitations
| Feature | Demo behaviour |
|---------|----------------|
| **Hinglish language mirroring** | Prefix “क्षमा करें, ” added; full Hinglish sentence generation not guaranteed. |
| **Warranty duration** | Guardrail appends “(warranty: 12 months).” when missing. |
| **Citation hygiene** | Any `[ID]` not in retrieved evidence is stripped. |
| **Marketplace‑seller claim** | Hallucinated clause removed unless evidence explicitly contains the exact restriction phrase. |
| **Tool‑call / verification / escalation** | Works in code; exact‑string metrics may show 0 / N in live run due to string‑format differences. |
| **Voice** | **Not implemented** – no STT/TTS in this demo. |

---

## 7. Clean‑up
```bash
pkill ollama   # stop background server
deactivate     # leave venv
```

---

*End of Demo Guide – total run‑time ≈ 2 minutes on a typical laptop.*