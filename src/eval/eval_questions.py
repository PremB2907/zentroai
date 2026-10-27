# src/eval/eval_questions.py
"""Question-level evaluator (Configs A and B).

Config A – Baseline LLM-only agent (``BaselineAgent``).
Config B – RAG agent without tools (``RAGAgent``).

Reads ``practice/questions.csv``.  Each question is evaluated independently
(fresh agent per question) matching the single-question evaluation protocol.

Error accounting (Phase 4.3 fix)
---------------------------------
* A call is counted as an API error if:
  - The runner catches an exception (including :class:`ProviderError`), OR
  - The response starts with the legacy ``[Error communicating`` prefix.
* API-error rows are never included in heuristic pass rates.
* A ``provider_used`` column records which provider actually answered.
* The configured provider priority is reported in the summary header.

Heuristic metrics (must_include / must_not_include) are INFORMATIONAL only.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import textwrap
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.eval.metrics import (
    check_must_include,
    check_must_not_include,
    check_retrieval_recall,
    summarise_questions,
)

QUESTIONS_CSV = ROOT / "practice" / "questions.csv"
EVAL_DIR = ROOT / "eval_outputs"
DATASET_DATE = "20 October 2026"

# Legacy error-string prefix written by the old _handle_error helper.
# Any model_response starting with this is treated as an API failure.
_ERROR_PREFIX = "[Error communicating"


def _is_error_response(text: str) -> bool:
    """Return True if ``text`` is a provider error string, not a real answer."""
    return text.startswith(_ERROR_PREFIX) or text.startswith("[Error from LLM")


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_questions(limit: int | None = None) -> list[dict]:
    with QUESTIONS_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------------------
# Single-question runners  (return provider_used alongside other fields)
# ---------------------------------------------------------------------------

def _run_baseline_question(question: str) -> tuple[str, str, str, float]:
    """Returns (response, error, provider_used, latency_s)."""
    from src.agent.baseline import BaselineAgent
    from src.llm.base import ProviderError
    t0 = time.monotonic()
    try:
        agent = BaselineAgent()
        provider_used = getattr(agent.provider, "PROVIDER_NAME", "unknown")
        agent.add_user_message(question)
        response = agent.get_response()
        # Guard: catch legacy error strings that slipped through.
        if _is_error_response(response):
            return "", response, provider_used, round(time.monotonic() - t0, 3)
        return response, "", provider_used, round(time.monotonic() - t0, 3)
    except ProviderError as exc:
        return "", exc.safe_message, exc.provider_name, round(time.monotonic() - t0, 3)
    except Exception as exc:
        return "", str(exc), "unknown", round(time.monotonic() - t0, 3)


def _run_rag_question(
    question: str,
    top_k: int = 5,
) -> tuple[str, str, str, str, float]:
    """Returns (response, retrieved_sections, error, provider_used, latency_s)."""
    from src.agent.rag_agent import RAGAgent
    from src.llm.base import ProviderError
    t0 = time.monotonic()
    try:
        agent = RAGAgent(top_k=top_k)
        provider_used = getattr(agent._provider, "PROVIDER_NAME", "unknown")
        result = agent.chat(question)
        retrieved = " ; ".join(
            (r.chunk.section_id or r.chunk.doc_id)
            for r in result.retrieval_results
        )
        # Guard: catch legacy error strings that slipped through.
        if _is_error_response(result.answer):
            return "", retrieved, result.answer, provider_used, round(time.monotonic() - t0, 3)
        return result.answer, retrieved, "", provider_used, round(time.monotonic() - t0, 3)
    except ProviderError as exc:
        return "", "", exc.safe_message, exc.provider_name, round(time.monotonic() - t0, 3)
    except Exception as exc:
        return "", "", str(exc), "unknown", round(time.monotonic() - t0, 3)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_question_eval(
    config: str,          # "baseline" or "rag"
    dry_run: bool = False,
    retrieve_only: bool = False,
    limit: int | None = None,
    top_k: int = 5,
    output_dir: Path | None = None,
) -> tuple[Path, dict]:
    """Run the question-level evaluation for one config.

    Parameters
    ----------
    config:
        ``"baseline"`` or ``"rag"``.
    dry_run:
        Skip LLM calls; record retrieved evidence only (RAG) or a stub
        response (baseline).
    retrieve_only:
        (RAG only) Run retrieval but skip the LLM call.
    limit:
        Evaluate only the first N questions.
    top_k:
        Number of KB chunks to retrieve (RAG only).
    output_dir:
        Directory for output files.  Defaults to ``eval_outputs/``.

    Returns
    -------
    (out_csv_path, summary_dict)
    """
    assert config in ("baseline", "rag"), f"Unknown config: {config!r}"
    out_dir = output_dir or EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(limit)
    configured_provider = os.getenv("LLM_PROVIDER", "ollama")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_csv = out_dir / f"eval_{config}_questions_{timestamp}.csv"

    mode_label = "retrieve-only" if retrieve_only else ("dry-run" if dry_run else "live")
    print(
        f"Questions eval | config={config} | mode={mode_label} | "
        f"provider={configured_provider} | n={len(questions)} | top_k={top_k}"
    )

    fieldnames = [
        "id", "language", "intent", "expected_decision", "answerable",
        "question", "gold_evidence", "reference_answer",
        "retrieved_sections", "retrieval_recall",
        "model_response", "provider_used", "error", "latency_s",
        "heuristic_must_include", "heuristic_must_not_include",
        "is_api_error", "needs_manual_review",
    ]

    results: list[dict] = []
    api_errors = 0

    for i, row in enumerate(questions, 1):
        qid = row["id"]
        question = row["question"]
        print(f"  [{i}/{len(questions)}] {qid}: {question[:65]}…")

        response = error = retrieved_sections = provider_used = ""
        latency_s = 0.0
        is_api_error = False

        if dry_run or retrieve_only:
            # RAG: run retrieval offline; skip LLM call.
            if config == "rag":
                from src.agent.rag_agent import RAGAgent
                from unittest.mock import MagicMock
                t0 = time.monotonic()
                try:
                    agent = RAGAgent(top_k=top_k)
                    # Mock the provider to avoid actual LLM calls
                    mock_provider = MagicMock()
                    mock_provider.generate = MagicMock(return_value="[DRY-RUN]")
                    mock_provider.PROVIDER_NAME = "ollama"
                    agent._provider = mock_provider
                    # Use the full chat flow which includes SKU-aware product injection
                    result = agent.chat(question)
                    retrieved_sections = " ; ".join(
                        (r.chunk.section_id or r.chunk.doc_id)
                        for r in result.retrieval_results
                    )
                    latency_s = round(time.monotonic() - t0, 3)
                except Exception as exc:
                    error = str(exc)
                    api_errors += 1
                    is_api_error = True
                response = f"[{mode_label.upper()}] retrieved: {retrieved_sections or 'none'}"
            else:
                response = f"[{mode_label.upper()}]"
            provider_used = "dry-run"
        else:
            if config == "baseline":
                response, error, provider_used, latency_s = _run_baseline_question(question)
            else:
                response, retrieved_sections, error, provider_used, latency_s = _run_rag_question(
                    question, top_k=top_k
                )
            if error:
                is_api_error = True
                api_errors += 1
                print(f"    API ERROR ({provider_used}): {error[:120]}")

        # Heuristic metrics — only scored for non-error responses.
        if is_api_error:
            mi_check = mn_check = "API_ERROR"
        else:
            mi_check = check_must_include(response, row.get("must_include", ""))
            mn_check = check_must_not_include(response, row.get("must_not_include", ""))

        ret_recall = check_retrieval_recall(
            retrieved_sections, row.get("gold_evidence", "")
        )

        results.append({
            "id": qid,
            "language": row.get("language", ""),
            "intent": row.get("intent", ""),
            "expected_decision": row.get("expected_decision", ""),
            "answerable": row.get("answerable", ""),
            "question": question,
            "gold_evidence": row.get("gold_evidence", ""),
            "reference_answer": row.get("reference_answer", ""),
            "retrieved_sections": retrieved_sections,
            "retrieval_recall": ret_recall,
            "model_response": response,
            "provider_used": provider_used,
            "error": error,
            "latency_s": latency_s,
            "heuristic_must_include": mi_check,
            "heuristic_must_not_include": mn_check,
            "is_api_error": is_api_error,
            "needs_manual_review": row.get("needs_native_review", "False"),
        })

    # Write CSV — partial results are preserved even if we stopped early.
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary = summarise_questions(results)
    summary.update({
        "config": config,
        "mode": mode_label,
        "configured_provider": configured_provider,
        "provider_priority": "ollama (default) | groq | nvidia | gemini (explicit override)",
        "dataset_date": DATASET_DATE,
        "timestamp": timestamp,
        "questions": len(questions),
        "top_k": top_k,
        "output_csv": str(out_csv),
    })
    return out_csv, summary
