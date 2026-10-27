# src/eval/baseline_eval.py
"""Baseline evaluation script.

Reads ``practice/questions.csv`` and sends each question to the baseline
agent, then saves:
  - ``outputs/eval_baseline_responses.csv`` – question ID, question, model
    response, any error.
  - ``outputs/eval_baseline_summary.txt``   – run metadata and failure count.

Usage::

    # Dry-run (first 3 questions, no API calls)
    python -m src.eval.baseline_eval --dry-run

    # Full run (requires GEMINI_API_KEY or GROQ_API_KEY)
    python -m src.eval.baseline_eval

    # Limit to first N questions
    python -m src.eval.baseline_eval --limit 5

IMPORTANT: This script measures raw model output only.
Automated grading (comparing against ``must_include`` / ``must_not_include``)
is produced as an INFORMATIONAL column only; human review is required before
citing any score as the final benchmark result.
"""

import argparse
import csv
import os
import re
import sys
import textwrap
from datetime import datetime, UTC
from pathlib import Path

# Allow running as ``python -m src.eval.baseline_eval`` from repo root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

QUESTIONS_CSV = ROOT / "practice" / "questions.csv"
OUTPUTS_DIR = ROOT / "outputs"

DATASET_DATE = "20 October 2026"


def load_questions(limit: int | None = None) -> list[dict]:
    with QUESTIONS_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    return rows


def _check_must_include(response: str, must_include: str) -> str:
    """
    Returns 'PASS', 'FAIL', or 'N/A'.
    ``must_include`` may contain ``/``-separated alternatives; any one suffices.
    NOTE: this is heuristic keyword matching, NOT authoritative scoring.
    """
    if not must_include.strip():
        return "N/A"
    resp_lower = response.lower()
    alternatives = [a.strip().lower() for a in must_include.split("/")]
    return "PASS" if any(alt in resp_lower for alt in alternatives) else "FAIL"


def _check_must_not_include(response: str, must_not: str) -> str:
    if not must_not.strip():
        return "N/A"
    resp_lower = response.lower()
    banned = [b.strip().lower() for b in must_not.split("/")]
    return "FAIL" if any(b in resp_lower for b in banned) else "PASS"


def run_evaluation(dry_run: bool = False, limit: int | None = None) -> None:
    """Main evaluation routine."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_questions(limit)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_csv = OUTPUTS_DIR / f"eval_baseline_responses_{timestamp}.csv"
    out_summary = OUTPUTS_DIR / f"eval_baseline_summary_{timestamp}.txt"

    provider = os.getenv("LLM_PROVIDER", "gemini")
    fieldnames = [
        "id", "language", "intent", "expected_decision", "answerable",
        "question",
        "model_response",
        "error",
        "heuristic_must_include",
        "heuristic_must_not_include",
    ]

    results = []
    api_errors = 0

    if not dry_run:
        # Import here so tests can import this module without needing real keys.
        from src.agent.baseline import BaselineAgent

    print(f"Running baseline evaluation | provider={provider} | questions={len(questions)} | dry_run={dry_run}")

    for i, row in enumerate(questions, 1):
        qid = row["id"]
        question = row["question"]
        print(f"  [{i}/{len(questions)}] {qid}: {question[:70]}...")

        error = ""
        response = ""

        if dry_run:
            response = f"[DRY-RUN] Would call {provider} with: {question}"
        else:
            # Each question is answered independently (no conversation state
            # carried across questions, matching single-question eval protocol).
            try:
                agent = BaselineAgent()
                agent.add_user_message(question)
                response = agent.get_response()
            except Exception as exc:
                error = str(exc)
                api_errors += 1
                print(f"    ERROR: {error}")

        must_include_check = _check_must_include(response, row.get("must_include", ""))
        must_not_check = _check_must_not_include(response, row.get("must_not_include", ""))

        results.append({
            "id": qid,
            "language": row.get("language", ""),
            "intent": row.get("intent", ""),
            "expected_decision": row.get("expected_decision", ""),
            "answerable": row.get("answerable", ""),
            "question": question,
            "model_response": response,
            "error": error,
            "heuristic_must_include": must_include_check,
            "heuristic_must_not_include": must_not_check,
        })

    # Write CSV
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Tally heuristic results (informational only)
    mi_pass = sum(1 for r in results if r["heuristic_must_include"] == "PASS")
    mi_fail = sum(1 for r in results if r["heuristic_must_include"] == "FAIL")
    mn_pass = sum(1 for r in results if r["heuristic_must_not_include"] == "PASS")
    mn_fail = sum(1 for r in results if r["heuristic_must_not_include"] == "FAIL")

    summary = textwrap.dedent(f"""
        ZentroAI Baseline Evaluation Summary
        =====================================
        Timestamp      : {timestamp}
        Provider       : {provider}
        Dataset date   : {DATASET_DATE}
        Questions run  : {len(questions)}
        Dry run        : {dry_run}
        API errors     : {api_errors}

        Heuristic keyword checks (INFORMATIONAL – not authoritative scores)
        -------------------------------------------------------------------
        must_include  PASS: {mi_pass}  FAIL: {mi_fail}  N/A: {len(questions) - mi_pass - mi_fail}
        must_not_incl PASS: {mn_pass}  FAIL: {mn_fail}  N/A: {len(questions) - mn_pass - mn_fail}

        Output file: {out_csv}

        NOTE: Heuristic keyword matching is a rough proxy. Human review of
        must_include / must_not_include columns in the CSV is required before
        reporting any official score.
    """).strip()

    with out_summary.open("w", encoding="utf-8") as f:
        f.write(summary)

    print()
    print(summary)
    print(f"\nFull responses saved to: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the baseline LLM-only agent on practice questions."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be sent without calling the API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only evaluate the first N questions (for quick smoke-tests).",
    )
    args = parser.parse_args()
    run_evaluation(dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
