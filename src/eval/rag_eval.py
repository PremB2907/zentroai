# src/eval/rag_eval.py
"""RAG evaluation script.

Reads ``practice/questions.csv`` and, for each question:
  1. Retrieves evidence chunks from the KB (offline — no API call).
  2. Optionally sends the RAG-augmented prompt to the LLM.
  3. Records question, retrieved evidence, generated answer, expected metadata,
     API errors, and timing.

Saved outputs (in ``outputs/``)
---------------------------------
  eval_rag_responses_<timestamp>.csv  – full results per question
  eval_rag_summary_<timestamp>.txt    – aggregate run metadata

Usage
-----
    # Retrieve-only (no LLM call) — FREE, offline
    python -m src.eval.rag_eval --retrieve-only

    # Dry-run (no LLM call, shows what would be sent)
    python -m src.eval.rag_eval --dry-run

    # Full run — requires API key; uses credits
    python -m src.eval.rag_eval --limit 5

IMPORTANT: Heuristic keyword scores are informational only.
Human review is required before citing any figure as an official benchmark.
"""

import argparse
import csv
import os
import sys
import textwrap
import time
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.rag.loader import load_chunks
from src.rag.retriever import build_index, retrieve, format_evidence

QUESTIONS_CSV = ROOT / "practice" / "questions.csv"
OUTPUTS_DIR = ROOT / "outputs"
DATASET_DATE = "20 October 2026"


def _load_questions(limit: int | None) -> list[dict]:
    with QUESTIONS_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def _check_keyword(response: str, field_value: str, sep: str = "/") -> str:
    """Heuristic keyword check. Returns PASS/FAIL/N/A."""
    if not field_value.strip():
        return "N/A"
    alternatives = [a.strip().lower() for a in field_value.split(sep)]
    return "PASS" if any(a in response.lower() for a in alternatives) else "FAIL"


def run_evaluation(
    dry_run: bool = False,
    retrieve_only: bool = False,
    limit: int | None = None,
    top_k: int = 5,
) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = _load_questions(limit)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_csv = OUTPUTS_DIR / f"eval_rag_responses_{timestamp}.csv"
    out_summary = OUTPUTS_DIR / f"eval_rag_summary_{timestamp}.txt"

    provider = os.getenv("LLM_PROVIDER", "gemini")

    # Build index once (offline, fast)
    print("Building retrieval index…", end=" ", flush=True)
    t0 = time.time()
    chunks = load_chunks()
    index = build_index(chunks)
    print(f"done ({len(chunks)} chunks, {time.time()-t0:.2f}s)")

    # Optionally load RAG agent (needs API key)
    rag_agent = None
    if not dry_run and not retrieve_only:
        from src.agent.rag_agent import RAGAgent
        rag_agent = RAGAgent(top_k=top_k)

    fieldnames = [
        "id", "language", "intent", "expected_decision", "answerable",
        "question",
        "retrieved_sections",      # section IDs retrieved
        "retrieved_scores",        # BM25 scores
        "n_retrieved",
        "model_response",
        "error",
        "latency_s",
        "heuristic_must_include",
        "heuristic_must_not_include",
    ]

    results = []
    api_errors = 0

    mode_label = "retrieve-only" if retrieve_only else ("dry-run" if dry_run else "full")
    print(
        f"RAG evaluation | mode={mode_label} | provider={provider} | "
        f"questions={len(questions)} | top_k={top_k}"
    )

    for i, row in enumerate(questions, 1):
        qid = row["id"]
        question = row["question"]
        print(f"  [{i}/{len(questions)}] {qid}: {question[:70]}…")

        error = model_response = ""
        retrieved_sections = retrieved_scores = ""
        n_retrieved = 0
        latency_s = 0.0

        t_start = time.time()

        # --- Retrieval (always offline) ---
        ret_results = retrieve(query=question, index=index, top_k=top_k)
        n_retrieved = len(ret_results)
        retrieved_sections = " ; ".join(r.chunk.section_id or r.chunk.doc_id for r in ret_results)
        retrieved_scores = " ; ".join(str(r.score) for r in ret_results)

        # --- LLM call (optional) ---
        if retrieve_only or dry_run:
            model_response = (
                f"[{mode_label.upper()}] Evidence: {retrieved_sections or 'none'}"
            )
        else:
            try:
                rag_agent.reset()  # independent per-question evaluation
                resp = rag_agent.chat(question)
                model_response = resp.answer
            except Exception as exc:
                error = str(exc)
                model_response = ""
                api_errors += 1
                print(f"    ERROR: {error}")

        latency_s = round(time.time() - t_start, 3)

        mi_check = _check_keyword(model_response, row.get("must_include", ""))
        mn_check = _check_keyword(model_response, row.get("must_not_include", ""))

        results.append({
            "id": qid,
            "language": row.get("language", ""),
            "intent": row.get("intent", ""),
            "expected_decision": row.get("expected_decision", ""),
            "answerable": row.get("answerable", ""),
            "question": question,
            "retrieved_sections": retrieved_sections,
            "retrieved_scores": retrieved_scores,
            "n_retrieved": n_retrieved,
            "model_response": model_response,
            "error": error,
            "latency_s": latency_s,
            "heuristic_must_include": mi_check,
            "heuristic_must_not_include": mn_check,
        })

    # Write CSV
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Aggregate heuristics (informational)
    mi_pass = sum(1 for r in results if r["heuristic_must_include"] == "PASS")
    mi_fail = sum(1 for r in results if r["heuristic_must_include"] == "FAIL")
    mn_pass = sum(1 for r in results if r["heuristic_must_not_include"] == "PASS")
    mn_fail = sum(1 for r in results if r["heuristic_must_not_include"] == "FAIL")
    avg_ret = sum(r["n_retrieved"] for r in results) / len(results) if results else 0

    summary = textwrap.dedent(f"""
        ZentroAI RAG Evaluation Summary
        =================================
        Timestamp      : {timestamp}
        Mode           : {mode_label}
        Provider       : {provider}
        Dataset date   : {DATASET_DATE}
        Questions run  : {len(questions)}
        top_k          : {top_k}
        API errors     : {api_errors}
        Avg retrieved  : {avg_ret:.1f} chunks/question

        Heuristic keyword checks (INFORMATIONAL – not authoritative scores)
        -------------------------------------------------------------------
        must_include  PASS: {mi_pass}  FAIL: {mi_fail}  N/A: {len(questions)-mi_pass-mi_fail}
        must_not_incl PASS: {mn_pass}  FAIL: {mn_fail}  N/A: {len(questions)-mn_pass-mn_fail}

        Output file: {out_csv}

        NOTE: Heuristic keyword matching is a rough proxy.
        Human review of the CSV columns is required before reporting official scores.
    """).strip()

    with out_summary.open("w", encoding="utf-8") as f:
        f.write(summary)

    print()
    print(summary)
    print(f"\nFull results saved to: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG agent on practice questions."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show retrieved evidence without LLM calls.")
    parser.add_argument("--retrieve-only", action="store_true",
                        help="Run retrieval only; skip LLM entirely.")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Only evaluate first N questions.")
    parser.add_argument("--top-k", type=int, default=5, metavar="K",
                        help="Number of evidence chunks to retrieve (default 5).")
    args = parser.parse_args()
    run_evaluation(
        dry_run=args.dry_run,
        retrieve_only=args.retrieve_only,
        limit=args.limit,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
