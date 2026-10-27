# src/eval/run_eval.py
"""Unified evaluation harness for ZentroAI.

Evaluates three configurations against the practice dataset:
  A  baseline   – LLM-only agent (practice/questions.csv)
  B  rag        – RAG agent, no tools (practice/questions.csv)
  C  full       – DialogueManager: RAG + tools (practice/conversations.csv)

Usage
-----
  # Cost estimate (no API calls, no output files)
  python -m src.eval.run_eval --estimate

  # Full offline dry-run (mocked LLM, real retrieval, no API cost)
  python -m src.eval.run_eval --dry-run

  # Retrieve-only for RAG and full configs (free)
  python -m src.eval.run_eval --retrieve-only

  # Run specific configs
  python -m src.eval.run_eval --dry-run --config baseline rag
  python -m src.eval.run_eval --dry-run --config full

  # Live run — REQUIRES API KEY, CONSUMES CREDITS
  # Must pass --confirm-live to prevent accidental execution
  python -m src.eval.run_eval --confirm-live

  # Quick sanity check (5 questions, 2 conversations)
  python -m src.eval.run_eval --dry-run --limit-q 5 --limit-c 2

Output files land in eval_outputs/ as timestamped CSVs plus a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

EVAL_DIR = ROOT / "eval_outputs"
DATASET_DATE = "20 October 2026"

# ---------------------------------------------------------------------------
# Cost / call estimation
# ---------------------------------------------------------------------------

_Q_ROWS = 52    # questions.csv rows
_C_ROWS = 96    # conversations.csv rows  (turns)
_C_CONVS = 23   # distinct conversations


def print_estimate() -> None:
    print(textwrap.dedent(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║        ZentroAI Evaluation — API Call Estimate           ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Dataset date     : {DATASET_DATE}                ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Config A  Baseline (questions.csv)                      ║
    ║    Questions      : {_Q_ROWS} rows                              ║
    ║    LLM calls/q    : 1 (fresh agent per question)         ║
    ║    Total calls    : {_Q_ROWS} API calls                         ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Config B  RAG (questions.csv)                           ║
    ║    Questions      : {_Q_ROWS} rows                              ║
    ║    Retrieval      : FREE (in-memory BM25, no API)        ║
    ║    LLM calls/q    : 1                                    ║
    ║    Total calls    : {_Q_ROWS} API calls                         ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Config C  Full DialogueManager (conversations.csv)      ║
    ║    Conversations  : {_C_CONVS} distinct                          ║
    ║    Total turns    : {_C_ROWS} rows                              ║
    ║    LLM calls/turn : 1                                    ║
    ║    Total calls    : {_C_ROWS} API calls                         ║
    ║    Tool calls     : ~54 (recorded in CSV, may write      ║
    ║                    tools/tickets_created.csv in LIVE)    ║
    ╠══════════════════════════════════════════════════════════╣
    ║  TOTAL (all 3 configs) : ~{_Q_ROWS + _Q_ROWS + _C_ROWS} API calls              ║
    ║  Per config            :  {_Q_ROWS} / {_Q_ROWS} / {_C_ROWS} calls              ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Gemini free tier  : 15 req/min, 1500/day (gemini-2.0)  ║
    ║  Estimated time    : ~3–5 min at free tier               ║
    ║  Cost at pay tier  : <$0.10 for full run                 ║
    ╚══════════════════════════════════════════════════════════╝

    To run live evaluation:
      python -m src.eval.run_eval --confirm-live

    To run offline (dry-run, no API cost):
      python -m src.eval.run_eval --dry-run
    """).strip())


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def _fmt_counts(d: dict, keys: list[str]) -> str:
    return "  ".join(f"{k}: {d.get(k, 0)}" for k in keys)


def format_summary(summaries: list[dict]) -> str:
    lines = [
        "",
        "══════════════════════════════════════════════════════════",
        f"  ZentroAI Evaluation Summary  ({datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')})",
        f"  Dataset date: {DATASET_DATE}",
        "══════════════════════════════════════════════════════════",
    ]
    for s in summaries:
        cfg = s.get("config", "?").upper()
        mode = s.get("mode", "?")
        lines += [
            "",
            f"  Config {cfg} ({mode})",
            f"  Provider   : {s.get('provider') or s.get('configured_provider','?')}",

        ]
        if "questions" in s:
            lines.append(f"  Questions  : {s['questions']}  (dataset: {s.get('dataset_date','')})")
        if "conversations" in s:
            lines.append(f"  Convs/turns: {s.get('conversations','?')} / {s.get('total_turns','?')}")
        lines.append(f"  API errors : {s.get('api_errors', 0)}")
        if s.get("avg_latency_s") is not None:
            lines.append(f"  Avg latency: {s['avg_latency_s']}s")

        mi = s.get("must_include", {})
        mn = s.get("must_not_include", {})
        ret = s.get("retrieval_recall", {})
        tc = s.get("tool_call_match", {})

        lines += [
            "",
            "  ── PROXY METRICS (INFORMATIONAL — not authoritative) ──",
            f"  must_include  : {_fmt_counts(mi, ['PASS','FAIL','N/A'])}",
            f"  must_not_incl : {_fmt_counts(mn, ['PASS','FAIL','N/A'])}",
        ]
        if any(ret.values()):
            lines.append(
                f"  retrieval     : PASS={ret.get('PASS',0)}  PARTIAL={ret.get('PARTIAL',0)}"
                f"  FAIL={ret.get('FAIL',0)}  N/A={ret.get('N/A',0)}"
            )
        if any(tc.values()):
            lines.append(f"  tool_call     : {_fmt_counts(tc, ['PASS','FAIL','N/A','UNEXPECTED'])}")
        if "verification_turns" in s:
            vt = s["verification_turns"]
            vc = s["verification_correct"]
            lines.append(f"  verif turns   : {vc}/{vt} correct")
        if "escalation_turns" in s:
            et = s["escalation_turns"]
            ec = s["escalation_correct"]
            lines.append(f"  escalation    : {ec}/{et} correct")

        lines.append(f"  Output CSV : {s.get('output_csv','—')}")

    lines += [
        "",
        "  ── METRIC DEFINITIONS ──────────────────────────────────",
        "  must_include   : keyword present in response (proxy only)",
        "  must_not_incl  : banned keyword absent (proxy only)",
        "  retrieval      : gold section IDs found in retrieved set",
        "  tool_call      : normalised string match of tool invocation",
        "  verif turns    : turns with expected verification tool call",
        "  escalation     : turns with expected create_ticket() call",
        "",
        "  Human review required for: semantic correctness, language",
        "  quality, policy compliance, unanswerable-question handling.",
        "══════════════════════════════════════════════════════════",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.eval.run_eval",
        description="ZentroAI unified evaluation harness.",
    )
    p.add_argument(
        "--estimate",
        action="store_true",
        help="Print API call estimate and exit (no evaluation run).",
    )
    p.add_argument(
        "--config",
        nargs="+",
        choices=["baseline", "rag", "full"],
        default=["baseline", "rag", "full"],
        help="Which configs to evaluate (default: all three).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Mock all LLM calls; run retrieval. No API cost.",
    )
    p.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Run retrieval only; skip LLM for RAG and full configs.",
    )
    p.add_argument(
        "--confirm-live",
        action="store_true",
        help="Allow live LLM API calls (required to prevent accidents).",
    )
    p.add_argument(
        "--limit-q",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N questions for configs A and B.",
    )
    p.add_argument(
        "--limit-c",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N conversations for config C.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="KB chunks to retrieve per query (default: 5).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=EVAL_DIR,
        help=f"Output directory (default: {EVAL_DIR}).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.estimate:
        print_estimate()
        return

    # Safety gate: refuse live run without explicit flag
    live = args.confirm_live and not args.dry_run and not args.retrieve_only
    if not args.dry_run and not args.retrieve_only and not args.confirm_live:
        print(
            "\n[STOPPED] Live evaluation requires real API calls.\n"
            "  Run with  --dry-run   for a free offline run.\n"
            "  Run with  --estimate  to see API call counts.\n"
            "  Add  --confirm-live   to proceed with live calls.\n"
        )
        sys.exit(1)

    from src.eval.eval_questions import run_question_eval
    from src.eval.eval_conversations import run_conversation_eval

    configs = args.config
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    all_summaries = []

    # Config A — Baseline
    if "baseline" in configs:
        print("\n─── Config A: Baseline LLM-only ───")
        _, summary = run_question_eval(
            config="baseline",
            dry_run=args.dry_run or args.retrieve_only,
            retrieve_only=False,
            limit=args.limit_q,
            top_k=args.top_k,
            output_dir=out_dir,
        )
        all_summaries.append(summary)

    # Config B — RAG
    if "rag" in configs:
        print("\n─── Config B: RAG (no tools) ───")
        _, summary = run_question_eval(
            config="rag",
            dry_run=args.dry_run,
            retrieve_only=args.retrieve_only,
            limit=args.limit_q,
            top_k=args.top_k,
            output_dir=out_dir,
        )
        all_summaries.append(summary)

    # Config C — Full DialogueManager
    if "full" in configs:
        print("\n─── Config C: Full Dialogue Manager ───")
        _, summary = run_conversation_eval(
            dry_run=args.dry_run,
            retrieve_only=args.retrieve_only,
            limit_convs=args.limit_c,
            top_k=args.top_k,
            output_dir=out_dir,
        )
        all_summaries.append(summary)

    # Print and save summary
    report = format_summary(all_summaries)
    print(report)

    summary_path = out_dir / f"eval_summary_{timestamp}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"Summary JSON saved to: {summary_path}")

    report_path = out_dir / f"eval_report_{timestamp}.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to:       {report_path}")


if __name__ == "__main__":
    main()
