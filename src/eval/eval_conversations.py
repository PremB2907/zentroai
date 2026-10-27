# src/eval/eval_conversations.py
"""Conversation-level evaluator (Config C – full DialogueManager).

Reads ``practice/conversations.csv``.  Each conversation is replayed in
order, maintaining dialogue state across turns.

For every turn records:
  conversation_id, turn, scenario, language, order_id,
  customer_message, expected_decision,
  expected_tool_calls, actual_tool_calls, tool_call_match,
  gold_evidence, retrieved_sections, retrieval_recall,
  model_response, error, latency_s,
  heuristic_must_include, heuristic_must_not_include,
  order_revealed, escalation_done, ticket_id,
  needs_manual_review

Multi-turn context: one DialogueManager instance is shared across all turns
of a single conversation, then reset before the next conversation.

Tool calls are intercepted using mock wrappers so the real dataset files are
NOT modified and no tickets are written during dry-run / retrieve-only mode.
In LIVE mode the real tools are called (tickets ARE written to
tools/tickets_created.csv as per the capstone spec).

Heuristic metrics are INFORMATIONAL proxies only.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import textwrap
from collections import defaultdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.eval.metrics import (
    check_must_include,
    check_must_not_include,
    check_retrieval_recall,
    check_tool_call,
    summarise_conversations,
)

CONVERSATIONS_CSV = ROOT / "practice" / "conversations.csv"
EVAL_DIR = ROOT / "eval_outputs"
DATASET_DATE = "20 October 2026"


# ---------------------------------------------------------------------------
# Tool-call interceptor
# ---------------------------------------------------------------------------

class ToolCallRecorder:
    """Records tool calls made during a single conversation turn.

    In dry-run / retrieve-only mode it also patches the real tool functions
    so no tickets are written to disk.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._patches: list[Any] = []

    def record_lookup_unverified(self, order_id: str) -> dict:
        self.calls.append(f"lookup_order(order_id={order_id})")
        # Return the real tool's unverified response shape
        return {
            "verification_required": True,
            "order_id": "".join(ch for ch in str(order_id) if ch.isdigit()),
            "message": "Ask for the registered phone last 4 digits OR delivery pincode.",
        }

    def record_lookup_verified(self, order_id: str, verification_value: str) -> dict:
        self.calls.append(
            f"lookup_order(order_id={order_id}, verification_value={verification_value})"
        )
        # In dry-run mode return a failed-verification response; the
        # dialogue manager handles it gracefully.
        return {"verified": False, "verification_required": True,
                "message": "Verification failed or order not found."}

    def record_create_ticket(
        self,
        summary: str,
        order_id: str | None = None,
        reason: str | None = None,
        intent: str | None = None,
    ) -> dict:
        self.calls.append("create_ticket()")
        # Return a synthetic success (no disk write in dry-run)
        return {"created": True, "ticket_id": "TKT-DRYRUN-001"}

    def latest_call(self) -> str:
        return self.calls[-1] if self.calls else ""

    def reset_turn(self) -> None:
        self.calls.clear()


def _normalise_tool_call(raw: str) -> str:
    """Strip whitespace from a tool call string for comparison."""
    return re.sub(r"\s+", "", raw)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_conversations(limit_convs: int | None = None) -> dict[str, list[dict]]:
    """Return an ordered dict of conversation_id → [sorted turns]."""
    with CONVERSATIONS_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    convs: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        convs[row["conversation_id"]].append(row)

    # Sort turns within each conversation
    for cid in convs:
        convs[cid].sort(key=lambda r: int(r["turn"]))

    if limit_convs:
        items = list(convs.items())[:limit_convs]
        return dict(items)
    return dict(convs)


# ---------------------------------------------------------------------------
# Live runner (real tools, real LLM)
# ---------------------------------------------------------------------------

def _run_live_conversation(
    turns: list[dict],
    top_k: int,
    recorder: ToolCallRecorder,
) -> list[dict]:
    """Run one conversation through the real DialogueManager."""
    from src.agent.dialogue_manager import DialogueManager
    from src.llm.base import ProviderError

    dm = DialogueManager(top_k=top_k)
    provider_used = getattr(dm._provider, "PROVIDER_NAME", "unknown")
    turn_results = []

    for turn_row in turns:
        recorder.reset_turn()
        message = turn_row["customer_message"]

        t0 = time.monotonic()
        try:
            resp = dm.chat(message)
            latency_s = round(time.monotonic() - t0, 3)
            response = resp.answer
            # Catch legacy error strings that survived the old architecture.
            if response.startswith("[Error communicating") or response.startswith("[Error from LLM"):
                error = response
                response = ""
                is_api_error = True
            else:
                error = ""
                is_api_error = False
            retrieved = " ; ".join(
                (r.chunk.section_id or r.chunk.doc_id) for r in resp.retrieval_results
            )
            order_revealed = resp.order_revealed
            esc_done = resp.escalation_done
            ticket_id = resp.ticket_id or ""
        except ProviderError as exc:
            latency_s = round(time.monotonic() - t0, 3)
            response = ""
            retrieved = ""
            order_revealed = False
            esc_done = False
            ticket_id = ""
            error = exc.safe_message
            is_api_error = True
            provider_used = exc.provider_name
        except Exception as exc:
            latency_s = round(time.monotonic() - t0, 3)
            response = ""
            retrieved = ""
            order_revealed = False
            esc_done = False
            ticket_id = ""
            error = str(exc)
            is_api_error = True

        actual_tc = recorder.latest_call()
        turn_results.append(
            _build_turn_result(
                turn_row, response, retrieved, actual_tc,
                order_revealed, esc_done, ticket_id, error,
                latency_s, is_api_error, provider_used,
            )
        )

    return turn_results


def _infer_tool_call_from_state(dm_state_before: dict, dm_state_after: dict) -> str:
    """Infer what tool was called by comparing state snapshots."""
    if dm_state_after.get("verified") and not dm_state_before.get("verified"):
        order_id = dm_state_after.get("order_id", "")
        return f"lookup_order(order_id={order_id}, verification_value=<verified>)"
    if dm_state_after.get("awaiting_verification") and not dm_state_before.get("awaiting_verification"):
        order_id = dm_state_after.get("order_id", "")
        return f"lookup_order(order_id={order_id})"
    if dm_state_after.get("escalation_done") and not dm_state_before.get("escalation_done"):
        return "create_ticket()"
    return ""


# ---------------------------------------------------------------------------
# Dry-run / retrieve-only runner (mocked tools)
# ---------------------------------------------------------------------------

def _run_mock_conversation(
    turns: list[dict],
    top_k: int,
    recorder: ToolCallRecorder,
    retrieve_only: bool = False,
) -> list[dict]:
    """Replay a conversation with mocked LLM and tools."""
    from unittest.mock import MagicMock, patch

    mock_provider = MagicMock()
    mock_provider.PROVIDER_NAME = "ollama"
    mock_provider.generate = MagicMock(return_value="[MOCK RESPONSE]")

    # Patch LLM provider and tools
    with patch("src.llm.provider_factory._instantiate", return_value=mock_provider), \
         patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}), \
         patch("src.agent.dialogue_manager.lookup_order_unverified",
               side_effect=recorder.record_lookup_unverified), \
         patch("src.agent.dialogue_manager.lookup_order_verified",
               side_effect=recorder.record_lookup_verified), \
         patch("src.agent.dialogue_manager.create_support_ticket",
               side_effect=recorder.record_create_ticket):

        from src.agent.dialogue_manager import DialogueManager
        dm = DialogueManager(top_k=top_k)
        dm._provider = mock_provider

        turn_results = []
        for turn_row in turns:
            recorder.reset_turn()
            message = turn_row["customer_message"]

            t0 = time.monotonic()
            try:
                resp = dm.chat(message)
                latency_s = round(time.monotonic() - t0, 3)
                response = resp.answer
                retrieved = " ; ".join(
                    (r.chunk.section_id or r.chunk.doc_id) for r in resp.retrieval_results
                )
                order_revealed = resp.order_revealed
                esc_done = resp.escalation_done
                ticket_id = resp.ticket_id or ""
                error = ""
            except Exception as exc:
                latency_s = round(time.monotonic() - t0, 3)
                response = error = str(exc)
                retrieved = ""
                order_revealed = esc_done = False
                ticket_id = ""

            actual_tc = recorder.latest_call()
            turn_results.append(
                _build_turn_result(
                    turn_row, response, retrieved, actual_tc,
                    order_revealed, esc_done, ticket_id, error, latency_s
                )
            )

    return turn_results


# ---------------------------------------------------------------------------
# Result builder (shared)
# ---------------------------------------------------------------------------

def _build_turn_result(
    row: dict,
    response: str,
    retrieved: str,
    actual_tc: str,
    order_revealed: bool,
    esc_done: bool,
    ticket_id: str,
    error: str,
    latency_s: float,
    is_api_error: bool = False,
    provider_used: str = "",
) -> dict:
    expected_tc = row.get("expected_tool_calls", "")
    tc_match = check_tool_call(actual_tc, expected_tc)
    ret_recall = check_retrieval_recall(retrieved, row.get("gold_evidence", ""))
    # Skip heuristic metrics for error turns — they would produce misleading scores.
    if is_api_error:
        mi_check = mn_check = "API_ERROR"
    else:
        mi_check = check_must_include(response, row.get("must_include", ""))
        mn_check = check_must_not_include(response, row.get("must_not_include", ""))

    return {
        "conversation_id": row["conversation_id"],
        "turn": row["turn"],
        "scenario": row.get("scenario", ""),
        "language": row.get("language", ""),
        "order_id": row.get("order_id", ""),
        "customer_message": row["customer_message"],
        "expected_decision": row.get("expected_decision", ""),
        "expected_tool_calls": expected_tc,
        "actual_tool_calls": actual_tc,
        "tool_call_match": tc_match,
        "gold_evidence": row.get("gold_evidence", ""),
        "retrieved_sections": retrieved,
        "retrieval_recall": ret_recall,
        "model_response": response,
        "provider_used": provider_used,
        "error": error,
        "latency_s": latency_s,
        "heuristic_must_include": mi_check,
        "heuristic_must_not_include": mn_check,
        "is_api_error": is_api_error,
        "order_revealed": order_revealed,
        "escalation_done": esc_done,
        "ticket_id": ticket_id,
        "needs_manual_review": "True" if row.get("scenario") in {
            "compensation_demand", "fraud"
        } else "False",
    }


# ---------------------------------------------------------------------------
# Public evaluation runner
# ---------------------------------------------------------------------------

def run_conversation_eval(
    dry_run: bool = False,
    retrieve_only: bool = False,
    limit_convs: int | None = None,
    top_k: int = 5,
    output_dir: Path | None = None,
) -> tuple[Path, dict]:
    """Run the full conversation evaluation (Config C).

    Parameters
    ----------
    dry_run / retrieve_only:
        If True, uses mocked LLM + tools (no API calls, no disk writes).
    limit_convs:
        Limit to first N conversations (all turns are replayed for each).
    top_k:
        KB chunks per retrieval.
    output_dir:
        Output directory. Defaults to ``eval_outputs/``.

    Returns
    -------
    (out_csv_path, summary_dict)
    """
    out_dir = output_dir or EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    conversations = load_conversations(limit_convs)
    configured_provider = os.getenv("LLM_PROVIDER", "ollama")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_csv = out_dir / f"eval_full_conversations_{timestamp}.csv"

    use_mock = dry_run or retrieve_only
    mode_label = "dry-run" if dry_run else ("retrieve-only" if retrieve_only else "live")

    total_turns = sum(len(turns) for turns in conversations.values())
    print(
        f"Conversation eval | config=full | mode={mode_label} | "
        f"provider={configured_provider} | convs={len(conversations)} | turns={total_turns}"
    )

    fieldnames = [
        "conversation_id", "turn", "scenario", "language", "order_id",
        "customer_message", "expected_decision",
        "expected_tool_calls", "actual_tool_calls", "tool_call_match",
        "gold_evidence", "retrieved_sections", "retrieval_recall",
        "model_response", "provider_used", "error", "latency_s",
        "heuristic_must_include", "heuristic_must_not_include",
        "is_api_error", "order_revealed", "escalation_done", "ticket_id",
        "needs_manual_review",
    ]

    all_results: list[dict] = []
    recorder = ToolCallRecorder()
    api_errors = 0

    for conv_id, turns in conversations.items():
        print(f"  Conversation {conv_id} ({len(turns)} turns, scenario={turns[0].get('scenario','?')})")
        if use_mock:
            turn_results = _run_mock_conversation(turns, top_k, recorder, retrieve_only)
        else:
            turn_results = _run_live_conversation(turns, top_k, recorder)

        for r in turn_results:
            if r.get("error"):
                api_errors += 1
                print(f"    ERROR turn {r['turn']}: {r['error']}")
        all_results.extend(turn_results)

    # Write CSV
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    summary = summarise_conversations(all_results)
    summary.update({
        "config": "full",
        "mode": mode_label,
        "configured_provider": configured_provider,
        "provider_priority": "ollama (default) | groq | nvidia | gemini (explicit override)",
        "dataset_date": DATASET_DATE,
        "timestamp": timestamp,
        "conversations": len(conversations),
        "total_turns": total_turns,
        "api_errors": api_errors,
        "top_k": top_k,
        "output_csv": str(out_csv),
    })
    return out_csv, summary
