# src/eval/metrics.py
"""Metric computation for the ZentroAI evaluation harness.

All functions operate on lists of result dicts produced by the evaluators.

IMPORTANT: keyword-based metrics are PROXY metrics only.
They cannot prove semantic correctness. Every section is clearly labelled
"INFORMATIONAL" to distinguish automated proxies from human judgements.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Keyword helpers (shared across all evaluators)
# ---------------------------------------------------------------------------

def check_must_include(response: str, field: str, sep: str = "/") -> str:
    """Return PASS / FAIL / N/A.

    ``field`` may contain ``sep``-separated alternatives; any one suffices.
    """
    if not field.strip():
        return "N/A"
    resp_lower = response.lower()
    alternatives = [a.strip().lower() for a in field.split(sep)]
    return "PASS" if any(a and a in resp_lower for a in alternatives) else "FAIL"


def check_must_not_include(response: str, field: str, sep: str = "/") -> str:
    """Return PASS (not present) / FAIL (present) / N/A."""
    if not field.strip():
        return "N/A"
    resp_lower = response.lower()
    banned = [b.strip().lower() for b in field.split(sep)]
    return "FAIL" if any(b and b in resp_lower for b in banned) else "PASS"


# ---------------------------------------------------------------------------
# Retrieval recall
# ---------------------------------------------------------------------------

def check_retrieval_recall(retrieved_sections: str, gold_evidence: str, sep: str = ";") -> str:
    """Check whether gold evidence sections appear in retrieved sections.

    Returns PASS / PARTIAL / FAIL / N/A.

    ``gold_evidence`` may contain ``sep``-separated IDs such as
    ``"REF-2 ; KB-00.1"``. We check each gold ID appears as a substring of
    ``retrieved_sections``.  This is a *lexical* proxy; it does not verify
    semantic relevance.
    """
    gold = gold_evidence.strip()
    if not gold:
        return "N/A"
    gold_ids = [g.strip() for g in gold.split(sep) if g.strip()]
    if not gold_ids:
        return "N/A"
    retrieved_lower = retrieved_sections.lower()
    hits = sum(1 for gid in gold_ids if gid.lower() in retrieved_lower)
    if hits == len(gold_ids):
        return "PASS"
    elif hits > 0:
        return "PARTIAL"
    return "FAIL"


# ---------------------------------------------------------------------------
# Tool-call matching
# ---------------------------------------------------------------------------

def check_tool_call(actual_call: str, expected_call: str) -> str:
    """Return PASS / FAIL / N/A for tool-call correctness.

    Comparison is normalised: whitespace and case are ignored.
    ``expected_call`` format: ``lookup_order(order_id=89782837)``
    or ``create_ticket()``.
    If either is absent, returns N/A.
    """
    exp = expected_call.strip()
    act = actual_call.strip()
    if not exp:
        return "N/A" if not act else "UNEXPECTED"
    if not act:
        return "FAIL"
    # Normalise and compare
    def _norm(s: str) -> str:
        return re.sub(r"\s+", "", s.lower())
    return "PASS" if _norm(act) == _norm(exp) else "FAIL"


# ---------------------------------------------------------------------------
# Aggregate summary computation
# ---------------------------------------------------------------------------

def _rate(counts: dict[str, int], key: str, total: int) -> str:
    """Format rate as 'n/total (pct%)'."""
    n = counts.get(key, 0)
    pct = 100 * n / total if total else 0
    return f"{n}/{total} ({pct:.1f}%)"


def summarise_questions(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics over question-level results."""
    total = len(results)
    if not total:
        return {}

    mi = {"PASS": 0, "FAIL": 0, "N/A": 0}
    mn = {"PASS": 0, "FAIL": 0, "N/A": 0}
    ret = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "N/A": 0}
    errors = 0
    latencies = []

    for r in results:
        mi[r.get("heuristic_must_include", "N/A")] = mi.get(r.get("heuristic_must_include", "N/A"), 0) + 1
        mn[r.get("heuristic_must_not_include", "N/A")] = mn.get(r.get("heuristic_must_not_include", "N/A"), 0) + 1
        ret_key = r.get("retrieval_recall", "N/A")
        ret[ret_key] = ret.get(ret_key, 0) + 1
        if r.get("error"):
            errors += 1
        lat = r.get("latency_s")
        if lat is not None:
            latencies.append(float(lat))

    avg_lat = round(sum(latencies) / len(latencies), 3) if latencies else None
    answerable_fail = sum(
        1 for r in results
        if r.get("answerable") == "False" and r.get("heuristic_must_include") == "FAIL"
    )

    return {
        "total": total,
        "api_errors": errors,
        "avg_latency_s": avg_lat,
        "must_include": mi,
        "must_not_include": mn,
        "retrieval_recall": ret,
        "unanswerable_refused": answerable_fail,
    }


def summarise_conversations(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics over conversation-turn results."""
    total = len(results)
    if not total:
        return {}

    tc = {"PASS": 0, "FAIL": 0, "N/A": 0, "UNEXPECTED": 0}
    mi = {"PASS": 0, "FAIL": 0, "N/A": 0}
    mn = {"PASS": 0, "FAIL": 0, "N/A": 0}
    ret = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "N/A": 0}
    errors = 0
    latencies = []

    for r in results:
        tc_key = r.get("tool_call_match", "N/A")
        tc[tc_key] = tc.get(tc_key, 0) + 1
        mi_key = r.get("heuristic_must_include", "N/A")
        mi[mi_key] = mi.get(mi_key, 0) + 1
        mn_key = r.get("heuristic_must_not_include", "N/A")
        mn[mn_key] = mn.get(mn_key, 0) + 1
        ret_key = r.get("retrieval_recall", "N/A")
        ret[ret_key] = ret.get(ret_key, 0) + 1
        if r.get("error"):
            errors += 1
        lat = r.get("latency_s")
        if lat is not None:
            latencies.append(float(lat))

    avg_lat = round(sum(latencies) / len(latencies), 3) if latencies else None

    # Verification gate correctness: turns where verification was expected
    verif_turns = [r for r in results if "verification_value" in r.get("expected_tool_calls", "")]
    verif_correct = sum(1 for r in verif_turns if r.get("tool_call_match") == "PASS")

    # Escalation correctness: turns where create_ticket was expected
    esc_turns = [r for r in results if "create_ticket" in r.get("expected_tool_calls", "")]
    esc_correct = sum(1 for r in esc_turns if r.get("tool_call_match") == "PASS")

    return {
        "total_turns": total,
        "api_errors": errors,
        "avg_latency_s": avg_lat,
        "tool_call_match": tc,
        "must_include": mi,
        "must_not_include": mn,
        "retrieval_recall": ret,
        "verification_turns": len(verif_turns),
        "verification_correct": verif_correct,
        "escalation_turns": len(esc_turns),
        "escalation_correct": esc_correct,
    }
