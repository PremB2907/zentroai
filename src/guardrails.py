"""Deterministic answer-quality guardrails for the RAG pipeline.

All functions are pure, side‑effect‑free, and use only the retrieved
evidence that is already available at answer time.  No live LLM calls.
"""

import re
from typing import List, Set

from .rag.retriever import RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[([^\]]+)\]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_WARRANTY_MONTHS_RE = re.compile(r"(\d+)[-\s]?month", re.IGNORECASE)


def _extract_citations(text: str) -> List[str]:
    """Return all citation IDs like 'RET-2' found in ``text``."""
    return _CITATION_RE.findall(text)


def _has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(text))


def _collect_section_ids(results: List[RetrievalResult]) -> Set[str]:
    """All section_ids (or doc_ids when section_id empty) present in retrieved evidence."""
    ids = set()
    for r in results:
        sid = r.chunk.section_id or r.chunk.doc_id
        if sid:
            ids.add(sid)
    return ids


def _product_warranty_months(results: List[RetrievalResult]) -> int | None:
    """Return warranty months from the first product‑catalog chunk, if any."""
    for r in results:
        if r.chunk.section_id and r.chunk.section_id.startswith("products.csv:"):
            # product chunk text contains a line like "Warranty: 12 months"
            m = _WARRANTY_MONTHS_RE.search(r.chunk.text)
            if m:
                return int(m.group(1))
    return None


def _contains_marketplace_restriction(text: str) -> bool:
    """Heuristic: does the answer assert a marketplace‑seller restriction?"""
    return "marketplace seller" in text.lower()


def _marketplace_supported(results: List[RetrievalResult]) -> bool:
    """True if any retrieved chunk explicitly states that ZentroMart Assured does not apply to marketplace sellers."""
    target = "zentromart assured does not apply to marketplace seller"
    for r in results:
        if target in r.chunk.text.lower():
            return True
    return False


# ---------------------------------------------------------------------------
# Public guardrail functions
# ---------------------------------------------------------------------------

def ensure_warranty_duration(answer: str, results: List[RetrievalResult]) -> str:
    """Append warranty‑months if a product chunk supplies it and the answer omits it."""
    months = _product_warranty_months(results)
    if months is None:
        return answer
    # already mentions a month number?
    if _WARRANTY_MONTHS_RE.search(answer):
        return answer
    # append concise notice
    return f"{answer.rstrip()} (warranty: {months} months)."


def validate_citations(answer: str, results: List[RetrievalResult]) -> str:
    """Remove citation markers whose ID is not present in retrieved evidence."""
    allowed = _collect_section_ids(results)
    # replace unsupported citations with empty string
    def repl(m):
        cid = m.group(1)
        return m.group(0) if cid in allowed else ""
    return _CITATION_RE.sub(repl, answer)


def strip_unsupported_marketplace_claim(answer: str, results: List[RetrievalResult]) -> str:
    """If answer asserts a marketplace‑seller restriction but evidence lacks it, drop that sentence."""
    if not _contains_marketplace_restriction(answer):
        return answer
    if _marketplace_supported(results):
        return answer
    # remove the offending sentence (naive split on '.')
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    cleaned = [s for s in sentences if "marketplace seller" not in s.lower()]
    return " ".join(cleaned).strip()


def enforce_language_match(answer: str, question: str) -> str:
    """Very small heuristic: if question contains Devanagari, ensure answer does too.
    If not, prepend a short Hindi apology – deterministic, no LLM."""
    if _has_devanagari(question) and not _has_devanagari(answer):
        return "क्षमा करें, " + answer
    return answer


def apply_guardrails(
    answer: str,
    question: str,
    results: List[RetrievalResult],
) -> str:
    """Run all deterministic guardrails in a fixed order."""
    answer = ensure_warranty_duration(answer, results)
    answer = strip_unsupported_marketplace_claim(answer, results)
    answer = validate_citations(answer, results)
    answer = enforce_language_match(answer, question)
    return answer