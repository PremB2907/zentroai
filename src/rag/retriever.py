# src/rag/retriever.py
"""Lightweight lexical retriever using Python's standard library.

Algorithm
---------
1. Build an inverted index: term → list of (chunk_index, tf) pairs where
   ``tf`` is the normalised term frequency.
2. At query time compute a BM25-lite score:

       score(q, d) = Σ_{t in q} IDF(t) × TF-IDF(t, d)

   where IDF = log((N + 1) / (df + 0.5) + 1)  (smoothed)
   and TF normalised by (0.5 + 0.5 × |d| / avg_dl).

   This gives good recall without NumPy or any ML library.

3. Apply **policy-precedence** rules *after* scoring:
   - Superseded chunks get a penalty multiplier (default 0.4) unless:
     (a) the query itself requests historical data, or
     (b) the chunk is the only evidence for the section.
   - Within equal-score chunks, lower ``priority`` (higher authority) wins.

4. A minimum score threshold (default 0.01) is applied; queries returning
   nothing above the threshold yield an empty list so the agent can say
   "I don't know".

No disk writes, no external packages, no model downloads.
"""

import math
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from .document import Chunk

# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[" + re.escape(string.punctuation) + r"₹]")
_WS = re.compile(r"\s+")

# Very small English stop-word list (keeps common query words intact so
# "return" / "refund" / "days" are still searchable).
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can "
    "i me my we our you your he she it its they them their "
    "and or but not in on at to for of with by from into "
    "this that these those there here all also just "
    "what when where how who which".split()
)


def _tokenise(text: str) -> list[str]:
    """Lower-case, strip punctuation, split, remove stop-words."""
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    tokens = _WS.split(text.strip())
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Index dataclass
# ---------------------------------------------------------------------------

@dataclass
class LexicalIndex:
    """Pre-built inverted index over a list of :class:`Chunk` objects.

    Build with :func:`build_index`.  The index is held entirely in-memory as
    a plain Python dict — no file I/O, no external libraries.
    """

    chunks: list[Chunk]
    # term → {chunk_index: raw_count}
    posting: dict[str, dict[int, int]] = field(default_factory=dict)
    # chunk_index → token count (for TF normalisation)
    doc_lengths: list[int] = field(default_factory=list)
    avg_dl: float = 0.0

    def __len__(self) -> int:
        return len(self.chunks)


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

def build_index(chunks: list[Chunk]) -> LexicalIndex:
    """Build an in-memory inverted index from a list of chunks.

    Parameters
    ----------
    chunks:
        Output of :func:`~src.rag.loader.load_chunks`.
    """
    idx = LexicalIndex(chunks=chunks)
    posting: dict[str, dict[int, int]] = defaultdict(dict)
    doc_lengths: list[int] = []

    for i, chunk in enumerate(chunks):
        tokens = _tokenise(chunk.text)
        tf = Counter(tokens)
        doc_lengths.append(len(tokens))
        for term, count in tf.items():
            posting[term][i] = count

    idx.posting = dict(posting)
    idx.doc_lengths = doc_lengths
    idx.avg_dl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0
    return idx


# ---------------------------------------------------------------------------
# Retrieval result
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single ranked retrieval result."""

    chunk: Chunk
    score: float
    rank: int

    @property
    def citation(self) -> str:
        return self.chunk.citation


# ---------------------------------------------------------------------------
# BM25-lite scorer
# ---------------------------------------------------------------------------

_B = 0.75      # BM25 length-normalisation parameter
_K1 = 1.2     # BM25 term-saturation parameter
_SUPERSEDED_PENALTY = 0.4  # Score multiplier for superseded chunks


def retrieve(
    query: str,
    index: LexicalIndex,
    top_k: int = 5,
    min_score: float = 0.01,
    allow_superseded: bool = False,
) -> list[RetrievalResult]:
    """Retrieve the top-k most relevant chunks for ``query``.

    Parameters
    ----------
    query:
        Natural-language customer question (any language; non-ASCII tokens
        are kept as-is after lower-casing).
    index:
        Pre-built :class:`LexicalIndex`.
    top_k:
        Maximum number of results to return.
    min_score:
        Minimum BM25 score; results below this are discarded.
    allow_superseded:
        If ``True``, superseded chunks are not penalised.  Set to ``True``
        only when the query explicitly references a historical date before the
        supersession date.

    Returns
    -------
    List of :class:`RetrievalResult` sorted by score descending, then by
    chunk priority ascending (higher-authority docs first on ties).
    """
    n_docs = len(index.chunks)
    if n_docs == 0:
        return []

    query_tokens = _tokenise(query)
    if not query_tokens:
        return []

    scores: dict[int, float] = {}

    for term in set(query_tokens):  # deduplicate query terms
        if term not in index.posting:
            continue
        postings = index.posting[term]
        df = len(postings)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

        for chunk_idx, raw_tf in postings.items():
            dl = index.doc_lengths[chunk_idx]
            tf_norm = (raw_tf * (_K1 + 1)) / (
                raw_tf + _K1 * (1 - _B + _B * dl / index.avg_dl)
            )
            scores[chunk_idx] = scores.get(chunk_idx, 0.0) + idf * tf_norm

    if not scores:
        return []

    # Apply superseded-doc penalty
    if not allow_superseded:
        for chunk_idx, score in list(scores.items()):
            if index.chunks[chunk_idx].is_superseded:
                scores[chunk_idx] = score * _SUPERSEDED_PENALTY

    # Filter by minimum score
    filtered = [(idx, sc) for idx, sc in scores.items() if sc >= min_score]

    # Sort: score DESC, then chunk priority ASC (lower = more authoritative)
    filtered.sort(key=lambda x: (-x[1], index.chunks[x[0]].priority))

    results: list[RetrievalResult] = []
    for rank, (chunk_idx, score) in enumerate(filtered[:top_k], start=1):
        results.append(
            RetrievalResult(
                chunk=index.chunks[chunk_idx],
                score=round(score, 4),
                rank=rank,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Convenience: format evidence for the LLM prompt
# ---------------------------------------------------------------------------

def format_evidence(results: Sequence[RetrievalResult], max_chars: int = 4000) -> str:
    """Return a concise evidence block for injection into an LLM prompt.

    Each section is labelled with its citation so the model can reference it.
    Truncates individual chunks if the total exceeds ``max_chars``.
    """
    if not results:
        return "(No relevant evidence found in the knowledge base.)"

    budget = max_chars
    parts: list[str] = []
    for r in results:
        header = f"--- Source: {r.citation} | score={r.score:.3f} ---"
        body = r.chunk.text
        if len(header) + len(body) + 2 > budget:
            body = body[: max(0, budget - len(header) - 5)] + "…"
        parts.append(f"{header}\n{body}")
        budget -= len(parts[-1]) + 1
        if budget <= 0:
            break

    return "\n\n".join(parts)
