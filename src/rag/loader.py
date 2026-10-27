# src/rag/loader.py
"""Markdown knowledge-base loader and chunker.

Reads every ``*.md`` file in the ``kb/`` directory, parses the YAML
frontmatter, and splits the document into section-level chunks (one chunk per
``## [SECTION-ID] Heading`` block).

Also loads the product catalog from ``data/products.csv`` for SKU-aware
retrieval.

Design decisions
----------------
* **No external dependencies.**  Pure standard-library: ``re``, ``pathlib``,
  ``csv``.
* **Section granularity.**  Each level-2 (``##``) heading with a ``[ID]``
  marker becomes one chunk.  Level-1 headings (document titles) are captured
  as an intro chunk if they carry meaningful prose before the first ``##``.
* **Metadata preserved.**  Every chunk carries ``doc_id``, ``section_id``,
  ``doc_type``, ``status``, ``effective_from``, ``source_file``.
* **Superseded docs are included** (with ``status="superseded"``) so that
  history-aware questions (e.g. electronics delivered before Aug 2026) can
  still be answered.  The *retriever* applies precedence rules at query time.
* **Product catalog** is loaded once and indexed by SKU for deterministic
  lookup.  Product rows are exposed as pseudo-chunks with provenance labels.
* **Original files are never modified.**
"""

import csv
import re
from pathlib import Path
from typing import Iterator, Dict, Optional

from .document import Chunk

# ---------------------------------------------------------------------------
# YAML frontmatter helpers (stdlib only – no PyYAML needed for this subset)
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_KEY_RE = re.compile(r"^(\w+):\s*(.*)", re.MULTILINE)

# SKU pattern: letters/digits/hyphens, e.g. ELE-0012, FSH-0026
# Matches case-insensitively with word boundaries
_SKU_RE = re.compile(r"\b([A-Z]{2,4}-\d{4})\b", re.IGNORECASE)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
PRODUCTS_CSV = DATA_ROOT / "products.csv"


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return ``(meta_dict, body_text)`` from a Markdown file with YAML frontmatter."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw_meta = m.group(1)
    body = text[m.end():]
    meta: dict[str, str] = {}
    for line_m in _KEY_RE.finditer(raw_meta):
        meta[line_m.group(1).strip()] = line_m.group(2).strip()
    return meta, body


# ---------------------------------------------------------------------------
# Product catalog loader
# ---------------------------------------------------------------------------

# In-memory product catalog cache
_product_catalog_cache: Optional[Dict[str, Dict[str, str]]] = None


def load_product_catalog(products_csv: Path | None = None) -> Dict[str, Dict[str, str]]:
    """Load the product catalog from ``products.csv`` into a dict keyed by SKU.

    Returns a mapping of SKU (uppercase) -> product row dict with all columns.
    The catalog is cached after first load.
    """
    global _product_catalog_cache
    if _product_catalog_cache is not None:
        return _product_catalog_cache

    csv_path = products_csv or PRODUCTS_CSV
    catalog: Dict[str, Dict[str, str]] = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sku = row["sku"].strip().upper()
            catalog[sku] = row
    _product_catalog_cache = catalog
    return catalog


def get_product(sku: str, products_csv: Path | None = None) -> Optional[Dict[str, str]]:
    """Return the product row for the given SKU, or ``None`` if not found.

    SKU matching is case-insensitive.
    """
    catalog = load_product_catalog(products_csv)
    return catalog.get(sku.strip().upper())


def extract_sku(query: str) -> Optional[str]:
    """Extract the first SKU-like token from a query string.

    Returns the SKU in uppercase (e.g. ``ELE-0012``) or ``None`` if no match.
    """
    m = _SKU_RE.search(query)
    if m:
        return m.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Section chunker
# ---------------------------------------------------------------------------

# Matches ``## [SECTION-ID] Heading text`` – section IDs can contain letters,
# digits, dots and hyphens.
_SECTION_RE = re.compile(r"^(#{1,3})\s+(?:\[([^\]]+)\]\s*)?(.+)$", re.MULTILINE)


def _split_into_sections(body: str) -> list[tuple[str, str, str]]:
    """Split document body into ``(section_id, title, text)`` tuples.

    Sections are delimited by ``##`` headings.  Text between the document
    title (``#``) and the first ``##`` is returned as a section with an empty
    ``section_id``.
    """
    # Find all heading positions.
    headings = list(_SECTION_RE.finditer(body))
    if not headings:
        return [("", "", body.strip())]

    sections: list[tuple[str, str, str]] = []

    for i, match in enumerate(headings):
        level = len(match.group(1))
        section_id = (match.group(2) or "").strip()
        title = match.group(3).strip()

        # Collect text from heading start to next heading (or end of body).
        start = match.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        text = body[start:end].strip()

        sections.append((section_id, title, text))

    return sections


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

KB_ROOT = Path(__file__).resolve().parents[2] / "kb"


def load_chunks(kb_dir: Path | None = None) -> list[Chunk]:
    """Load and chunk all Markdown files in ``kb_dir`` (default: ``kb/``).

    Returns a list of :class:`Chunk` objects sorted by:
    1. ``priority`` ascending (policy > agent_guide > help > faq > marketing)
    2. ``effective_from`` descending (newer first within the same type)

    This ordering means that when the retriever iterates results, higher-
    authority, more-recent chunks appear first.
    """
    kb_dir = kb_dir or KB_ROOT
    chunks: list[Chunk] = []

    for md_file in sorted(kb_dir.glob("*.md")):
        raw = md_file.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)

        doc_id = meta.get("doc_id", md_file.stem)
        doc_type = meta.get("doc_type", "unknown")
        status = meta.get("status", "current")
        effective_from = meta.get("effective_from", "")
        superseded_by = meta.get("superseded_by", "")
        source_file = f"kb/{md_file.name}"

        for section_id, title, text in _split_into_sections(body):
            if not text:
                continue
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    section_id=section_id,
                    title=title,
                    text=text,
                    doc_type=doc_type,
                    status=status,
                    effective_from=effective_from,
                    source_file=source_file,
                    superseded_by=superseded_by,
                )
            )

    # Sort: priority ASC, then effective_from DESC (newer = lexicographically
    # larger ISO date string).
    chunks.sort(key=lambda c: (c.priority, c.status == "superseded", c.effective_from), reverse=False)
    # Re-sort effective_from descending within same priority group.
    chunks.sort(key=lambda c: (c.priority, c.status == "superseded", [-ord(ch) for ch in c.effective_from]))

    return chunks


def create_product_chunk(product: Dict[str, str]) -> Chunk:
    """Create a :class:`Chunk` from a product catalog row for evidence injection.

    The chunk carries provenance metadata so the LLM knows it came from the
    product catalog, not a policy document.
    """
    sku = product["sku"]
    title = product["title"]
    assured = product.get("assured", "False")
    warranty_months = product.get("warranty_months", "0")
    return_category = product.get("return_category", "unknown")
    price_inr = product.get("price_inr", "0")
    fulfilled_by = product.get("fulfilled_by", "")
    brand = product.get("brand", "")
    description = product.get("description", "")

    # Build a concise, structured text block for the LLM
    text = (
        f"Product: {title} (SKU: {sku})\n"
        f"Brand: {brand}\n"
        f"Category: {return_category}\n"
        f"Price: ₹{price_inr}\n"
        f"Fulfilled by: {fulfilled_by}\n"
        f"ZentroMart Assured: {assured}\n"
        f"Warranty: {warranty_months} months\n"
        f"Description: {description}"
    )

    # Use the same format as gold_evidence in questions.csv for retrieval recall matching
    section_id = f"products.csv:{sku}"

    return Chunk(
        doc_id="PRODUCT-CATALOG",
        section_id=section_id,
        title=f"Product: {title} (SKU: {sku})",
        text=text,
        doc_type="meta",  # same priority as KB meta docs
        status="current",
        effective_from="",
        source_file="data/products.csv",
        superseded_by="",
    )


def iter_chunks(kb_dir: Path | None = None) -> Iterator[Chunk]:
    """Lazy iterator version of :func:`load_chunks`."""
    yield from load_chunks(kb_dir)
