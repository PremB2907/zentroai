# src/rag/document.py
"""Dataclass representing a single indexed chunk from a KB document.

Each chunk corresponds to one markdown section delimited by a heading with a
``[SECTION-ID]`` marker.  Metadata from the YAML frontmatter is carried
alongside so the retriever can apply policy-precedence rules at query time.
"""

from dataclasses import dataclass, field


# Doc-type priority (lower number = higher priority).
# Source: kb/00_about_this_knowledge_base.md [KB-00.1]
DOC_TYPE_PRIORITY: dict[str, int] = {
    "policy": 1,
    "agent_guide": 2,
    "help": 3,
    "faq": 4,
    "marketing": 5,
    "meta": 0,  # KB meta docs are authoritative on process rules
}


@dataclass
class Chunk:
    """A retrievable unit of text from the knowledge base.

    Attributes
    ----------
    doc_id : str
        The ``doc_id`` from the YAML frontmatter (e.g. ``POL-RET``).
    section_id : str
        The section identifier extracted from the heading, e.g. ``RET-2``.
        Empty string when a chunk has no ``[ID]`` marker (intro paragraphs).
    title : str
        Section heading text (without the ``[ID]`` part).
    text : str
        Full text of the section, including the heading.
    doc_type : str
        Document type: ``policy``, ``agent_guide``, ``help``, ``faq``,
        ``marketing``, or ``meta``.
    status : str
        ``current`` or ``superseded``.
    effective_from : str
        ISO date string, e.g. ``2026-08-01``.
    source_file : str
        Relative path from the project root, e.g. ``kb/pol_returns_v4.md``.
    priority : int
        Lower is higher priority; derived from ``doc_type`` via
        ``DOC_TYPE_PRIORITY``.
    superseded_by : str
        ``doc_id`` of the document that supersedes this one (if any).
    """

    doc_id: str
    section_id: str
    title: str
    text: str
    doc_type: str
    status: str
    effective_from: str
    source_file: str
    priority: int = field(init=False)
    superseded_by: str = ""

    def __post_init__(self) -> None:
        self.priority = DOC_TYPE_PRIORITY.get(self.doc_type, 99)

    @property
    def is_superseded(self) -> bool:
        return self.status == "superseded"

    @property
    def citation(self) -> str:
        """Short human-readable citation for use in LLM prompts."""
        if self.section_id:
            return f"[{self.section_id}] ({self.source_file})"
        return f"[{self.doc_id}] ({self.source_file})"

    def __repr__(self) -> str:
        return (
            f"Chunk(section_id={self.section_id!r}, doc_type={self.doc_type!r}, "
            f"status={self.status!r}, priority={self.priority})"
        )
