# src/agent/state.py
"""Conversation state for the Phase 3 dialogue manager.

All security-sensitive flags are set by Python code, NEVER by the LLM.
The LLM receives a read-only summary of this state as context.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationState:
    """Per-session state maintained by the DialogueManager.

    Attributes
    ----------
    order_id : str | None
        Current order ID in context (8-digit string, preserved as str).
    verified : bool
        ``True`` ONLY when ``lookup_order`` returned ``verified=True``.
        Set by Python; never inferred from LLM output.
    verification_attempts : int
        Number of *failed* verification attempts so far.
        Policy allows 1 retry (max 2 total attempts).
    order_data : dict | None
        Full order payload from ``lookup_order``.  Populated only when
        ``verified`` is ``True``.  Never populated from LLM-generated text.
    awaiting_verification : bool
        ``True`` after we have asked the customer for verification digits.
    pending_clarification : str | None
        What the agent is waiting for (e.g. "order_id", "payment_method").
    escalation_done : bool
        ``True`` only when ``create_ticket`` returned ``created=True``.
    ticket_id : str | None
        Ticket ID returned by the tool on success; never invented.
    turn : int
        1-based turn counter within this session.
    """

    # Order context
    order_id: str | None = None
    verified: bool = False
    verification_attempts: int = 0
    order_data: dict | None = None
    awaiting_verification: bool = False

    # Conversation tracking
    pending_clarification: str | None = None

    # Escalation
    escalation_done: bool = False
    ticket_id: str | None = None

    # Turn counter
    turn: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def verification_exhausted(self) -> bool:
        """True when the caller has used all allowed attempts (1 retry)."""
        return self.verification_attempts >= 2

    # ------------------------------------------------------------------
    # State transitions (called only from Python control code)
    # ------------------------------------------------------------------

    def mark_verified(self, order_data: dict) -> None:
        """Record a successful verification.  Must only be called when
        the tool returned ``verified=True``."""
        self.verified = True
        self.order_data = order_data
        self.awaiting_verification = False
        self.pending_clarification = None

    def record_failed_verification(self) -> None:
        """Record one failed attempt."""
        self.verification_attempts += 1
        if self.verification_exhausted:
            self.awaiting_verification = False

    def set_escalated(self, ticket_id: str) -> None:
        """Record a successful escalation.  Must only be called when
        ``create_ticket`` returned ``created=True``."""
        self.escalation_done = True
        self.ticket_id = ticket_id

    def reset(self) -> None:
        """Full reset — clears all state.  Use at session end or /reset."""
        self.__init__()

    # ------------------------------------------------------------------
    # Serialisation helpers (for context injection into LLM prompt)
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a compact, human-readable state summary for LLM context.

        This is informational context — not an instruction the LLM can
        act upon to change its security behaviour.
        """
        lines = [
            f"turn={self.turn}",
            f"order_id={self.order_id or 'none'}",
            f"verified={self.verified}",
            f"verification_attempts={self.verification_attempts}",
            f"verification_exhausted={self.verification_exhausted}",
            f"awaiting_verification={self.awaiting_verification}",
            f"escalation_done={self.escalation_done}",
            f"ticket_id={self.ticket_id or 'none'}",
            f"pending_clarification={self.pending_clarification or 'none'}",
        ]
        return " | ".join(lines)
