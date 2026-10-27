# src/tools/order_tools.py
"""Thin, typed wrappers around the capstone's ``tools/tools.py``.

Why a wrapper?
- Keeps the import path stable even as the project structure evolves.
- Makes mocking in tests straightforward: tests patch THIS module's names.
- Documents the exact signatures and return schemas without repeating logic.

The original ``tools/tools.py`` is NEVER modified.  All data files in
``data/`` are read-only.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so ``import tools`` resolves to
# tools/__init__.py at the repo root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import lookup_order as _raw_lookup, create_ticket as _raw_ticket


# ---------------------------------------------------------------------------
# lookup_order
# ---------------------------------------------------------------------------

def lookup_order_unverified(order_id: str) -> dict:
    """Request an order without supplying verification.

    Always returns ``{'verification_required': True, 'order_id': <digits>,
    'message': ...}`` with NO order details disclosed — regardless of whether
    the order_id exists.

    Parameters
    ----------
    order_id:
        Raw order ID string.  Non-digit characters are stripped by the tool.

    Returns
    -------
    dict
        Keys: ``verification_required`` (True), ``order_id``, ``message``.
    """
    return _raw_lookup(order_id)


def lookup_order_verified(order_id: str, verification_value: str) -> dict:
    """Supply a verification value (4-digit phone suffix or 6-digit pincode).

    On success returns the full order payload:
        ``found=True, verified=True, <all order fields>, items=[...],
        customer_first_name=..., is_plus=...``.
    On failure returns ``{'verified': False, 'verification_required': True,
    'message': 'Verification failed or order not found.'}``.

    The caller must NEVER disclose ``verification_value`` back to the customer.
    """
    return _raw_lookup(order_id, verification_value=verification_value)


# ---------------------------------------------------------------------------
# create_ticket
# ---------------------------------------------------------------------------

def create_support_ticket(
    summary: str,
    order_id: str | None = None,
    reason: str | None = None,
    intent: str | None = None,
) -> dict:
    """Log a human-handover ticket.

    Parameters
    ----------
    summary:
        Must be at least 20 characters and include the issue, what was
        checked, and the reason for escalation.  Shorter summaries are
        rejected by the tool with ``created=False``.
    order_id:
        Order ID string (optional).
    reason:
        Short machine-readable reason code, e.g. ``high_value_damage``.
    intent:
        Detected conversation intent, e.g. ``damaged_item``.

    Returns
    -------
    dict
        On success: ``{'created': True, 'ticket_id': 'TKT...'}``.
        On failure: ``{'created': False, 'error': '...'}``.

    IMPORTANT: Only tell the customer a ticket was created if ``created``
    is ``True``.  Never invent ticket IDs.
    """
    return _raw_ticket(summary=summary, order_id=order_id,
                       reason=reason, intent=intent)
