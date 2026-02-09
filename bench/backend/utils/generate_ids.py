from __future__ import annotations

import uuid
import hashlib
from typing import Callable, Optional


def _generate_with_prefix(prefix: str, length: int = 8) -> str:
    """Generate an ID using the given prefix and hex token."""
    token = uuid.uuid4().hex[:length]
    return f"{prefix}_{token}"


def _deterministic_token(
    prefix: str,
    seed: int,
    tool_name: str,
    call_index: int,
    length: int = 8,
) -> str:
    """
    Deterministically generate a hex token based on tool call context.
    Uses seed + tool_name + call_index + prefix to avoid collisions across tools.
    """
    payload = f"{seed}:{tool_name}:{call_index}:{prefix}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def generate_transaction_id(
    seed: Optional[int] = None,
    tool_name: Optional[str] = None,
    call_index: Optional[int] = None,
) -> str:
    """Transaction ID: txn_XXXXXXXX."""
    if seed is not None and tool_name is not None and call_index is not None:
        token = _deterministic_token("txn", seed, tool_name, call_index)
        return f"txn_{token}"
    return _generate_with_prefix("txn")


def generate_appointment_id(
    seed: Optional[int] = None,
    tool_name: Optional[str] = None,
    call_index: Optional[int] = None,
) -> str:
    """Appointment ID: appt_XXXXXXXX."""
    if seed is not None and tool_name is not None and call_index is not None:
        token = _deterministic_token("appt", seed, tool_name, call_index)
        return f"appt_{token}"
    return _generate_with_prefix("appt")


def generate_care_plan_id(
    seed: Optional[int] = None,
    tool_name: Optional[str] = None,
    call_index: Optional[int] = None,
) -> str:
    """Care plan ID: cp_XXXXXXXX."""
    if seed is not None and tool_name is not None and call_index is not None:
        token = _deterministic_token("cp", seed, tool_name, call_index)
        return f"cp_{token}"
    return _generate_with_prefix("cp")


def generate_session_id(
    seed: Optional[int] = None,
    tool_name: Optional[str] = None,
    call_index: Optional[int] = None,
) -> str:
    """Session record ID: session_XXXXXXXX."""
    if seed is not None and tool_name is not None and call_index is not None:
        token = _deterministic_token("session", seed, tool_name, call_index)
        return f"session_{token}"
    return _generate_with_prefix("session")


def generate_meal_id(
    seed: Optional[int] = None,
    tool_name: Optional[str] = None,
    call_index: Optional[int] = None,
) -> str:
    """Meal record ID: meal_XXXXXXXX."""
    if seed is not None and tool_name is not None and call_index is not None:
        token = _deterministic_token("meal", seed, tool_name, call_index)
        return f"meal_{token}"
    return _generate_with_prefix("meal")


def generate_cart_item_id(
    seed: Optional[int] = None,
    tool_name: Optional[str] = None,
    call_index: Optional[int] = None,
) -> str:
    """Cart item ID: cart_XXXXXXXX."""
    if seed is not None and tool_name is not None and call_index is not None:
        token = _deterministic_token("cart", seed, tool_name, call_index)
        return f"cart_{token}"
    return _generate_with_prefix("cart")


def generate_reminder_id(
    seed: Optional[int] = None,
    tool_name: Optional[str] = None,
    call_index: Optional[int] = None,
) -> str:
    """Reminder ID: r_XXXXXXXX."""
    if seed is not None and tool_name is not None and call_index is not None:
        token = _deterministic_token("r", seed, tool_name, call_index)
        return f"r_{token}"
    return _generate_with_prefix("r")


def generate_plot_id(
    seed: Optional[int] = None,
    tool_name: Optional[str] = None,
    call_index: Optional[int] = None,
) -> str:
    """Plot ID token."""
    if seed is not None and tool_name is not None and call_index is not None:
        return _deterministic_token("plot", seed, tool_name, call_index, length=16)
    return uuid.uuid4().hex[:16]


def generate_note_id(
    seed: Optional[int] = None,
    tool_name: Optional[str] = None,
    call_index: Optional[int] = None,
) -> str:
    """Note ID: note_XXXXXXXX."""
    if seed is not None and tool_name is not None and call_index is not None:
        token = _deterministic_token("note", seed, tool_name, call_index)
        return f"note_{token}"
    return _generate_with_prefix("note")

def generate_health_provider_id(
    seed: Optional[int] = None,
    tool_name: Optional[str] = None,
    call_index: Optional[int] = None,
) -> str:
    if seed is not None and tool_name is not None and call_index is not None:
        token = _deterministic_token("provider", seed, tool_name, call_index)
        return f"provider_{token}"
    return _generate_with_prefix("provider")