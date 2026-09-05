"""Ephemeral handling for external advisory signals."""

from __future__ import annotations

from .models import AdvisoryFieldSignal


class AdvisorySignalStore:
    """Keeps the latest device observation; it cannot approve a procedure step."""

    def __init__(self) -> None:
        self._field_signals: dict[str, AdvisoryFieldSignal] = {}

    def record_field_signal(
        self, session_id: str, signal: AdvisoryFieldSignal
    ) -> AdvisoryFieldSignal:
        self._field_signals[session_id] = signal
        return signal

    def latest_field_signal(self, session_id: str) -> AdvisoryFieldSignal | None:
        return self._field_signals.get(session_id)
