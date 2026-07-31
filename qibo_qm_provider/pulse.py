"""Qibolab pulse-level lowering helpers towards QUA-compatible instructions."""

from __future__ import annotations


class QibolabPulseLowerer:
    """Converts simple Qibolab-like pulse dictionaries into QUA-like operations."""

    def lower(self, sequence: list[dict[str, object]]) -> list[str]:
        lowered: list[str] = []
        for pulse in sequence:
            channel = str(pulse.get("channel", ""))
            operation = str(pulse.get("operation", ""))
            duration = pulse.get("duration")
            if not channel or not operation or duration is None:
                raise ValueError(
                    "Each pulse must define 'channel', 'operation', and 'duration'."
                )
            lowered.append(f"play({operation}, {channel}, duration={duration})")
        return lowered
