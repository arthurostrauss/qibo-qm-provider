"""Exceptions raised by :mod:`qibo_qm_provider`."""

from __future__ import annotations


class GateNameMappingError(ValueError):
    """Raised when a Qibo gate has no known OpenQASM2/Qiskit equivalent."""


class UnsupportedGateError(NotImplementedError):
    """Raised when a Qibo gate cannot be lowered through the OpenQASM2/3 detour.

    ``qibo.gates.Align`` is the primary example: it has no OpenQASM
    representation, and ``Circuit.to_qasm()`` raises ``NotImplementedError``
    for it directly. This exception wraps that (and similar) failures with a
    message that names the offending gate and qubits.
    """
