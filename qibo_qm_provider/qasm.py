"""OpenQASM conversion helpers for Qibo to QM lowering flows."""

from __future__ import annotations

from typing import Any, Callable


def _load_qiskit_qasm_functions() -> tuple[Callable[[str], Any], Callable[[Any], str]]:
    try:
        from qiskit import qasm2, qasm3
    except Exception as exc:  # pragma: no cover - exercised through RuntimeError path
        raise RuntimeError(
            "Qiskit is required for OpenQASM2→OpenQASM3 conversion. "
            "Install qiskit to enable circuit lowering."
        ) from exc

    return qasm2.loads, qasm3.dumps


def qasm2_to_qasm3(
    qasm2_source: str,
    *,
    qasm2_loader: Callable[[str], Any] | None = None,
    qasm3_dumper: Callable[[Any], str] | None = None,
) -> str:
    """Convert OpenQASM2 source to OpenQASM3 using Qiskit loaders/dumpers."""
    if not qasm2_source or not qasm2_source.strip():
        raise ValueError("qasm2_source must be a non-empty OpenQASM2 string.")

    if qasm2_loader is None or qasm3_dumper is None:
        qasm2_loader, qasm3_dumper = _load_qiskit_qasm_functions()

    circuit = qasm2_loader(qasm2_source)
    return qasm3_dumper(circuit)
