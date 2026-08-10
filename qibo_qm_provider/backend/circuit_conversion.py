"""Convert a Qibo circuit into a Qiskit ``QuantumCircuit`` via OpenQASM2.

``qibo.models.Circuit`` only exports natively to OpenQASM2 (``to_qasm``); there
is no direct Qibo-to-Qiskit object conversion. ``qiskit_qm_provider.backend.
qm_backend.QMBackend.quantum_circuit_to_qua`` itself re-exports to OpenQASM3
internally (via ``qiskit.qasm3.dumps``) before handing the source to
``qm_qasm.Compiler`` -- this module only needs to produce the Qiskit
``QuantumCircuit`` that step consumes.

``qibo.gates.Align`` has no OpenQASM representation: ``Circuit.to_qasm()``
raises ``NotImplementedError("Align is not supported by OpenQASM")`` directly
(confirmed empirically against qibo 0.3.3), regardless of
``extended_compatibility``. That failure is allowed to propagate, wrapped in
:class:`~qibo_qm_provider.exceptions.UnsupportedGateError` so callers get an
actionable, provider-specific error instead of a bare upstream
``NotImplementedError``.

``extended_compatibility=True`` (the default here) asks ``Circuit.to_qasm()``
to additionally emit an inline ``gate <name>(...) q {...}`` decomposition for
any gate with no entry in OpenQASM2's standard ``qelib1.inc`` library (e.g.
``gpi``, ``gpi2``). This matters because ``qiskit.qasm2.loads`` -- Qiskit's
own OQ2 parser, used below -- raises ``QASM2ParseError: '<gate>' is not
defined in this scope`` for any such gate when no definition is present,
regardless of Qibo-side settings; confirmed empirically that supplying the
decomposition (``extended_compatibility=True``) is what fixes this, since the
gate name at the OQ2 text level is unaffected either way. This does **not**
fix every gate-coverage gap: ``qibo.gates.I`` already round-trips fine as
``id`` at the OQ2-text level in both modes, but ``qiskit.qasm2.loads`` itself
normalizes the standard-library ``id`` gate straight to a generic
``u(0, 0, 0)`` instruction (confirmed via direct inspection of its parsed
output) -- a Qiskit-side normalization this parameter has no effect on. Since
no ``OperationIdentifier("u", ...)`` is installed by ``add_basic_macros``,
circuits containing ``I`` still fail later, inside ``qm_qasm``'s compiler,
with an actionable "missing operation" error rather than a parse error --
tracked as an open item, not fixed by this parameter.
"""

from __future__ import annotations

from qibo.models import Circuit as QiboCircuit
from qiskit import qasm2
from qiskit.circuit import QuantumCircuit

from qibo_qm_provider.exceptions import UnsupportedGateError

__all__ = ["qibo_circuit_to_qiskit"]


def qibo_circuit_to_qiskit(circuit: QiboCircuit, extended_compatibility: bool = True) -> QuantumCircuit:
    """Convert a Qibo circuit to a Qiskit ``QuantumCircuit`` via OpenQASM2.

    Args:
        circuit: A Qibo circuit using only OpenQASM2-representable gates.
            ``qibo.gates.Align`` is not supported -- see module docstring.
        extended_compatibility: Forwarded to ``Circuit.to_qasm()``. Emits an
            inline decomposition for gates absent from OpenQASM2's standard
            library (e.g. ``gpi``, ``gpi2``), letting Qiskit's OQ2 parser
            accept them instead of raising ``QASM2ParseError``. Defaults to
            ``True``; no regression observed for gates that already worked
            without it (``x``, ``rz``, ``cz``, ``id``, ``measure``) -- see
            module docstring for what it does and does not fix.

    Returns:
        The equivalent Qiskit ``QuantumCircuit``, including its classical
        registers as declared by ``circuit.to_qasm()`` (one register per
        Qibo measurement gate, named after ``gates.M.register_name``).

    Raises:
        UnsupportedGateError: If the circuit contains a gate with no OpenQASM2
            representation (e.g. ``Align``).
    """
    try:
        qasm_source = circuit.to_qasm(extended_compatibility=extended_compatibility)
    except NotImplementedError as exc:
        raise UnsupportedGateError(
            f"Circuit could not be converted to OpenQASM2, and therefore cannot "
            f"be lowered through qibo_qm_provider's QMBackend detour: {exc}"
        ) from exc
    return qasm2.loads(qasm_source)
