"""Convert a Qibo circuit into a Qiskit ``QuantumCircuit`` directly, gate by
gate (:func:`build_qiskit_circuit_directly`).

``qibo.models.Circuit`` only exports natively to OpenQASM2 (``to_qasm``), and
an earlier version of this module round-tripped through it (``to_qasm()`` ->
``qiskit.qasm2.loads``) for any circuit with no symbolic gate parameters. That
route is gone: ``Circuit.to_qasm()`` raises ``NotImplementedError`` outright
for several gates the direct builder already supports (``PRX``, ``U1q``,
``MS``, ``fSim``, confirmed empirically against qibo 0.3.3), so building
directly is a strict coverage improvement with nothing lost -- gates the old
route could not express either
(:data:`~qibo_qm_provider.backend.gate_map.DEFERRED_GATES`) still fail, with
the same actionable error. It also closes bug #3 (``qibo.gates.I`` reaching
``qm_qasm`` as a generic, unregistered ``u(0, 0, 0)`` after an OpenQASM2
round-trip) by construction, since the direct builder never produces OpenQASM
text at all.

``qibo.gates.Align`` has no representation on this path either (it is in
:data:`~qibo_qm_provider.backend.gate_map.DEFERRED_GATES`), so it fails with
an actionable :class:`~qibo_qm_provider.exceptions.UnsupportedGateError`
naming the gate, rather than a bare upstream error.

Gate coverage is an explicit table
(:mod:`qibo_qm_provider.backend.gate_map`) rather than whatever Qibo's
``qasm_label``/``to_qasm()`` happens to emit, so an unsupported gate fails
with a message naming it instead of surfacing later as a missing-operation
error.

Symbolic parameters
--------------------
A Qibo circuit may carry a ``sympy`` expression as a gate parameter --
something OpenQASM2 could never express in the first place (no symbolic
``input`` declaration). This module maps each sympy symbol to a Qiskit
``Parameter`` (:mod:`~qibo_qm_provider.backend.symbolic_parameters`).
Everything downstream already handles that correctly: ``qiskit.qasm3``
exports a ``Parameter`` as a first-class ``input float[64]``, and
``qm_qasm.Compiler.compile(code, inputs=...)`` binds it to a live QUA
variable.
"""

from __future__ import annotations

from typing import Dict, Optional

from qibo.models import Circuit as QiboCircuit
from qiskit.circuit import ClassicalRegister, Parameter, QuantumCircuit

from qibo_qm_provider.exceptions import UnsupportedConnectivityError, UnsupportedGateError

from .gate_map import DEFERRED_GATES, QIBO_TO_OPERATION_NAME, QIBO_TO_QISKIT
from .symbolic_parameters import sympy_to_qiskit_parameter

__all__ = [
    "qibo_circuit_to_qiskit",
    "build_qiskit_circuit_directly",
    "validate_two_qubit_connectivity",
]


def _measurement_is_plain(gate) -> bool:
    """Whether an ``M`` gate is a plain computational-basis measurement.

    ``collapse``, a non-``Z`` basis, and the ``p0``/``p1`` readout-error model
    all change what the gate *means*, and none of them has a representation on
    this path. Silently dropping them would produce results that look fine and
    are wrong, so they are refused.
    """
    if getattr(gate, "collapse", False):
        return False
    if getattr(gate, "p0", None) is not None or getattr(gate, "p1", None) is not None:
        return False
    basis = getattr(gate, "basis", None) or []
    return all(b in ("Z", "z") for b in basis)


def build_qiskit_circuit_directly(circuit: QiboCircuit) -> QuantumCircuit:
    """Build a Qiskit ``QuantumCircuit`` from a Qibo circuit gate by gate.

    The sole conversion path, for symbolic and concrete circuits alike -- see
    module docstring. Handles symbolic (sympy) gate parameters by mapping
    each sympy symbol to a Qiskit ``Parameter``.

    Classical registers reproduce ``Circuit.to_qasm()``'s historical contract
    exactly, because :func:`~qibo_qm_provider.backend.measurement_translation.
    translate_measurements` depends on it: one register per Qibo ``M`` gate,
    named after ``gate.register_name``, declared in gate order, with the bit
    for ``gate.qubits[j]`` at index ``j`` of that register.

    Raises:
        UnsupportedGateError: For a gate with no entry in
            :data:`~qibo_qm_provider.backend.gate_map.QIBO_TO_QISKIT`, for an
            ``M`` gate that is not a plain computational-basis measurement, or
            for a gate carrying extra controls via ``.controlled_by(...)``.
        UnsupportedParameterError: For a symbolic parameter outside the
            supported grammar -- see :mod:`qibo_qm_provider.backend.
            symbolic_parameters`.
    """
    qc = QuantumCircuit(circuit.nqubits)
    registry: dict[str, Parameter] = {}

    # Names a parameter must not take, because Qiskit's exporter would silently
    # rename it (see `validate_symbol_name`). Scoped to the operations *this
    # circuit* emits rather than every name the table can produce: `t`, `x`,
    # `s`, `p` are all gate names but also perfectly reasonable parameter
    # names, and rejecting them unconditionally would be hostile.
    #
    # This is necessarily a lower bound. `Exporter`'s `basis_gates` comes from
    # the backend's whole Target, so a parameter named after a macro installed
    # on the machine but *not* used in this circuit still collides. Catching
    # that needs the machine, which this function does not have -- it belongs in
    # the backend-level compile step.
    emitted_here = {
        QIBO_TO_OPERATION_NAME[name]
        for name in (type(gate).__name__ for gate in circuit.queue)
        if name in QIBO_TO_OPERATION_NAME
    }
    if circuit.measurements:
        emitted_here.add("measure")
    forbidden = frozenset(emitted_here)

    for gate in circuit.queue:
        name = type(gate).__name__

        if name == "M":
            if not _measurement_is_plain(gate):
                raise UnsupportedGateError(
                    f"Measurement gate on qubits {gate.qubits} uses a non-default "
                    f"configuration (collapse={getattr(gate, 'collapse', None)!r}, "
                    f"basis={getattr(gate, 'basis', None)!r}, "
                    f"p0={getattr(gate, 'p0', None)!r}, p1={getattr(gate, 'p1', None)!r}). "
                    f"Only plain computational-basis (Z) measurement without collapse "
                    f"or a readout-error model can be lowered to QUA through this path."
                )
            creg = ClassicalRegister(len(gate.qubits), name=gate.register_name)
            qc.add_register(creg)
            for index, qubit in enumerate(gate.qubits):
                qc.measure(qubit, creg[index])
            continue

        if name in DEFERRED_GATES:
            raise UnsupportedGateError(
                f"Qibo gate {name} on qubits {gate.qubits} is not supported: "
                f"{DEFERRED_GATES[name]}. Decompose it into supported gates first."
            )

        builder = QIBO_TO_QISKIT.get(name)
        if builder is None:
            raise UnsupportedGateError(
                f"Qibo gate {name} on qubits {gate.qubits} has no Qiskit mapping in "
                f"qibo_qm_provider.backend.gate_map.QIBO_TO_QISKIT, so it cannot be "
                f"lowered through the symbolic path. Add an entry there (with a "
                f"verified unitary equivalence) to support it."
            )

        params = [
            sympy_to_qiskit_parameter(param, registry, forbidden)
            for param in gate.parameters
        ]
        qiskit_gate = builder(*params)

        if qiskit_gate.num_qubits != len(gate.qubits):
            # The usual cause is `.controlled_by(...)`, which bolts extra
            # control qubits onto a gate whose mapped Qiskit counterpart has a
            # fixed arity. Appending anyway would silently mis-wire the circuit.
            raise UnsupportedGateError(
                f"Qibo gate {name} acts on {len(gate.qubits)} qubits {gate.qubits}, but "
                f"its Qiskit counterpart {qiskit_gate.name!r} takes "
                f"{qiskit_gate.num_qubits}. This usually means extra control qubits "
                f"were added with .controlled_by(...), which the symbolic lowering "
                f"path does not support."
            )

        qc.append(qiskit_gate, list(gate.qubits))

    return qc


def _resolve_wire_names(
    circuit: QiboCircuit, qc: QuantumCircuit, qubit_dict: Optional[Dict[str, int]]
) -> QuantumCircuit:
    """Remap ``qc``'s qubit indices from ``circuit.wire_names`` to the
    physical Qiskit indices ``qubit_dict`` (a QuAM qubit name -> Qiskit index
    mapping, e.g. ``QMBackend.qubit_dict``) assigns them.

    A no-op in the common case: ``circuit.wire_names`` defaults to plain
    ``list(range(nqubits))`` (Qibo's own default, per
    ``qibo.models.circuit.Circuit.wire_names``'s getter -- it is never
    consulted by ``Circuit.to_qasm()`` or by this module otherwise), and
    ``qubit_dict`` is ``None`` unless a caller with an actual machine (i.e.
    ``QiboQMBackend``) passes one in. Either condition alone is enough to
    keep today's behaviour (Qibo index ``i`` addresses Qiskit index ``i``
    verbatim) exactly as it was before this existed.

    When ``circuit.wire_names`` *is* set to physical qubit names, each
    logical index ``i`` is remapped so that the gate originally written for
    Qibo qubit ``i`` now addresses Qiskit qubit ``qubit_dict[wire_names[i]]``
    -- i.e. the machine's own qubit, wherever ``QMBackend`` put it in its
    ``Target``. This is the same contract ``qibolab``'s own compiler already
    honours for ``QiboQMPlatformBackend`` (``Compiler.get_sequence`` reads
    ``wire_names[q] for q in gate.qubits``, order preserving) -- this
    package's OpenQASM/``qm_qasm`` path previously had no equivalent at all.
    """
    if qubit_dict is None:
        return qc
    wire_names = circuit.wire_names
    if list(wire_names) == list(range(circuit.nqubits)):
        return qc
    try:
        mapping = [qubit_dict[name] for name in wire_names]
    except KeyError as exc:
        raise ValueError(
            f"circuit.wire_names names an unknown qubit {exc.args[0]!r}. "
            f"Known qubits on this machine: {sorted(qubit_dict)}."
        ) from exc
    remapped = QuantumCircuit(qc.num_qubits)
    for creg in qc.cregs:
        remapped.add_register(creg)
    remapped.compose(qc, qubits=mapping, clbits=list(range(qc.num_clbits)), inplace=True)
    return remapped


def validate_two_qubit_connectivity(
    qc: QuantumCircuit,
    qubit_pair_dict: Dict[str, tuple],
    qubit_dict: Optional[Dict[str, int]] = None,
) -> None:
    """Raise :class:`~qibo_qm_provider.exceptions.UnsupportedConnectivityError`
    if any two-qubit gate in ``qc`` addresses a physical qubit pair that has
    no registered connectivity on the machine, in that direction.

    ``qubit_pair_dict`` (e.g. ``QMBackend.qubit_pair_dict``) holds one
    ordered ``(control_index, target_index)`` tuple per configured
    ``QubitPair`` -- ``_populate_target`` never installs the reverse, and
    many QM two-qubit natives are physically asymmetric (see
    :class:`~qibo_qm_provider.exceptions.UnsupportedConnectivityError`'s
    docstring), so an exact ordered match is required; there is no
    order-agnostic fallback anywhere in ``qm_qasm`` either. Call this after
    :func:`qibo_circuit_to_qiskit` (so ``qc``'s qubit indices already reflect
    any ``wire_names`` remapping) and before handing ``qc`` to
    ``quantum_circuit_to_qua``/``run`` -- otherwise the same condition
    surfaces much later as an opaque ``qm_qasm.compiler.CompilationException``
    wrapping ``UnresolvableOperation``, naming neither the gate nor the qubits.

    Args:
        qc: The Qiskit circuit to check, with physical qubit indices already
            resolved.
        qubit_pair_dict: The machine's registered two-qubit connectivity.
        qubit_dict: Optional, for a more readable error naming the QuAM
            qubits by name rather than by raw Qiskit index.
    """
    valid_pairs = set(qubit_pair_dict.values())
    reverse_lookup = {index: name for name, index in (qubit_dict or {}).items()}
    for instruction in qc.data:
        qubits = instruction.qubits
        if len(qubits) != 2 or instruction.operation.name in ("measure", "barrier", "delay"):
            continue
        a, b = (qc.find_bit(qubits[0]).index, qc.find_bit(qubits[1]).index)
        if (a, b) in valid_pairs:
            continue
        name_a, name_b = reverse_lookup.get(a, a), reverse_lookup.get(b, b)
        gate_name = instruction.operation.name
        if (b, a) in valid_pairs:
            raise UnsupportedConnectivityError(
                f"{gate_name}({name_a}, {name_b}) has no registered connectivity in "
                f"this direction -- the installed macro only supports "
                f"{gate_name}({name_b}, {name_a}). This hardware's two-qubit gate is "
                f"physically asymmetric (see UnsupportedConnectivityError's docstring); "
                f"swap the gate's qubit order, or set circuit.wire_names to route "
                f"logical qubits onto physical qubits in the direction that is "
                f"actually calibrated."
            )
        raise UnsupportedConnectivityError(
            f"{gate_name}({name_a}, {name_b}): no registered connectivity between "
            f"these qubits in either direction on this machine."
        )


def qibo_circuit_to_qiskit(
    circuit: QiboCircuit,
    qubit_dict: Optional[Dict[str, int]] = None,
) -> QuantumCircuit:
    """Convert a Qibo circuit to a Qiskit ``QuantumCircuit``.

    Thin wrapper around :func:`build_qiskit_circuit_directly` that also
    resolves ``circuit.wire_names`` against ``qubit_dict`` if both are given
    -- see :func:`_resolve_wire_names`. A caller with no machine
    (``qubit_dict=None``, the default) gets Qibo qubit index ``i`` mapped to
    Qiskit qubit index ``i``, unconditionally.

    Args:
        circuit: A Qibo circuit, symbolic or concrete. ``qibo.gates.Align``
            is not supported -- see module docstring.
        qubit_dict: A QuAM qubit name -> Qiskit index mapping (e.g.
            ``QiboQMBackend.qiskit_backend.qubit_dict``), used to resolve
            ``circuit.wire_names`` if it has been set to physical qubit
            names. ``None`` (the default) disables this entirely, matching
            behaviour before ``wire_names`` support existed.

    Returns:
        The equivalent Qiskit ``QuantumCircuit``, with one classical register
        per Qibo measurement gate, named after ``gates.M.register_name`` --
        see :func:`build_qiskit_circuit_directly`.

    Raises:
        UnsupportedGateError: If the circuit contains a gate with no entry in
            :data:`~qibo_qm_provider.backend.gate_map.QIBO_TO_QISKIT` (e.g.
            ``Align``).
        UnsupportedParameterError: For a symbolic parameter outside the
            supported grammar.
        ValueError: If ``circuit.wire_names`` names a qubit absent from
            ``qubit_dict``.
    """
    qc = build_qiskit_circuit_directly(circuit)
    return _resolve_wire_names(circuit, qc, qubit_dict)
