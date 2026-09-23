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
from qiskit.compiler import transpile
from qiskit.providers import BackendV2
from qiskit.transpiler import Layout

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
    circuit: QiboCircuit,
    qc: QuantumCircuit,
    qubit_dict: Optional[Dict[str, int]],
    backend: Optional[BackendV2] = None,
) -> QuantumCircuit:
    """Place ``qc`` onto physical qubits, via a real Qiskit ``Layout`` and
    :func:`qiskit.compiler.transpile` -- not a hand-rolled index remap.

    Two cases, both driven by ``circuit.wire_names`` (Qibo's own logical
    index -> physical-qubit-name channel; defaults to plain
    ``list(range(nqubits))``, per ``qibo.models.circuit.Circuit.wire_names``'s
    getter):

    * **Explicit** (``wire_names`` set to physical qubit names): resolve each
      name through ``qubit_dict`` (a QuAM qubit name -> Qiskit index mapping,
      e.g. ``QMBackend.qubit_dict``) to get the physical index for every
      logical position, build a :class:`~qiskit.transpiler.Layout` from it
      (``Layout.from_intlist``), and transpile ``qc`` against *only* that
      pinned layout -- deliberately **not** against ``backend``/its
      ``Target``, so the requested physical assignment is honoured verbatim,
      with no gate-direction "fix-up" or rerouting Qiskit's target-aware
      passes might otherwise apply (a symmetric-looking gate like ``CZ`` can
      still be physically asymmetric on this hardware -- see
      :func:`validate_two_qubit_connectivity`, called by every caller of this
      function right after, which is the one place direction is judged).
      This is the same contract ``qibolab``'s own compiler already honours
      for ``QiboQMPlatformBackend`` (``Compiler.get_sequence`` reads
      ``wire_names[q] for q in gate.qubits``, order preserving).
    * **Default/unset**, with a ``backend`` given: run a full, ordinary
      ``transpile(qc, backend=backend)`` -- Qiskit's own preset pass manager
      picks layout and routing against the backend's real ``Target`` (which
      already reflects this machine's directional two-qubit connectivity),
      so a circuit whose plain Qibo indices don't already happen to line up
      with a valid physical assignment can still execute, without the
      caller having to place it by hand.
    * **Default/unset**, no ``backend``: unchanged, a no-op -- matches
      behaviour from before ``wire_names``/placement support existed, for
      every caller with no machine to transpile against (e.g. plain
      ``qibo_circuit_to_qiskit(circuit)``).

    Args:
        circuit: The source Qibo circuit (read only for ``.wire_names`` and
            ``.nqubits``).
        qc: The already gate-converted Qiskit circuit to place.
        qubit_dict: Required to resolve an explicit ``wire_names`` list;
            ignored otherwise.
        backend: The target ``BackendV2`` to transpile the default/unset
            case against. Optional -- with none given, that case is a no-op.

    Returns:
        The placed ``QuantumCircuit``, with its qubit indices now physical.
    """
    wire_names = circuit.wire_names
    is_default = list(wire_names) == list(range(circuit.nqubits))

    if is_default:
        if backend is None:
            return qc
        # optimization_level=0 deliberately: layout/routing/translation
        # still run (needed for both automatic placement and direction
        # correction -- see this function's docstring), but a higher level
        # also enables passes like RemoveDiagonalGatesBeforeMeasure, which
        # would e.g. drop a CZ immediately preceding a Z-basis measurement
        # of both its qubits -- mathematically output-equivalent, but not
        # the pulse sequence the caller actually asked to run on hardware.
        return transpile(qc, backend=backend, optimization_level=0)

    if qubit_dict is None:
        return qc
    try:
        mapping = [qubit_dict[name] for name in wire_names]
    except KeyError as exc:
        raise ValueError(
            f"circuit.wire_names names an unknown qubit {exc.args[0]!r}. "
            f"Known qubits on this machine: {sorted(qubit_dict)}."
        ) from exc
    layout = Layout.from_intlist(mapping, qc.qregs[0])
    # No backend/target here (see docstring): this is a pinned placement,
    # not a target-aware transpile, so it must not translate gates or
    # "correct" a two-qubit gate's direction -- validate_two_qubit_
    # connectivity (called by every caller of this function) is the one
    # place a direction mismatch is judged.
    return transpile(qc, initial_layout=layout, optimization_level=0)


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
    backend: Optional[BackendV2] = None,
) -> QuantumCircuit:
    """Convert a Qibo circuit to a Qiskit ``QuantumCircuit``, placed on
    physical qubits.

    Thin wrapper around :func:`build_qiskit_circuit_directly` that also
    places the result according to ``circuit.wire_names`` -- see
    :func:`_resolve_wire_names` for the full contract (explicit
    ``wire_names`` -> a pinned :class:`~qiskit.transpiler.Layout`; default
    ``wire_names`` + a ``backend`` -> an ordinary backend-targeted
    ``transpile()`` picks layout/routing automatically; neither -> a no-op,
    matching behaviour before ``wire_names``/placement support existed).

    Args:
        circuit: A Qibo circuit, symbolic or concrete. ``qibo.gates.Align``
            is not supported -- see module docstring.
        qubit_dict: A QuAM qubit name -> Qiskit index mapping (e.g.
            ``QiboQMBackend.qiskit_backend.qubit_dict``), used to resolve
            ``circuit.wire_names`` if it has been set to physical qubit
            names. ``None`` (the default) disables this entirely.
        backend: The ``BackendV2`` to transpile against when
            ``circuit.wire_names`` is left at its default -- see
            :func:`_resolve_wire_names`. ``None`` (the default) disables
            this entirely, so an unset ``wire_names`` stays a no-op.

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
    return _resolve_wire_names(circuit, qc, qubit_dict, backend)
