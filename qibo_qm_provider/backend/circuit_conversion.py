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
gate name at the OQ2 text level is unaffected either way.

``qelib1.inc`` and ``LEGACY_CUSTOM_INSTRUCTIONS`` (fixes bug #3)
---------------------------------------------------------------
``Circuit.to_qasm()`` always emits ``include "qelib1.inc";``, but
``qiskit.qasm2.loads`` does **not** build in the whole of that library -- only
a minimal core. Passing ``custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS``
supplies the rest, and is simply honouring the include Qibo asked for. Verified
directly, this fixes two separate things at once:

* **Gate coverage.** Without it, ``swap``, ``crx``, ``cry``, ``crz``, ``rzz``,
  ``rxx``, ``ryy``, ... all fail with ``QASM2ParseError: '<gate>' is not
  defined in this scope``, despite being ordinary ``qelib1.inc`` gates. This was
  a real, previously unnoticed limitation of this route -- found by the
  direct-builder equivalence test in ``test/test_symbolic_lowering.py``.
* **Bug #3.** ``qibo.gates.I`` round-trips as ``id`` at the OQ2-text level, but
  the bare parser normalised the standard-library ``id`` straight to a generic
  ``u(0, 0, 0)`` instruction, which then failed inside ``qm_qasm`` because no
  ``OperationIdentifier("u", ...)`` is installed by ``add_basic_macros``. With
  the legacy definitions supplied, ``id`` stays ``id``. **Bug #3 is closed.**

No regressions from enabling it: ``x``, ``rz``, ``cz``, ``u3``, ``ccx``,
``cu1``, ``gpi`` and ``measure`` all parse identically either way (checked
gate by gate). ``iSWAP`` still fails -- it is genuinely absent from
``qelib1.inc``, and is supported on the symbolic route instead.

The symbolic path
-----------------
A Qibo circuit may carry a ``sympy`` expression as a gate parameter, and
``Circuit.to_qasm()`` then raises ``TypeError: Cannot convert expression to
float`` -- OpenQASM **2** has no symbolic ``input`` declaration, so the route
above cannot express it at all. For such circuits this module bypasses
``to_qasm()`` entirely and builds the Qiskit ``QuantumCircuit`` directly
(:func:`build_qiskit_circuit_directly`), mapping each sympy symbol to a Qiskit
``Parameter``. Everything downstream already handles that correctly:
``qiskit.qasm3`` exports a ``Parameter`` as a first-class ``input float[64]``,
and ``qm_qasm.Compiler.compile(code, inputs=...)`` binds it to a live QUA
variable.

The branch is chosen by :func:`~qibo_qm_provider.backend.symbolic_parameters.
circuit_has_symbols`, not by a ``try``/``except`` around ``to_qasm()``: the
existing OpenQASM2 route is hardware-validated and stays byte-for-byte
unchanged for every circuit that can use it.

One deliberate difference on the direct path:

* Gate coverage is an explicit table (:mod:`qibo_qm_provider.backend.gate_map`)
  rather than whatever Qibo's ``qasm_label`` happens to emit, so an unsupported
  gate fails with a message naming it instead of surfacing later as a
  missing-operation error.
"""

from __future__ import annotations

from typing import Dict, Optional

from qibo.models import Circuit as QiboCircuit
from qiskit import qasm2
from qiskit.circuit import ClassicalRegister, Parameter, QuantumCircuit

from qibo_qm_provider.exceptions import UnsupportedConnectivityError, UnsupportedGateError

from .gate_map import DEFERRED_GATES, QIBO_TO_OPERATION_NAME, QIBO_TO_QISKIT
from .symbolic_parameters import circuit_has_symbols, sympy_to_qiskit_parameter

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

    Used for circuits carrying symbolic (sympy) gate parameters, which
    ``Circuit.to_qasm()`` cannot express. Safe for non-symbolic circuits too --
    ``test/test_symbolic_lowering.py`` asserts it agrees with the OpenQASM2
    path on those, which is what pins this function against the already
    hardware-validated route.

    Classical registers reproduce ``to_qasm()``'s contract exactly, because
    :func:`~qibo_qm_provider.backend.measurement_translation.
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
                f"Qibo gate {name} on qubits {gate.qubits} is not supported by the "
                f"symbolic lowering path: {DEFERRED_GATES[name]}. Bind it to a "
                f"concrete circuit and use the OpenQASM2 path, or decompose it "
                f"into supported gates first."
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
    extended_compatibility: bool = True,
    qubit_dict: Optional[Dict[str, int]] = None,
) -> QuantumCircuit:
    """Convert a Qibo circuit to a Qiskit ``QuantumCircuit``.

    Takes one of two routes, chosen automatically:

    * **Symbolic** (any gate parameter is a ``sympy`` expression with free
      symbols): builds the Qiskit circuit directly via
      :func:`build_qiskit_circuit_directly`, mapping sympy symbols to Qiskit
      ``Parameter`` objects. ``Circuit.to_qasm()`` cannot represent these at
      all (OpenQASM2 has no symbolic ``input``).
    * **Concrete** (everything else): the original, hardware-validated
      ``Circuit.to_qasm()`` -> ``qiskit.qasm2.loads`` route, unchanged.

    Either route's result then has ``circuit.wire_names`` resolved against
    ``qubit_dict`` if both are given -- see :func:`_resolve_wire_names`. A
    caller with no machine (``qubit_dict=None``, the default) gets exactly
    today's behaviour: Qibo qubit index ``i`` becomes Qiskit qubit index
    ``i``, unconditionally.

    Args:
        circuit: A Qibo circuit. ``qibo.gates.Align`` is not supported on
            either route -- see module docstring.
        extended_compatibility: Forwarded to ``Circuit.to_qasm()``. Emits an
            inline decomposition for gates absent from OpenQASM2's standard
            library (e.g. ``gpi``, ``gpi2``), letting Qiskit's OQ2 parser
            accept them instead of raising ``QASM2ParseError``. Defaults to
            ``True``; no regression observed for gates that already worked
            without it (``x``, ``rz``, ``cz``, ``id``, ``measure``) -- see
            module docstring for what it does and does not fix. **Ignored on
            the symbolic route**, which never produces OpenQASM2 text.
        qubit_dict: A QuAM qubit name -> Qiskit index mapping (e.g.
            ``QiboQMBackend.qiskit_backend.qubit_dict``), used to resolve
            ``circuit.wire_names`` if it has been set to physical qubit
            names. ``None`` (the default) disables this entirely, matching
            behaviour before ``wire_names`` support existed.

    Returns:
        The equivalent Qiskit ``QuantumCircuit``, with one classical register
        per Qibo measurement gate, named after ``gates.M.register_name``.
        Both routes honour that contract -- see
        :func:`build_qiskit_circuit_directly`.

    Raises:
        UnsupportedGateError: If the circuit contains a gate with no
            representation on the chosen route (e.g. ``Align``).
        UnsupportedParameterError: On the symbolic route only, for a symbolic
            parameter ``qm_qasm`` could not compile.
        ValueError: If ``circuit.wire_names`` names a qubit absent from
            ``qubit_dict``.
    """
    if circuit_has_symbols(circuit):
        qc = build_qiskit_circuit_directly(circuit)
    else:
        try:
            qasm_source = circuit.to_qasm(extended_compatibility=extended_compatibility)
        except (NotImplementedError, TypeError) as exc:
            # TypeError is included because qibo raises a bare `TypeError: Cannot
            # convert expression to float` for a symbolic parameter. That is
            # normally caught by the `circuit_has_symbols` branch above, but a
            # non-sympy non-numeric parameter object would land here, and a bare
            # upstream TypeError names neither the circuit nor this package.
            raise UnsupportedGateError(
                f"Circuit could not be converted to OpenQASM2, and therefore cannot "
                f"be lowered through qibo_qm_provider's QMBackend detour: {exc}"
            ) from exc
        # `custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS` supplies the qelib1.inc
        # gate definitions Qiskit's OQ2 parser does not build in. Qibo always emits
        # `include "qelib1.inc"`, so honouring it is what it asked for -- see the
        # module docstring for what this fixes (bug #3, plus swap/crx/rzz/rxx/...).
        qc = qasm2.loads(qasm_source, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)

    return _resolve_wire_names(circuit, qc, qubit_dict)
