"""Tests for the symbolic (sympy-parameter) Qibo-circuit -> QUA lowering path.

``qibo.models.Circuit`` accepts a ``sympy`` expression as a gate parameter;
``qibo_circuit_to_qiskit``/``build_qiskit_circuit_directly`` (one and the same
function's public entry point, see ``circuit_conversion``'s module docstring)
maps each sympy symbol to a Qiskit ``Parameter``.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import sympy as sp
from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qibo_qm_provider import (
    QiboParameterTable,
    QiboQMBackend,
    circuit_has_symbols,
    qibo_circuit_to_qiskit,
)
from qibo_qm_provider.backend.gate_map import (
    EMITTED_OPERATION_NAMES,
    QIBO_TO_OPERATION_NAME,
    QIBO_TO_QISKIT,
)
from qibo_qm_provider.backend.qibo_qiskit_gates import (
    FSimGate,
    GPI2Gate,
    GPIGate,
    QiboMSGate,
)
from qibo_qm_provider.backend.symbolic_parameters import sympy_to_qiskit_parameter
from qibo_qm_provider.exceptions import UnsupportedGateError, UnsupportedParameterError

NUMPY_BACKEND = NumpyBackend()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _qibo_unitary(circuit: Circuit) -> np.ndarray:
    return np.array(circuit.unitary(NUMPY_BACKEND))


def _qiskit_unitary(qc: QuantumCircuit) -> np.ndarray:
    """Qiskit's ``Operator`` is little-endian; ``reverse_bits()`` puts it in
    Qibo's big-endian convention so the two are directly comparable.

    Skipping this step makes every asymmetric two-qubit gate look wrong -- the
    trap noted in ``gate_map``'s module docstring.
    """
    return Operator(qc.reverse_bits()).data


def _equal_up_to_phase(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    index = np.unravel_index(np.argmax(np.abs(a)), a.shape)
    if abs(b[index]) < 1e-12:
        return False
    phase = a[index] / b[index]
    return bool(np.allclose(a, phase * b, atol=1e-8) and abs(abs(phase) - 1) < 1e-8)


def _measurement_map(qc: QuantumCircuit) -> dict:
    """``{qubit_index: (creg_name, clbit_index_within_creg)}`` for every measure."""
    starts = {}
    offset = 0
    for creg in qc.cregs:
        starts[creg.name] = offset
        offset += creg.size
    out = {}
    for instruction in qc.data:
        if instruction.operation.name != "measure":
            continue
        qubit = qc.find_bit(instruction.qubits[0]).index
        clbit = qc.find_bit(instruction.clbits[0])
        creg_name = clbit.registers[0][0].name
        out[qubit] = (creg_name, clbit.index - starts[creg_name])
    return out


# --------------------------------------------------------------------------- #
# 1. the direct builder agrees with Qibo's own unitary
#
# The OpenQASM2 round-trip this used to be pinned against is gone (see
# circuit_conversion's module docstring) -- qibo_circuit_to_qiskit *is*
# build_qiskit_circuit_directly now, so an equivalence test between them
# would be trivially true. What still matters is agreement with Qibo itself.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "gate,nqubits",
    [
        (gates.SWAP(0, 1), 2),
        (gates.CRX(0, 1, 0.3), 2),
        (gates.RZZ(0, 1, 0.3), 2),
        (gates.RXX(0, 1, 0.3), 2),
        (gates.H(0), 1),
        (gates.I(0), 1),
    ],
    ids=["swap", "crx", "rzz", "rxx", "h", "i"],
)
def test_concrete_gates_match_qibo_unitary(gate, nqubits):
    circuit = Circuit(nqubits)
    circuit.add(gate)

    qc = qibo_circuit_to_qiskit(circuit)

    assert _equal_up_to_phase(_qiskit_unitary(qc), _qibo_unitary(circuit))


# --------------------------------------------------------------------------- #
# 2-3. symbols and expressions
# --------------------------------------------------------------------------- #
def test_circuit_has_symbols_detects_symbolic_parameters():
    theta = sp.Symbol("theta")
    symbolic = Circuit(1)
    symbolic.add(gates.RZ(0, theta=theta))
    concrete = Circuit(1)
    concrete.add(gates.RZ(0, theta=0.3))

    assert circuit_has_symbols(symbolic)
    assert not circuit_has_symbols(concrete)


def test_bare_symbol_becomes_qiskit_parameter():
    theta = sp.Symbol("theta")
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=theta))

    qc = qibo_circuit_to_qiskit(circuit)

    assert [p.name for p in qc.parameters] == ["theta"]
    assert [i.operation.name for i in qc.data] == ["rz"]


def test_symbol_is_shared_across_gates():
    """One sympy symbol must map to one Qiskit Parameter object, or the
    exported OpenQASM3 would declare the same input twice."""
    theta = sp.Symbol("theta")
    circuit = Circuit(2)
    circuit.add(gates.RZ(0, theta=theta))
    circuit.add(gates.RX(1, theta=theta))

    qc = qibo_circuit_to_qiskit(circuit)

    assert len(qc.parameters) == 1


@pytest.mark.parametrize(
    "build_expr",
    [
        lambda t, p: 2 * t,
        lambda t, p: 2 * t + np.pi / 2,
        lambda t, p: t / 2,
        lambda t, p: -t,
        lambda t, p: t * p,
        lambda t, p: (t + p) / 2,
        lambda t, p: -(t + p) / 2,
        lambda t, p: sp.pi * t,
    ],
    ids=["2t", "2t+pi/2", "t/2", "-t", "t*p", "(t+p)/2", "-(t+p)/2", "pi*t"],
)
def test_supported_expressions_translate_and_evaluate(build_expr):
    """A rebuilt ParameterExpression must be numerically identical to the
    sympy expression it came from."""
    t, p = sp.symbols("theta phi")
    expr = build_expr(t, p)
    registry: dict = {}

    translated = sympy_to_qiskit_parameter(expr, registry)

    substitutions = {t: 0.3, p: 0.7}
    expected = float(expr.subs(substitutions))
    binding = {registry[sym.name]: substitutions[sym] for sym in expr.free_symbols}
    assert float(translated.bind(binding)) == pytest.approx(expected)


def test_expression_reaches_the_circuit():
    t = sp.Symbol("theta")
    circuit = Circuit(1)
    circuit.add(gates.RX(0, theta=2 * t + np.pi / 2))

    qc = qibo_circuit_to_qiskit(circuit)
    (parameter,) = qc.parameters
    bound = qc.assign_parameters({parameter: 0.3})

    assert float(bound.data[0].operation.params[0]) == pytest.approx(2 * 0.3 + np.pi / 2)


# --------------------------------------------------------------------------- #
# 4. rejections
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "build_expr,reason",
    [
        (lambda t: sp.sin(t), "sin -> openqasm3 FunctionCall, rejected by qm_qasm"),
        (lambda t: sp.cos(t), "cos -> openqasm3 FunctionCall, rejected by qm_qasm"),
        (lambda t: sp.exp(t), "exp -> openqasm3 FunctionCall, rejected by qm_qasm"),
        (lambda t: t**2, "** -> 'Binary expression undefined' in qm_qasm"),
        (lambda t: 1 / t, "Pow(-1) is still a Pow node"),
    ],
    ids=["sin", "cos", "exp", "pow2", "inverse"],
)
def test_unsupported_expressions_are_rejected(build_expr, reason):
    """qm_qasm accepts only ``+ - * /`` on a gate argument, so anything else is
    refused here where the message can name the gate, rather than surfacing as
    a DisallowedNodeTypeException from inside the compiler."""
    t = sp.Symbol("theta")
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=build_expr(t)))

    with pytest.raises(UnsupportedParameterError):
        qibo_circuit_to_qiskit(circuit)


@pytest.mark.parametrize("name", ["measure", "input", "delay", "float", "gate", "qubit"])
def test_reserved_parameter_names_are_rejected(name):
    """Qiskit's exporter *renames* a reserved-word parameter (``allow_rename=
    True``), and the ``inputs`` dict is still keyed by the original name, so the
    compile fails later with ``MissingInputException: The input measure_0 was
    not provided`` -- an error naming neither the gate nor the cause."""
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=sp.Symbol(name)))

    with pytest.raises(UnsupportedParameterError, match="reserved"):
        qibo_circuit_to_qiskit(circuit)


@pytest.mark.parametrize("name", ["pi", "tau", "euler"])
def test_builtin_constant_parameter_names_are_rejected(name):
    """These are not in Qiskit's reserved set, so the exporter emits
    ``input float[64] pi;`` happily -- and qm_qasm then raises
    ``RedeclarationException``."""
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=sp.Symbol(name)))

    with pytest.raises(UnsupportedParameterError, match="built-in constant"):
        qibo_circuit_to_qiskit(circuit)


def test_parameter_named_after_a_gate_in_the_circuit_is_rejected():
    """A parameter sharing a name with a gate the circuit actually emits gets
    renamed by the exporter, for the same reason as a reserved word."""
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=sp.Symbol("rz")))

    with pytest.raises(UnsupportedParameterError, match="collides"):
        qibo_circuit_to_qiskit(circuit)


def test_measure_named_parameter_is_rejected_via_reserved_word():
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=sp.Symbol("measure")))
    circuit.add(gates.M(0))

    with pytest.raises(UnsupportedParameterError):
        qibo_circuit_to_qiskit(circuit)


def test_collision_check_is_scoped_to_gates_actually_used():
    """The forbidden set is the operations *this circuit* emits, not every name
    the mapping table can produce.

    ``t`` and ``x`` are gate names, but they are also entirely reasonable
    parameter names, and a circuit with no T or X gate never emits them. This
    is deliberately a lower bound -- see ``build_qiskit_circuit_directly``.
    """
    for name in ("t", "x", "s"):
        circuit = Circuit(1)
        circuit.add(gates.RZ(0, theta=sp.Symbol(name)))
        assert [p.name for p in qibo_circuit_to_qiskit(circuit).parameters] == [name]

    # ... but the same name *is* refused once the colliding gate is present.
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=sp.Symbol("x")))
    circuit.add(gates.X(0))
    with pytest.raises(UnsupportedParameterError, match="collides"):
        qibo_circuit_to_qiskit(circuit)


def test_ordinary_parameter_names_are_accepted():
    """Guard against over-rejecting: ``sin`` and ``q`` are *not* reserved by
    Qiskit and compile fine (verified), so they must not be refused."""
    for name in ("theta", "q", "sin", "my_angle"):
        circuit = Circuit(1)
        circuit.add(gates.RZ(0, theta=sp.Symbol(name)))
        assert [p.name for p in qibo_circuit_to_qiskit(circuit).parameters] == [name]


@pytest.mark.parametrize(
    "gate",
    [
        gates.GIVENS(0, 1, 0.3),
        gates.RBS(0, 1, 0.3),
        gates.Align(0, 1),
    ],
    ids=["GIVENS", "RBS", "Align"],
)
def test_deferred_gates_raise_naming_the_gate(gate):
    circuit = Circuit(2)
    circuit.add(gate)
    circuit.add(gates.RZ(0, theta=sp.Symbol("theta")))

    with pytest.raises(UnsupportedGateError, match=type(gate).__name__):
        qibo_circuit_to_qiskit(circuit)


def test_collapsed_measurement_is_rejected():
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=sp.Symbol("theta")))
    circuit.add(gates.M(0, collapse=True))

    with pytest.raises(UnsupportedGateError, match="collapse"):
        qibo_circuit_to_qiskit(circuit)


def test_extra_controls_are_rejected():
    """``.controlled_by(...)`` widens a gate's arity past what its mapped Qiskit
    counterpart accepts; appending anyway would silently mis-wire the circuit."""
    circuit = Circuit(3)
    circuit.add(gates.RZ(0, theta=sp.Symbol("theta")).controlled_by(1, 2))

    with pytest.raises(UnsupportedGateError, match="controlled_by"):
        qibo_circuit_to_qiskit(circuit)


# --------------------------------------------------------------------------- #
# 5. measurement contract, on the symbolic route
# --------------------------------------------------------------------------- #
def test_symbolic_measurement_registers_follow_gate_order():
    circuit = Circuit(3)
    circuit.add(gates.RZ(0, theta=sp.Symbol("theta")))
    circuit.add(gates.M(0, 2))

    qc = qibo_circuit_to_qiskit(circuit)

    assert len(qc.cregs) == 1
    assert qc.cregs[0].name == circuit.measurements[0].register_name
    assert qc.cregs[0].size == 2
    # measure q[0] -> creg[0]; measure q[2] -> creg[1], matching gate.qubits
    assert _measurement_map(qc)[0][1] == 0
    assert _measurement_map(qc)[2][1] == 1


def test_symbolic_multiple_measurements_get_distinct_registers():
    circuit = Circuit(3)
    circuit.add(gates.RZ(0, theta=sp.Symbol("theta")))
    circuit.add(gates.M(0))
    circuit.add(gates.M(1, 2))

    qc = qibo_circuit_to_qiskit(circuit)

    assert len(qc.cregs) == 2
    assert {r.name for r in qc.cregs} == {m.register_name for m in circuit.measurements}


# --------------------------------------------------------------------------- #
# 6. QiboParameterTable
# --------------------------------------------------------------------------- #
def test_parameter_table_from_qibo_circuit_matches_from_qiskit():
    """Ordering is the reason ``from_qibo_circuit`` delegates: Qiskit sorts
    ``qc.parameters`` by name, so a circuit-order sympy walk would diverge.
    Symbols are deliberately declared in non-alphabetical circuit order here.
    """
    b, a = sp.symbols("b_angle a_angle")
    circuit = Circuit(2)
    circuit.add(gates.RZ(0, theta=b))
    circuit.add(gates.RX(1, theta=a))

    table = QiboParameterTable.from_qibo_circuit(circuit)
    reference = QiboParameterTable.from_qiskit(qibo_circuit_to_qiskit(circuit))

    assert isinstance(table, QiboParameterTable)
    assert list(table.table.keys()) == list(reference.table.keys())
    assert list(table.table.keys()) == ["a_angle", "b_angle"]


def test_parameter_table_is_none_without_parameters():
    """``from_qiskit`` returns None rather than an empty table; callers already
    branch on that, so it is preserved rather than smoothed over."""
    circuit = Circuit(1)
    circuit.add(gates.X(0))

    assert QiboParameterTable.from_qibo_circuit(circuit) is None


def test_parameter_table_types_angles_as_fixed():
    """Angles become QUA ``fixed``, matching ``from_qiskit``'s own choice.

    Worth pinning because ``fixed`` covers ``[-8, 8)`` (a signed 4.28
    fixed-point number) while Qibo angles are in radians -- a composite
    expression (e.g. ``2*theta + phi``) or a multi-turn sweep can exceed that,
    even though a single 0-2*pi sweep fits comfortably. See the overflow
    caveat documented on :class:`QiboParameterTable`.
    """
    from qm.qua import fixed

    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=sp.Symbol("theta")))

    table = QiboParameterTable.from_qibo_circuit(circuit)

    assert table.table["theta"].type is fixed


def test_fixed_type_overflow_bound_is_eight_not_two():
    """Pins the real overflow bound for a circuit angle typed as ``fixed``.

    ``qm.qua.declare(fixed, ...)``'s own docstring calls it "a signed 4.28
    fixed point number" -- ``[-8, 8)``, from 4 integer bits (including sign)
    plus 28 fractional bits. ``[-2, 2)`` is a different QUA convention (the
    valid ``amplitude_scale`` range for ``play``/``measure``), not this
    type's range -- an earlier version of this codebase's docs conflated the
    two. Pinned against ``qiskit_qm_provider``'s own ``FixedPoint`` (the same
    4.28 arithmetic QUA's own ``fixed`` type uses) so a future change to the
    fractional-bit width would be caught here rather than silently
    invalidating the documentation.
    """
    from qiskit_qm_provider.fixed_point import FixedPoint

    reference = FixedPoint(0.0)

    assert reference.fractional_bits == 28
    assert reference.bit_width == 32
    assert reference.min_value / reference.scale == -8.0
    assert reference.max_value / reference.scale == pytest.approx(8.0, abs=2**-28)


# --------------------------------------------------------------------------- #
# 7. gate unitaries
# --------------------------------------------------------------------------- #
def _gate_cases():
    """(id, qibo gate, qubit indices) for every gate in the mapping table."""
    a, b, c = 0.37, 1.13, 0.91
    return [
        ("I", gates.I(0), (0,)),
        ("H", gates.H(0), (0,)),
        ("X", gates.X(0), (0,)),
        ("Y", gates.Y(0), (0,)),
        ("Z", gates.Z(0), (0,)),
        ("S", gates.S(0), (0,)),
        ("SDG", gates.SDG(0), (0,)),
        ("T", gates.T(0), (0,)),
        ("TDG", gates.TDG(0), (0,)),
        ("SX", gates.SX(0), (0,)),
        ("SXDG", gates.SXDG(0), (0,)),
        ("RX", gates.RX(0, a), (0,)),
        ("RY", gates.RY(0, a), (0,)),
        ("RZ", gates.RZ(0, a), (0,)),
        ("U1", gates.U1(0, a), (0,)),
        ("U2", gates.U2(0, a, b), (0,)),
        ("U3", gates.U3(0, a, b, c), (0,)),
        ("PRX", gates.PRX(0, a, b), (0,)),
        ("U1q", gates.U1q(0, a, b), (0,)),
        ("GPI", gates.GPI(0, a), (0,)),
        ("GPI2", gates.GPI2(0, a), (0,)),
        ("CNOT", gates.CNOT(0, 1), (0, 1)),
        ("CZ", gates.CZ(0, 1), (0, 1)),
        ("CY", gates.CY(0, 1), (0, 1)),
        ("SWAP", gates.SWAP(0, 1), (0, 1)),
        ("iSWAP", gates.iSWAP(0, 1), (0, 1)),
        ("CSX", gates.CSX(0, 1), (0, 1)),
        ("CRX", gates.CRX(0, 1, a), (0, 1)),
        ("CRY", gates.CRY(0, 1, a), (0, 1)),
        ("CRZ", gates.CRZ(0, 1, a), (0, 1)),
        ("CU1", gates.CU1(0, 1, a), (0, 1)),
        ("CU2", gates.CU2(0, 1, a, b), (0, 1)),
        ("CU3", gates.CU3(0, 1, a, b, c), (0, 1)),
        ("RXX", gates.RXX(0, 1, a), (0, 1)),
        ("RYY", gates.RYY(0, 1, a), (0, 1)),
        ("RZZ", gates.RZZ(0, 1, a), (0, 1)),
        ("RZX", gates.RZX(0, 1, a), (0, 1)),
        ("RXXYY", gates.RXXYY(0, 1, a), (0, 1)),
        ("fSim", gates.fSim(0, 1, a, b), (0, 1)),
        ("MS", gates.MS(0, 1, a, b, 1.2), (0, 1)),
        ("TOFFOLI", gates.TOFFOLI(0, 1, 2), (0, 1, 2)),
        ("CCZ", gates.CCZ(0, 1, 2), (0, 1, 2)),
    ]


@pytest.mark.parametrize("label,gate,qubits", _gate_cases(), ids=[case[0] for case in _gate_cases()])
def test_mapped_gate_unitaries_match_qibo(label, gate, qubits):
    """Every entry in ``QIBO_TO_QISKIT`` must reproduce Qibo's own unitary.

    This is what makes the mapping table trustworthy, notably for the two
    non-obvious cases: ``CU2``/``CU3`` need ``CUGate(..., gamma=-(phi+lam)/2)``
    (``CU3Gate`` and ``gamma=0`` are both wrong), and ``U2`` is
    ``UGate(pi/2, phi, lam)``.
    """
    nqubits = max(qubits) + 1
    reference = Circuit(nqubits)
    reference.add(gate)

    builder = QIBO_TO_QISKIT[type(gate).__name__]
    qiskit_gate = builder(*gate.parameters)
    qc = QuantumCircuit(nqubits)
    qc.append(qiskit_gate, list(qubits))

    assert _equal_up_to_phase(_qiskit_unitary(qc), _qibo_unitary(reference))
    assert qiskit_gate.name == QIBO_TO_OPERATION_NAME[type(gate).__name__]


@pytest.mark.parametrize(
    "custom_gate,qibo_gate,nqubits",
    [
        (GPIGate(0.83), gates.GPI(0, 0.83), 1),
        (GPI2Gate(0.83), gates.GPI2(0, 0.83), 1),
        (QiboMSGate(0.53, 0.91, 1.1), gates.MS(0, 1, 0.53, 0.91, 1.1), 2),
        (FSimGate(0.3, 0.4), gates.fSim(0, 1, 0.3, 0.4), 2),
    ],
    ids=["gpi", "gpi2", "ms", "fsim"],
)
def test_custom_gate_definitions_match_their_arrays(custom_gate, qibo_gate, nqubits):
    """``_define()`` is unused on the QUA path (these names are emitted
    opaquely as basis gates), but it must still agree with ``__array__`` and
    with Qibo -- otherwise transpiling or simulating the same circuit silently
    means something else.
    """
    reference = Circuit(nqubits)
    reference.add(qibo_gate)
    expected = _qibo_unitary(reference)

    qc = QuantumCircuit(nqubits)
    qc.append(custom_gate, list(range(nqubits)))

    assert _equal_up_to_phase(_qiskit_unitary(qc), expected)
    assert _equal_up_to_phase(_qiskit_unitary(qc.decompose()), expected)


def test_ms_reduces_to_msgate_without_phases():
    """Pins the custom MS definition against Qiskit's own MSGate in the one
    case where they do agree -- ``phi0 = phi1 = 0``. They diverge for non-zero
    phases, which is why the custom gate exists.
    """
    from qiskit.circuit.library import MSGate

    theta = 0.37
    ours = QuantumCircuit(2)
    ours.append(QiboMSGate(0.0, 0.0, theta), [0, 1])
    theirs = QuantumCircuit(2)
    theirs.append(MSGate(2, theta), [0, 1])

    assert _equal_up_to_phase(_qiskit_unitary(ours), _qiskit_unitary(theirs))


def test_gate_map_tables_are_consistent():
    assert set(QIBO_TO_QISKIT) == set(QIBO_TO_OPERATION_NAME)
    assert set(QIBO_TO_OPERATION_NAME.values()) <= EMITTED_OPERATION_NAMES


# --------------------------------------------------------------------------- #
# 8. end to end: a symbolic RZ becomes a real-time QUA frame rotation
# --------------------------------------------------------------------------- #
def test_symbolic_rz_compiles_to_negated_frame_rotation(add_basic_macros_installed):
    """The headline end-to-end check.

    ``add_basic_macros`` installs ``rz`` as ``VirtualZMacro``, whose ``apply``
    is ``qubit.xy.frame_rotation(-angle)``. Compiling a symbolic ``RZ`` through
    the whole pipeline and reading back the generated QUA script asserts three
    things at once: the sympy symbol survived as a live QUA *variable* (not a
    baked-in float), the **minus sign** was applied, and it landed on the right
    qubit's ``xy`` element.

    ``generate_qua_script`` needs no hardware or QM connection.
    """
    from qm import generate_qua_script
    from qm.qua import program

    theta = sp.Symbol("theta")
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=theta))

    backend = QiboQMBackend(add_basic_macros_installed)
    qc = qibo_circuit_to_qiskit(circuit)
    table = QiboParameterTable.from_qibo_circuit(circuit)
    assert table is not None

    with program() as prog:
        backend.qiskit_backend.quantum_circuit_to_qua(qc, table)
    script = generate_qua_script(prog)

    rotations = [line.strip() for line in script.splitlines() if "frame_rotation_2pi" in line]
    assert len(rotations) == 1, script
    # e.g. frame_rotation_2pi(((0.0-v2)*0.15915494309189535), 'q0.xy')
    match = re.search(
        r"frame_rotation_2pi\(\(\(0\.0-(v\d+)\)\*([0-9.]+)\),\s*'(?P<element>[^']+)'\)",
        rotations[0],
    )
    assert match is not None, rotations[0]
    # the angle is a declared QUA variable, negated (the minus sign of RZ)
    assert f"{match.group(1)} = declare(fixed" in script
    # ... scaled by 1/(2*pi), since frame_rotation_2pi takes turns
    assert float(match.group(2)) == pytest.approx(1 / (2 * np.pi))
    assert match.group("element") == "q0.xy"


def _recording_macro_class(nparams: int):
    """A ``QubitMacro`` subclass recording the arguments its ``apply`` gets.

    Must be a real ``QubitMacro``, not a bare function:
    ``QMBackend._populate_target`` reads ``macro.apply`` unconditionally, so a
    plain callable raises ``AttributeError: 'function' object has no attribute
    'apply'`` -- despite ``QiboQMBackend.register_gate``'s docstring claiming a
    bare callable is accepted.
    """
    from quam.components.macro import QubitMacro
    from quam.core import quam_dataclass

    received: list = []

    if nparams == 1:

        @quam_dataclass
        class Recording(QubitMacro):
            def apply(self, angle, **kwargs):
                received.append(angle)

    else:

        @quam_dataclass
        class Recording(QubitMacro):
            def apply(self, theta, phi, **kwargs):
                received.append((theta, phi))

    return Recording, received


def test_symbolic_gate_reaches_macro_as_qua_variable(add_basic_macros_installed):
    """Complements the script assertion above: a recording macro proves the
    value handed to ``apply`` is a live QUA object, not a Python float.

    Installs the macro under ``prx``, Qibo's own name for the gate -- and,
    since ``gate_map.QIBO_TO_QISKIT["PRX"]`` builds a :class:`QiboPRXGate`
    (not a bare Qiskit ``RGate``), ``prx`` is exactly the operation name the
    lowering emits. There is no rename to route around here (see
    ``qibo_qiskit_gates`` module docstring), and no reason to override
    ``rz`` instead -- see ``test_register_gate_cannot_override_an_existing_operation``.
    """
    from qm.qua import program

    machine = add_basic_macros_installed
    recording_class, received = _recording_macro_class(2)
    machine.qubits["q0"].macros["prx"] = recording_class()

    backend = QiboQMBackend(machine)
    circuit = Circuit(1)
    circuit.add(gates.PRX(0, sp.Symbol("theta"), sp.Symbol("phi")))
    qc = qibo_circuit_to_qiskit(circuit)
    table = QiboParameterTable.from_qibo_circuit(circuit)

    with program():
        backend.qiskit_backend.quantum_circuit_to_qua(qc, table)

    assert len(received) == 1
    for value in received[0]:
        assert not isinstance(value, float), f"angle was baked in as {value!r}"


@pytest.mark.xfail(
    reason=(
        "Upstream limitation, found while building this path -- root-caused precisely: "
        "qm_qasm.OperationIdentifier has no __eq__/__hash__ (falls back to identity), so "
        "qiskit-qm-provider's internal QUA-operation cache, keyed by that class directly, "
        "silently added a duplicate entry instead of overwriting one for an operation name "
        "that already existed. register_gate('rz', ...) after backend construction showed "
        "up in .natives while the compiler kept running the original macro. Fixed upstream "
        "in source (qiskit_qm_provider.backend.backend_utils.operation_key, used throughout "
        "QMBackend) but not yet released -- this package's dependency floor "
        "(qiskit-qm-provider>=0.3.4) predates it, so xfail here reflects what a plain "
        "`pip install` gets today. Installing the macro before constructing the backend "
        "remains the workaround either way. XPASS (harmless, strict=False) once the fix is "
        "installed -- e.g. locally, while testing against the patched source directly."
    ),
    strict=False,
)
def test_register_gate_cannot_override_an_existing_operation(add_basic_macros_installed):
    from qm.qua import program

    backend = QiboQMBackend(add_basic_macros_installed)
    recording_class, received = _recording_macro_class(1)
    backend.register_gate("rz", "q0", recording_class())
    assert "RZ" in backend.natives

    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=sp.Symbol("theta")))
    qc = qibo_circuit_to_qiskit(circuit)
    table = QiboParameterTable.from_qibo_circuit(circuit)

    with program():
        backend.qiskit_backend.quantum_circuit_to_qua(qc, table)

    assert len(received) == 1


def test_macro_installed_before_backend_construction_is_used(add_basic_macros_installed):
    """The documented workaround for the limitation above."""
    from qm.qua import program

    machine = add_basic_macros_installed
    recording_class, received = _recording_macro_class(1)
    machine.qubits["q0"].macros["rz"] = recording_class()

    backend = QiboQMBackend(machine)
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=sp.Symbol("theta")))
    qc = qibo_circuit_to_qiskit(circuit)
    table = QiboParameterTable.from_qibo_circuit(circuit)

    with program():
        backend.qiskit_backend.quantum_circuit_to_qua(qc, table)

    assert len(received) == 1


def test_concrete_circuit_still_compiles_without_parameters(add_basic_macros_installed):
    """A concrete (non-symbolic) circuit must be unaffected by any of the above."""
    from qm.qua import program

    circuit = Circuit(1)
    circuit.add(gates.X(0))

    backend = QiboQMBackend(add_basic_macros_installed)
    qc = qibo_circuit_to_qiskit(circuit)
    assert not qc.parameters

    with program():
        backend.qiskit_backend.quantum_circuit_to_qua(qc)


# --------------------------------------------------------------------------- #
# 9. no gate is ever renamed
# --------------------------------------------------------------------------- #
# Nine gates (U1, U2, U3, PRX, U1q, CU1, CU2, CU3, RXXYY) map onto a Qiskit
# standard gate that has a *different* `.name` than Qibo's own -- e.g. a bare
# `RGate` emits "r", but Qibo calls this gate "prx". Building one directly
# would rename the operation, so a QuAM macro installed under the Qibo name a
# user actually wrote (`prx`) would get a missing-operation error for a name
# they never used (`r`). This used to require an alias-installing step
# (`QiboQMBackend.install_qibo_gate_aliases`, now removed) to close that gap
# on the machine. `qibo_qiskit_gates` closes it at the source instead: each of
# the nine gates is a thin subclass of the real standard gate with `.name`
# fixed to Qibo's own name, so there is no rename left to alias, ever.
def test_no_gate_is_ever_renamed():
    """The direct invariant this fix establishes: every entry in
    ``QIBO_TO_QISKIT`` emits the mapped gate's own Qibo name -- comparing
    ``QIBO_TO_OPERATION_NAME`` (i.e. what a QuAM macro must be named) against
    ``gate.name`` (i.e. the name a user actually wrote), for every gate in the
    mapping table, not just the nine known to be affected."""
    for label, gate, _qubits in _gate_cases():
        name = type(gate).__name__
        assert QIBO_TO_OPERATION_NAME[name] == gate.name, (
            f"{name}: emits {QIBO_TO_OPERATION_NAME[name]!r}, but qibo calls "
            f"this gate {gate.name!r} -- a macro installed under the Qibo "
            f"name would not be found."
        )


@pytest.mark.parametrize(
    "label,gate,qubits",
    [case for case in _gate_cases() if case[0] in {"U1", "U2", "U3", "PRX", "U1q", "CU1", "CU2", "CU3", "RXXYY"}],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_renamed_standard_gate_keeps_isinstance(label, gate, qubits):
    """The nine gates subclass the real Qiskit standard gate rather than
    building a bare instance and mutating ``.name`` after the fact --
    verified here that ``isinstance`` against the real class still holds, so
    anything downstream that pattern-matches on gate type (a transpiler pass,
    a custom calibration rule) is unaffected by the rename.
    """
    import qiskit.circuit.library as qlib

    standard_class = {
        "U1": qlib.PhaseGate,
        "U2": qlib.UGate,
        "U3": qlib.UGate,
        "PRX": qlib.RGate,
        "U1q": qlib.RGate,
        "CU1": qlib.CPhaseGate,
        "CU2": qlib.CUGate,
        "CU3": qlib.CUGate,
        "RXXYY": qlib.XXPlusYYGate,
    }[label]

    qiskit_gate = QIBO_TO_QISKIT[label](*gate.parameters)

    assert isinstance(qiskit_gate, standard_class)
    assert qiskit_gate.name == gate.name != standard_class(*qiskit_gate.params).name
