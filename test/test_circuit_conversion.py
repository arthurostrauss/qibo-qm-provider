"""Tests for qibo_qm_provider.backend.circuit_conversion."""

import pytest
from qibo import Circuit, gates
from qiskit.circuit import QuantumCircuit

from qibo_qm_provider.backend.circuit_conversion import (
    qibo_circuit_to_qiskit,
    normalize_symmetric_two_qubit_directions,
    validate_two_qubit_connectivity,
)
from qibo_qm_provider.exceptions import UnsupportedConnectivityError, UnsupportedGateError


def test_basic_gate_conversion():
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.RZ(1, theta=0.3))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.M(0, 1))

    qc = qibo_circuit_to_qiskit(circuit)

    names = [instr.operation.name for instr in qc.data]
    assert "x" in names
    assert "rz" in names
    assert "cz" in names
    assert "measure" in names
    assert qc.num_qubits == 2


def test_measurement_ordering_matches_gate_qubit_order():
    """circuit.to_qasm() names one creg per M gate and orders bits by the
    gate's own qubit argument order -- confirmed empirically (see
    measurement_translation.py docstring), not assumed here."""
    circuit = Circuit(3)
    circuit.add(gates.M(0, 2))

    qc = qibo_circuit_to_qiskit(circuit)

    assert len(qc.cregs) == 1
    creg = qc.cregs[0]
    assert creg.name == circuit.measurements[0].register_name
    assert creg.size == 2

    measure_instructions = [instr for instr in qc.data if instr.operation.name == "measure"]
    assert len(measure_instructions) == 2
    # measure q[0] -> creg[0]; measure q[2] -> creg[1], matching gate.qubits == (0, 2)
    qubit_to_clbit = {
        qc.find_bit(instr.qubits[0]).index: qc.find_bit(instr.clbits[0]).index for instr in measure_instructions
    }
    assert qubit_to_clbit[0] == 0
    assert qubit_to_clbit[2] == 1


def test_multiple_measurement_gates_get_distinct_registers():
    circuit = Circuit(3)
    circuit.add(gates.M(0))
    circuit.add(gates.M(1, 2))

    qc = qibo_circuit_to_qiskit(circuit)

    assert len(qc.cregs) == 2
    names = {creg.name for creg in qc.cregs}
    assert names == {m.register_name for m in circuit.measurements}


def test_gpi_gate_is_supported_directly():
    """gpi/gpi2 have no entry in OpenQASM2's standard qelib1.inc library, so
    the old OpenQASM2 round-trip needed a special-cased inline decomposition
    to accept them at all (removed along with that route, see
    circuit_conversion's module docstring). The direct builder maps them
    through gate_map.QIBO_TO_QISKIT like any other gate."""
    circuit = Circuit(1)
    circuit.add(gates.GPI(0, phi=0.3))
    circuit.add(gates.M(0))

    qc = qibo_circuit_to_qiskit(circuit)

    assert [instr.operation.name for instr in qc.data][0] == "gpi"


def test_align_gate_is_explicitly_unsupported():
    """qibo.gates.Align(q, delay=0) is a single-qubit synchronization gate
    (aligning multiple qubits means adding one Align per qubit, not a single
    variadic-qubit gate -- gates.Align(0, 1) would actually mean q=0,
    delay=1, not "align qubits 0 and 1"). It has no representation on this
    path at all (gate_map.DEFERRED_GATES), regardless of which qubit(s)/delay
    are used. This must surface as an actionable, provider-specific error,
    not silently drop the gate or produce an incorrect circuit."""
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.Align(0, delay=10))
    circuit.add(gates.Align(1, delay=10))
    circuit.add(gates.M(0, 1))

    with pytest.raises(UnsupportedGateError):
        qibo_circuit_to_qiskit(circuit)


# --------------------------------------------------------------------------- #
# wire_names resolution
#
# Root-caused by a real "arbel" live-hardware failure (UnresolvableOperation
# compiling CZ(0,1)): Circuit.to_qasm() never consults circuit.wire_names at
# all (verified against qibo 0.3.3 source), so this package previously passed
# Qibo's plain gate.qubits straight through with no way for a caller to
# route logical qubits onto specific physical ones.
# --------------------------------------------------------------------------- #


def test_qubit_dict_none_is_identity_even_with_wire_names_set():
    """The default (qubit_dict=None, e.g. every call not going through
    QiboQMBackend) must behave exactly as before wire_names support existed:
    plain gate.qubits, unaffected by whatever wire_names happens to be set."""
    circuit = Circuit(2, wire_names=["q1", "q0"])
    circuit.add(gates.X(0))

    qc = qibo_circuit_to_qiskit(circuit)

    x_instr = next(instr for instr in qc.data if instr.operation.name == "x")
    assert qc.find_bit(x_instr.qubits[0]).index == 0


def test_default_wire_names_is_identity_even_with_qubit_dict_given():
    """Qibo's own default wire_names is list(range(nqubits)) -- with no
    explicit remap requested, a qubit_dict must not change anything either."""
    circuit = Circuit(2)
    circuit.add(gates.X(0))

    qc = qibo_circuit_to_qiskit(circuit, qubit_dict={"q0": 0, "q1": 1})

    x_instr = next(instr for instr in qc.data if instr.operation.name == "x")
    assert qc.find_bit(x_instr.qubits[0]).index == 0


def test_wire_names_remaps_qubit_indices_via_qubit_dict():
    """circuit.wire_names=["q1","q0"] means logical qubit 0 IS "q1" -- so a
    gate written for logical qubit 0 must land on qubit_dict["q1"]."""
    circuit = Circuit(2, wire_names=["q1", "q0"])
    circuit.add(gates.X(0))
    circuit.add(gates.M(0, 1))

    qc = qibo_circuit_to_qiskit(circuit, qubit_dict={"q0": 0, "q1": 1})

    x_instr = next(instr for instr in qc.data if instr.operation.name == "x")
    assert qc.find_bit(x_instr.qubits[0]).index == 1


def test_wire_names_unknown_qubit_raises_value_error():
    circuit = Circuit(1, wire_names=["ghost"])
    circuit.add(gates.X(0))

    with pytest.raises(ValueError, match="ghost"):
        qibo_circuit_to_qiskit(circuit, qubit_dict={"q0": 0})


def test_explicit_wire_names_uses_a_pinned_layout_not_target_translation():
    """The explicit-wire_names path goes through a real qiskit.transpiler.
    Layout + qiskit.compiler.transpile() (issue #7), not the old hand-rolled
    QuantumCircuit.compose() remap -- but deliberately with no backend/target
    involved, so gate content is untouched (only the physical index of each
    qubit changes)."""
    circuit = Circuit(2, wire_names=["q1", "q0"])
    circuit.add(gates.CZ(0, 1))

    qc = qibo_circuit_to_qiskit(circuit, qubit_dict={"q0": 0, "q1": 1})

    assert [instr.operation.name for instr in qc.data] == ["cz"]
    cz_instr = next(instr for instr in qc.data if instr.operation.name == "cz")
    # logical (0, 1) -> physical (qubit_dict["q1"], qubit_dict["q0"]) = (1, 0)
    assert [qc.find_bit(q).index for q in cz_instr.qubits] == [1, 0]


# --------------------------------------------------------------------------- #
# backend-targeted automatic layout (issue #7)
#
# circuit.wire_names left at its default with a real backend given: rather
# than staying a no-op, qibo_circuit_to_qiskit now runs an ordinary
# qiskit.compiler.transpile(qc, backend=...) against the backend's real
# Target, so Qiskit's own preset pass manager can place/route/correct the
# circuit automatically -- see QiboQMBackend.execute_circuit's docstring for
# the full rationale (and why it's gated behind transpile=True there).
# --------------------------------------------------------------------------- #


def test_backend_none_default_wire_names_is_still_a_no_op(add_basic_macros_installed):
    """No backend given -> unchanged no-op, regardless of qubit_dict --
    matches every caller with no machine to transpile against."""
    from qibo_qm_provider.backend.qibo_qm_backend import QiboQMBackend

    backend = QiboQMBackend(add_basic_macros_installed)
    circuit = Circuit(2)
    circuit.add(gates.CZ(1, 0))

    qc = qibo_circuit_to_qiskit(circuit, qubit_dict=backend.qiskit_backend.qubit_dict)

    cz_instr = next(instr for instr in qc.data if instr.operation.name == "cz")
    assert [qc.find_bit(q).index for q in cz_instr.qubits] == [1, 0]


def test_backend_given_default_wire_names_auto_places_against_real_target(add_basic_macros_installed):
    """With a real backend and default wire_names, the resulting circuit is
    placed/corrected against the backend's actual Target -- here, the only
    registered "q0-q1" direction is (0, 1), so a CZ(1, 0) gets its qubit
    order corrected automatically (CZ is unitarily symmetric)."""
    from qibo_qm_provider.backend.qibo_qm_backend import QiboQMBackend

    backend = QiboQMBackend(add_basic_macros_installed)
    assert backend.qiskit_backend.qubit_pair_dict["q0-q1"] == (0, 1)
    circuit = Circuit(2)
    circuit.add(gates.CZ(1, 0))

    qc = qibo_circuit_to_qiskit(
        circuit,
        qubit_dict=backend.qiskit_backend.qubit_dict,
        backend=backend.qiskit_backend,
    )

    cz_instr = next(instr for instr in qc.data if instr.operation.name == "cz")
    assert [qc.find_bit(q).index for q in cz_instr.qubits] == [0, 1]


# --------------------------------------------------------------------------- #
# validate_two_qubit_connectivity
#
# qm_qasm registers a two-qubit macro under exactly one ordered
# (control_index, target_index) pair (verified against qiskit_qm_provider's
# _populate_target and qm_qasm's own qubit-pattern matching, both exact,
# order-sensitive, with no symmetric fallback) -- many QM two-qubit natives
# (e.g. a flux-tunable CZ) really are physically asymmetric.
# --------------------------------------------------------------------------- #


def test_validate_two_qubit_connectivity_accepts_registered_direction():
    qc = QuantumCircuit(2)
    qc.cz(0, 1)

    validate_two_qubit_connectivity(qc, qubit_pair_dict={"q0-q1": (0, 1)})  # must not raise


def test_validate_two_qubit_connectivity_rejects_reversed_direction():
    qc = QuantumCircuit(2)
    qc.cz(1, 0)

    with pytest.raises(UnsupportedConnectivityError, match="cz"):
        validate_two_qubit_connectivity(qc, qubit_pair_dict={"q0-q1": (0, 1)})


def test_validate_two_qubit_connectivity_names_qubits_when_qubit_dict_given():
    qc = QuantumCircuit(2)
    qc.cz(1, 0)

    with pytest.raises(UnsupportedConnectivityError, match=r"cz\(q1, q0\).*cz\(q0, q1\)"):
        validate_two_qubit_connectivity(
            qc, qubit_pair_dict={"q0-q1": (0, 1)}, qubit_dict={"q0": 0, "q1": 1}
        )


def test_validate_two_qubit_connectivity_rejects_unconnected_pair():
    qc = QuantumCircuit(3)
    qc.cz(0, 2)

    with pytest.raises(UnsupportedConnectivityError, match="either direction"):
        validate_two_qubit_connectivity(qc, qubit_pair_dict={"q0-q1": (0, 1)})


def test_validate_two_qubit_connectivity_ignores_measure_and_single_qubit_gates():
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.measure(0, 0)
    qc.measure(1, 1)

    validate_two_qubit_connectivity(qc, qubit_pair_dict={})  # must not raise


# --------------------------------------------------------------------------- #
# normalize_symmetric_two_qubit_directions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("gate", ["cz", "iswap"])
def test_normalize_reorders_reversed_symmetric_gate(gate):
    from qiskit.quantum_info import Operator

    qc = QuantumCircuit(2)
    qc.h(0)
    getattr(qc, gate)(1, 0)

    out = normalize_symmetric_two_qubit_directions(qc, {"q0-q1": (0, 1)})

    assert [out.find_bit(q).index for q in out.data[1].qubits] == [0, 1]
    assert Operator(out).equiv(Operator(qc))
    validate_two_qubit_connectivity(out, qubit_pair_dict={"q0-q1": (0, 1)})  # must not raise
    # the input circuit is left untouched
    assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [1, 0]


def test_normalize_leaves_asymmetric_gate_for_validation():
    qc = QuantumCircuit(2)
    qc.cx(1, 0)

    out = normalize_symmetric_two_qubit_directions(qc, {"q0-q1": (0, 1)})

    assert out is qc
    with pytest.raises(UnsupportedConnectivityError, match=r"cx"):
        validate_two_qubit_connectivity(out, qubit_pair_dict={"q0-q1": (0, 1)})


def test_normalize_leaves_registered_and_unconnected_pairs_alone():
    qc = QuantumCircuit(3)
    qc.cz(0, 1)
    qc.cz(0, 2)

    assert normalize_symmetric_two_qubit_directions(qc, {"q0-q1": (0, 1)}) is qc


def test_normalize_keeps_both_directions_when_both_are_registered():
    """Two separately calibrated directions: each written order already
    matches its own macro, nothing to reorder."""
    qc = QuantumCircuit(2)
    qc.cz(1, 0)

    pairs = {"q0-q1": (0, 1), "q1-q0": (1, 0)}
    assert normalize_symmetric_two_qubit_directions(qc, pairs) is qc
