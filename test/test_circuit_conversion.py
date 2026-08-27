"""Tests for qibo_qm_provider.backend.circuit_conversion."""

import pytest
from qibo import Circuit, gates
from qiskit.circuit import QuantumCircuit

from qibo_qm_provider.backend.circuit_conversion import (
    qibo_circuit_to_qiskit,
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


def test_extended_compatibility_allows_non_standard_gates():
    """gpi/gpi2 have no entry in OpenQASM2's standard qelib1.inc library, so
    qiskit.qasm2.loads rejects them (QASM2ParseError: not defined in this
    scope) unless Circuit.to_qasm() also emits an inline decomposition --
    confirmed empirically, not assumed. extended_compatibility=True (the
    default) requests that decomposition."""
    circuit = Circuit(1)
    circuit.add(gates.GPI(0, phi=0.3))
    circuit.add(gates.M(0))

    qc = qibo_circuit_to_qiskit(circuit)  # default extended_compatibility=True

    assert [instr.operation.name for instr in qc.data][0] == "gpi"


def test_extended_compatibility_false_rejects_non_standard_gates():
    from qiskit.qasm2.exceptions import QASM2ParseError

    circuit = Circuit(1)
    circuit.add(gates.GPI(0, phi=0.3))
    circuit.add(gates.M(0))

    with pytest.raises(QASM2ParseError):
        qibo_circuit_to_qiskit(circuit, extended_compatibility=False)


def test_align_gate_is_explicitly_unsupported():
    """qibo.gates.Align(q, delay=0) is a single-qubit synchronization gate
    (aligning multiple qubits means adding one Align per qubit, not a single
    variadic-qubit gate -- gates.Align(0, 1) would actually mean q=0,
    delay=1, not "align qubits 0 and 1"). It has no OpenQASM representation;
    Circuit.to_qasm() raises NotImplementedError directly (confirmed
    empirically against qibo 0.3.3), regardless of which qubit(s)/delay are
    used. This must surface as an actionable, provider-specific error, not
    silently drop the gate or produce an incorrect circuit."""
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
