"""Tests for qibo_qm_provider.backend.circuit_conversion."""

import pytest
from qibo import Circuit, gates

from qibo_qm_provider.backend.circuit_conversion import qibo_circuit_to_qiskit
from qibo_qm_provider.exceptions import UnsupportedGateError


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
