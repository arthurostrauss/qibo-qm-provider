"""Tests for qibo_qm_provider.backend.measurement_translation.

Bit-ordering fixtures below were derived from direct, empirical testing of
qiskit.primitives.SamplerPubResult/DataBin/BitArray join behavior (see the
approved plan and measurement_translation.py's module docstring) -- not
assumed from documentation.
"""

import numpy as np
from qibo import Circuit, gates
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData

from qibo_qm_provider.backend.circuit_conversion import qibo_circuit_to_qiskit
from qibo_qm_provider.backend.measurement_translation import translate_measurements


def _make_result(memory):
    data = ExperimentResultData(memory=memory)
    exp = ExperimentResult(shots=len(memory), success=True, data=data, header={"name": "circ"})
    return Result(results=[exp], backend_name="dummy", backend_version="2", qobj_id=None, job_id="1", success=True)


def test_single_register_translation():
    circuit = Circuit(3)
    circuit.add(gates.M(0, 2))
    qc = qibo_circuit_to_qiskit(circuit)

    # shot0: q0=1,q2=0 -> "01"; shot1: q0=0,q2=1 -> "10" (confirmed empirically)
    result = _make_result(["01", "10"])

    translate_measurements(circuit, qc, result)

    samples = circuit.measurements[0].result.samples()
    np.testing.assert_array_equal(samples, np.array([[1, 0], [0, 1]]))


def test_multi_register_translation_respects_declaration_order():
    """Two M gates -> two classical registers. Confirmed empirically: the
    *last-declared* register occupies the leftmost (most-significant)
    characters of the joined memory bitstring; the first-declared register
    occupies the rightmost chunk."""
    circuit = Circuit(3)
    circuit.add(gates.M(0, 2))  # declared first -> register0, rightmost chunk (2 bits)
    circuit.add(gates.M(1))  # declared second -> register1, leftmost chunk (1 bit)
    qc = qibo_circuit_to_qiskit(circuit)

    # shot0: register0=[q0=1,q2=0]="01", register1=[q1=1]="1" -> joined "101"
    # shot1: register0=[q0=0,q2=1]="10", register1=[q1=0]="0" -> joined "010"
    result = _make_result(["101", "010"])

    translate_measurements(circuit, qc, result)

    m0, m1 = circuit.measurements
    np.testing.assert_array_equal(m0.result.samples(), np.array([[1, 0], [0, 1]]))
    np.testing.assert_array_equal(m1.result.samples(), np.array([[1], [0]]))
