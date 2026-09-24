"""Pins ``gate_map.SYMMETRIC_TWO_QUBIT_NATIVE_GATES`` against its sources:
qibolab's ``TwoQubitNatives`` metadata, Qibo's ``NativeGates``, and the
unitaries themselves."""

from __future__ import annotations

import pytest
from qibo.transpiler.unroller import NativeGates
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qibo_qm_provider.backend.gate_map import (
    QIBO_TO_QISKIT,
    SYMMETRIC_TWO_QUBIT_NATIVE_GATES,
    SYMMETRIC_TWO_QUBIT_OPERATION_NAMES,
)

#: Two-qubit members of qibo's NativeGates -- the scope of the register.
TWO_QUBIT_NATIVE_GATES = ("CZ", "iSWAP", "CNOT")


def _is_exchange_symmetric(name: str) -> bool:
    qc = QuantumCircuit(2)
    qc.append(QIBO_TO_QISKIT[name](), [0, 1])
    return Operator(qc).equiv(Operator(qc.reverse_bits()))


def test_register_matches_qibolab_two_qubit_natives_metadata():
    from qibolab._core.native import TwoQubitNatives

    qibolab_symmetric = {
        name for name, info in TwoQubitNatives.model_fields.items() if info.metadata[0]["symmetric"]
    }
    assert SYMMETRIC_TWO_QUBIT_NATIVE_GATES == qibolab_symmetric


def test_register_is_within_qibo_native_gates():
    # NativeGates[name] is NONE (falsy) for a non-member, as in default_transpile.
    assert all(NativeGates[name] for name in TWO_QUBIT_NATIVE_GATES)
    assert SYMMETRIC_TWO_QUBIT_NATIVE_GATES <= set(TWO_QUBIT_NATIVE_GATES)


@pytest.mark.parametrize("name", TWO_QUBIT_NATIVE_GATES)
def test_register_membership_matches_unitary_exchange_symmetry(name):
    assert _is_exchange_symmetric(name) == (name in SYMMETRIC_TWO_QUBIT_NATIVE_GATES)


def test_operation_names():
    assert SYMMETRIC_TWO_QUBIT_OPERATION_NAMES == {"cz", "iswap"}
