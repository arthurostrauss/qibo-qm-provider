"""Tests for qibo_qm_provider.backend.qibo_qm_backend.QiboQMBackend.

No live QOP hardware: qmm/_qm are left None throughout (QMBackend's own
lazy-connection properties are never touched by these tests).
"""

import pytest

from qibo_qm_provider import QiboQMBackend


@pytest.fixture
def backend(dummy_machine):
    return QiboQMBackend(dummy_machine)


def test_construction_wraps_qiskit_backend(backend, dummy_machine):
    from qiskit_qm_provider.backend.qm_backend import QMBackend

    assert isinstance(backend.qiskit_backend, QMBackend)
    assert backend.qiskit_backend.machine is dummy_machine


def test_from_qiskit_backend_wraps_without_reconstruction(dummy_machine):
    from qiskit_qm_provider.backend.qm_backend import QMBackend

    qiskit_backend = QMBackend(dummy_machine)
    backend = QiboQMBackend.from_qiskit_backend(qiskit_backend)

    assert backend.qiskit_backend is qiskit_backend


def test_qubits_and_connectivity_reflect_wrapped_backend(backend):
    assert backend.qubits == list(backend.qiskit_backend.qubit_dict.values())
    assert backend.connectivity == list(backend.qiskit_backend.qubit_pair_dict.values())


def test_natives_reflect_wrapped_target_after_add_basic_macros(add_basic_macros_installed):
    backend = QiboQMBackend(add_basic_macros_installed)
    backend.qiskit_backend.update_target()

    assert set(backend.natives) == set(backend.qiskit_backend.target.operation_names)
    assert len(backend.natives) > 0


def test_cz_macro_installs_cleanly(add_basic_macros_installed):
    """Regression check for the flux_pulse_control/flux_pulse_qubit version
    mismatch fixed in qiskit-qm-provider 0.3.3 -- previously add_basic_macros
    silently left qubit_pairs["q0-q1"].macros["cz"] absent (or dangling
    None) on this fixture; it must now install a real CZGate."""
    from quam_builder.architecture.superconducting.custom_gates.flux_tunable_transmon_pair.two_qubit_gates import (
        CZGate,
    )

    cz_macro = add_basic_macros_installed.qubit_pairs["q0-q1"].macros.get("cz")
    assert isinstance(cz_macro, CZGate)

    backend = QiboQMBackend(add_basic_macros_installed)
    assert "cz" in backend.natives


def test_apply_gate_raises_not_implemented(backend):
    with pytest.raises(NotImplementedError):
        backend.apply_gate(None)
    with pytest.raises(NotImplementedError):
        backend.apply_gate_density_matrix(None)


def test_execute_circuit_rejects_non_circuit_initial_state(backend):
    from qibo import Circuit, gates

    circuit = Circuit(1)
    circuit.add(gates.M(0))
    with pytest.raises(ValueError):
        backend.execute_circuit(circuit, initial_state=[0, 1])


def test_update_target_delegates_to_wrapped_backend(backend, dummy_machine):
    # Manually install a macro (bypassing register_gate) then confirm the
    # thin update_target() delegate actually resyncs the registry.
    from quam.components.macro import PulseMacro

    q0 = dummy_machine.qubits["q0"]
    assert "foo" not in backend.natives
    q0.macros["foo"] = PulseMacro(pulse="x180")

    backend.update_target()

    assert "foo" in backend.natives


def test_register_gate_installs_macro_on_single_qubit_and_resyncs(backend, dummy_machine):
    from quam.components.macro import PulseMacro

    assert "foo" not in backend.natives

    backend.register_gate("foo", 0, PulseMacro(pulse="x180"))

    assert dummy_machine.qubits["q0"].macros["foo"] is not None
    assert "foo" in backend.natives  # resynced automatically, no manual update_target() needed


def test_register_gate_installs_macro_on_qubit_pair(backend, dummy_machine):
    from quam.components.macro import PulseMacro

    pair = dummy_machine.qubit_pairs["q0-q1"]
    backend.register_gate("bar", (0, 1), PulseMacro(pulse="#/qubits/q0/z/operations/const"))

    assert pair.macros["bar"] is not None
    assert "bar" in backend.natives
