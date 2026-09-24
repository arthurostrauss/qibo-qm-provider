"""GPI2Macro: decomposition into the qubit's sx/rz macros, and its seeding by
``add_basic_macros``."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from qibo_qm_provider import QiboQMBackend, add_basic_macros
from qibo_qm_provider.quam_macros.superconducting import GPI2Macro, ZMacro


@pytest.mark.parametrize("phi", [0.0, 0.37, -1.2, np.pi / 2, 2.9])
def test_decomposition_matches_qibo_gpi2_up_to_global_phase(phi):
    """rz(-phi) -> sx -> rz(phi) (time order) is Qibo's GPI2(phi)."""
    from qibo import gates
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    qc = QuantumCircuit(1)
    qc.rz(-phi, 0)
    qc.sx(0)
    qc.rz(phi, 0)

    assert Operator(qc).equiv(Operator(gates.GPI2(0, phi).matrix()))


def test_apply_delegates_to_rz_and_sx_macros_in_order(add_basic_macros_installed):
    from quam.components.macro import QubitMacro
    from quam.core import quam_dataclass

    calls: list = []

    @quam_dataclass
    class RecordingRZ(QubitMacro):
        def apply(self, angle, **kwargs):
            calls.append(("rz", angle))

    @quam_dataclass
    class RecordingSX(QubitMacro):
        def apply(self, **kwargs):
            calls.append(("sx",))

    qubit = add_basic_macros_installed.qubits["q0"]
    qubit.macros["rz"] = RecordingRZ()
    qubit.macros["sx"] = RecordingSX()

    qubit.macros["gpi2"].apply(0.4)

    assert calls == [("rz", -0.4), ("sx",), ("rz", 0.4)]


def test_add_basic_macros_installs_gpi2_and_z_on_every_active_qubit(add_basic_macros_installed):
    for qubit in add_basic_macros_installed.active_qubits:
        assert isinstance(qubit.macros["gpi2"], GPI2Macro)
        assert isinstance(qubit.macros["z"], ZMacro)


def test_z_macro_decomposition_matches_qibo_z_up_to_global_phase():
    from qibo import gates
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    qc = QuantumCircuit(1)
    qc.rz(np.pi, 0)
    assert Operator(qc).equiv(Operator(gates.Z(0).matrix()))


def test_z_macro_delegates_to_rz_macro(add_basic_macros_installed):
    from quam.components.macro import QubitMacro
    from quam.core import quam_dataclass

    calls: list = []

    @quam_dataclass
    class RecordingRZ(QubitMacro):
        def apply(self, angle, **kwargs):
            calls.append(angle)

    qubit = add_basic_macros_installed.qubits["q0"]
    qubit.macros["rz"] = RecordingRZ()
    qubit.macros["z"].apply()

    assert calls == [np.pi]


def test_add_basic_macros_tops_up_already_seeded_machine_without_overwriting(add_basic_macros_installed):
    machine = add_basic_macros_installed
    q0, q1 = machine.qubits["q0"], machine.qubits["q1"]
    custom = GPI2Macro(sx_macro="x")
    q0.macros["gpi2"] = custom
    q1.macros.pop("gpi2")

    add_basic_macros(machine)

    assert q0.macros["gpi2"] is custom
    assert isinstance(q1.macros["gpi2"], GPI2Macro)


def test_add_basic_macros_refreshes_backend_target_with_gpi2(dummy_machine):
    backend = QiboQMBackend(dummy_machine)
    assert "GPI2" not in backend.natives

    add_basic_macros(backend)

    assert "gpi2" in backend.qiskit_backend.target.operation_names
    assert "GPI2" in backend.natives


def test_symbolic_gpi2_compiles_to_one_pulse_and_two_frame_rotations(add_basic_macros_installed):
    from qibo import Circuit, gates
    from qm import generate_qua_script
    from qm.qua import program

    backend = QiboQMBackend(add_basic_macros_installed)
    circuit = Circuit(1)
    circuit.add(gates.GPI2(0, phi=sp.Symbol("phi")))

    with program() as prog:
        backend.circuit_to_qua(circuit)
    script = generate_qua_script(prog)

    assert sum("frame_rotation_2pi" in line for line in script.splitlines()) == 2, script
    assert sum("play('x90'" in line for line in script.splitlines()) == 1, script


def test_default_transpile_lowers_h_onto_gpi2(add_basic_macros_installed):
    """With gpi2 installed, a non-native single-qubit gate like H decomposes
    onto the machine's natives instead of raising UnsupportedGateError."""
    from qibo import Circuit, gates
    from qm import generate_qua_script
    from qm.qua import program

    backend = QiboQMBackend(add_basic_macros_installed)
    circuit = Circuit(1)
    circuit.add(gates.H(0))

    with pytest.warns(UserWarning):
        transpiled = backend._default_transpile(circuit)
    names = {gate.__class__.__name__ for gate in transpiled.queue}
    assert "GPI2" in names
    # qibo's gpi2_dec emits a bare Z (H -> [Z, GPI2(pi/2)]), backed by ZMacro.
    assert names <= set(backend.natives)

    with program() as prog:
        backend.circuit_to_qua(transpiled)
    assert "play('x90'" in generate_qua_script(prog)


def test_default_transpile_rejects_decomposition_into_non_native_gates(add_basic_macros_installed):
    """Without a z macro, qibo's H -> [Z, GPI2] rule emits a gate the machine
    cannot run: that must be an UnsupportedGateError at transpile time, not a
    qm_qasm failure at QUA compilation."""
    from qibo import Circuit, gates
    from qibo_qm_provider.exceptions import UnsupportedGateError

    for qubit in add_basic_macros_installed.active_qubits:
        qubit.macros.pop("z")
    backend = QiboQMBackend(add_basic_macros_installed)
    assert "GPI2" in backend.natives and "Z" not in backend.natives

    circuit = Circuit(1)
    circuit.add(gates.H(0))
    with pytest.raises(UnsupportedGateError, match=r"H .*\['Z'\]"):
        backend._default_transpile(circuit)
