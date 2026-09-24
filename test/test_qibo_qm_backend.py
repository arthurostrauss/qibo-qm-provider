"""Tests for qibo_qm_provider.backend.qibo_qm_backend.QiboQMBackend.

No live QOP hardware: qmm/_qm are left None throughout (QMBackend's own
lazy-connection properties are never touched by these tests).
"""

from unittest.mock import patch

import pytest
import sympy as sp

from qibo_qm_provider import QiboParameterTable, QiboQMBackend
from qibo_qm_provider.exceptions import UnsupportedConnectivityError, UnsupportedParameterError


@pytest.fixture
def backend(dummy_machine):
    return QiboQMBackend(dummy_machine)


def _install_gpi2_macro(backend, machine, qubit="q0"):
    """Give ``machine``/``backend`` a native single-qubit-universal family
    (GPI2), so a plain H has something to be decomposed into -- neither
    add_basic_macros_installed nor dummy_machine installs one by default
    (see test_natives_reflect_wrapped_target_after_add_basic_macros)."""
    from quam.components.macro import QubitMacro
    from quam.core import quam_dataclass

    @quam_dataclass
    class _NoOpGPI2(QubitMacro):
        def apply(self, phi, **kwargs):
            pass

    machine.qubits[qubit].macros["gpi2"] = _NoOpGPI2()
    backend.update_target()


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

    # add_basic_macros installs id/x/sx/sy/sydg/rz/z/gpi2/measure/reset/delay
    # on every qubit and cz on the pair -- natives is the shared Enum∩QuAM-
    # macro set (issue #6), so only id/z/rz/gpi2/measure/cz (-> I/Z/RZ/GPI2/
    # M/CZ) survive; x/sx/sy/sydg/reset/delay have no Enum-compatible
    # NativeGates∩qibolab-compiler entry and are correctly absent, unlike the
    # raw (unfiltered) target.operation_names.
    assert set(backend.natives) == {"I", "Z", "RZ", "GPI2", "M", "CZ"}
    # The raw, unfiltered view still carries the rest (x/sx/sy/... plus
    # control-flow op names), confirming natives is a real filter, not a
    # renaming of the whole set.
    assert len(backend.qiskit_backend.target.operation_names) > len(backend.natives)


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
    assert "CZ" in backend.natives


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


def test_execute_circuit_rejects_symbolic_circuit(add_basic_macros_installed):
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=sp.Symbol("theta")))

    with pytest.raises(ValueError, match="circuit_to_qua"):
        backend.execute_circuit(circuit)


# --------------------------------------------------------------------------- #
# circuit_to_qua
# --------------------------------------------------------------------------- #


def test_circuit_to_qua_builds_table_and_compiles_once(add_basic_macros_installed):
    """The one-call entry point: converts once, builds a table from the
    same object, and produces the same live-QUA-variable frame rotation as
    the manual three-step composition in test_symbolic_lowering.py."""
    from qm import generate_qua_script
    from qm.qua import program
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=sp.Symbol("theta")))

    with program() as prog:
        backend.circuit_to_qua(circuit)
    script = generate_qua_script(prog)

    rotations = [line.strip() for line in script.splitlines() if "frame_rotation_2pi" in line]
    assert len(rotations) == 1, script
    assert "declare(fixed" in script


def test_circuit_to_qua_accepts_explicit_param_table(add_basic_macros_installed):
    """Two circuits sharing one pre-built table -- the motivating use case
    for accepting param_table explicitly instead of always rebuilding it.

    Each ``quantum_circuit_to_qua`` call makes its own local copy of the
    table's variable (``assign(v2, v1)``, ``assign(v3, v1)`` below), so
    "one shared table" shows up as both copies being assigned *from* the
    same source variable, not as a single ``declare(fixed`` in the script.
    """
    import re

    from qm import generate_qua_script
    from qm.qua import program
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)
    theta = sp.Symbol("theta")
    circuit_a = Circuit(1)
    circuit_a.add(gates.RZ(0, theta=theta))
    circuit_b = Circuit(1)
    circuit_b.add(gates.RZ(0, theta=2 * theta))

    table = QiboParameterTable.from_qibo_circuit(circuit_a)

    with program() as prog:
        table.declare()
        backend.circuit_to_qua(circuit_a, param_table=table)
        backend.circuit_to_qua(circuit_b, param_table=table)
    script = generate_qua_script(prog)

    sources = re.findall(r"assign\(v\d+, (v\d+)\)", script)
    assert len(sources) == 2, script
    assert sources[0] == sources[1], script


def test_circuit_to_qua_rejects_parameter_name_colliding_with_machine_gate(add_basic_macros_installed):
    """The precise version of the lower-bound check: a parameter named after
    a macro installed on the machine (here, "x") but not used by *this*
    circuit still collides, because Exporter's basis_gates comes from the
    machine's whole Target. test_symbolic_lowering.py's
    test_collision_check_is_scoped_to_gates_actually_used pins that such a
    name is accepted when there is no machine to check against; this is the
    complementary case where there is one."""
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)
    assert "x" in backend.qiskit_backend.qm_qasm_basis_gates

    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=sp.Symbol("x")))

    with pytest.raises(UnsupportedParameterError, match="collides"):
        backend.circuit_to_qua(circuit)


# --------------------------------------------------------------------------- #
# Two-qubit connectivity direction
#
# Root-caused by a real "arbel" live-hardware failure: CZ(0,1) raised
# qm_qasm's opaque UnresolvableOperation because the installed macro's flux
# pulse only plays in one physical direction ("arbel"'s qA2-qA1 pair is only
# registered as (control=qA2, target=qA1), never the reverse). On the
# add_basic_macros_installed fixture, "q0-q1" has qubit_control="q0"
# (index 0), qubit_target="q1" (index 1), so CZ(0,1) is the one direction
# that is actually installed.
#
# CZ/iSWAP are unitarily symmetric under exchanging the qubits
# (gate_map.SYMMETRIC_TWO_QUBIT_NATIVE_GATES), so a reversed one is reordered
# onto the registered direction on every path; direction is only enforced for
# genuinely asymmetric gates (CNOT -> "cx" below).
# --------------------------------------------------------------------------- #


def test_circuit_to_qua_compiles_valid_cz_direction(add_basic_macros_installed):
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)
    assert backend.qiskit_backend.qubit_pair_dict["q0-q1"] == (0, 1)

    circuit = Circuit(2)
    circuit.add(gates.CZ(0, 1))

    backend.circuit_to_qua(circuit)  # must not raise


def test_circuit_to_qua_reorders_reversed_cz_direction(add_basic_macros_installed):
    """The "arbel" case: CZ written in the uncalibrated order is the same
    unitary as the calibrated one, so it compiles onto the installed macro
    instead of raising."""
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)

    circuit = Circuit(2)
    circuit.add(gates.CZ(1, 0))

    backend.circuit_to_qua(circuit)  # must not raise


def test_circuit_to_qua_compiles_qibo_swap_decomposition(add_basic_macros_installed):
    """Qibo's CZ-based SWAP rule contains CZ(0, 1) *and* CZ(1, 0) -- no
    wire_names assignment can satisfy both, so this only compiles because
    symmetric gates are reordered."""
    import warnings

    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)
    circuit = Circuit(2)
    circuit.add(gates.SWAP(0, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transpiled = backend._default_transpile(circuit)
    assert {gate.qubits for gate in transpiled.queue if isinstance(gate, gates.CZ)} == {(0, 1), (1, 0)}

    backend.circuit_to_qua(transpiled)  # must not raise


def test_circuit_to_qua_rejects_reversed_asymmetric_gate_direction(add_basic_macros_installed):
    """Direction is still enforced for an asymmetric gate: a clear,
    actionable error naming the correct direction, not qm_qasm's opaque
    UnresolvableOperation surfacing from deep inside the compiler."""
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)

    circuit = Circuit(2)
    circuit.add(gates.CNOT(1, 0))

    with pytest.raises(UnsupportedConnectivityError, match=r"cx\(q1, q0\).*cx\(q0, q1\)"):
        backend.circuit_to_qua(circuit)


def test_execute_circuit_default_layout_auto_corrects_reversed_cz_direction(add_basic_macros_installed):
    """With ``circuit.wire_names`` left unset (no explicit placement
    requested), ``execute_circuit``'s default (``transpile=True``) automatic
    layout step now runs a real ``qiskit.compiler.transpile(qc,
    backend=...)`` against the wrapped machine's actual ``Target`` -- since
    ``CZ`` is unitarily symmetric, Qiskit's own transpiler corrects the
    gate's qubit order to the one direction that is actually registered,
    rather than requiring the caller to have written it correctly by hand
    (the old "arbel" failure mode this package used to just reject)."""
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)
    assert backend.qiskit_backend.qubit_pair_dict["q0-q1"] == (0, 1)

    circuit = Circuit(2)
    circuit.add(gates.CZ(1, 0))
    circuit.add(gates.M(0, 1))

    with (
        patch.object(backend.qiskit_backend, "run") as mock_run,
        patch("qibo_qm_provider.backend.qibo_qm_backend.translate_measurements"),
    ):
        backend.execute_circuit(circuit)  # must not raise

    (qc,), _ = mock_run.call_args
    cz_instr = next(instr for instr in qc.data if instr.operation.name == "cz")
    assert [qc.find_bit(q).index for q in cz_instr.qubits] == [0, 1]


def test_execute_circuit_transpile_false_leaves_reversed_asymmetric_direction_unfixed(add_basic_macros_installed):
    """``transpile=False`` opts out of *all* automatic behaviour, not just
    Qibo-level gate decomposition -- including the automatic-layout step
    above, so a caller who wants full manual control still gets exactly
    that (the same "compile it verbatim, fail later if it's wrong" contract
    this parameter has always documented). Only a reversed *asymmetric* gate
    can show this now -- reordering a symmetric one is exact, not a
    transpile step, and happens regardless."""
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)

    circuit = Circuit(2)
    circuit.add(gates.CNOT(1, 0))
    circuit.add(gates.M(0, 1))

    with pytest.raises(UnsupportedConnectivityError):
        backend.execute_circuit(circuit, transpile=False)


def test_execute_circuit_transpile_false_still_reorders_reversed_cz(add_basic_macros_installed):
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)
    circuit = Circuit(2)
    circuit.add(gates.CZ(1, 0))
    circuit.add(gates.M(0, 1))

    with (
        patch.object(backend.qiskit_backend, "run") as mock_run,
        patch("qibo_qm_provider.backend.qibo_qm_backend.translate_measurements"),
    ):
        backend.execute_circuit(circuit, transpile=False)

    (qc,), _ = mock_run.call_args
    cz_instr = next(instr for instr in qc.data if instr.operation.name == "cz")
    assert [qc.find_bit(q).index for q in cz_instr.qubits] == [0, 1]


def test_circuit_to_qua_wire_names_routes_logical_qubits_to_the_calibrated_direction(add_basic_macros_installed):
    """circuit.wire_names lets a caller route logical qubits onto whichever
    physical qubits/direction is actually calibrated -- the fix for the
    "arbel" failure, which had no such lever available at all."""
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)

    # logical 0 -> "q1", logical 1 -> "q0": CNOT(0,1) (logical) now addresses
    # physical (1, 0) -- the direction that is NOT installed. (An asymmetric
    # gate: a reversed CZ would simply be reordered.)
    wrong_direction = Circuit(2, wire_names=["q1", "q0"])
    wrong_direction.add(gates.CNOT(0, 1))
    with pytest.raises(UnsupportedConnectivityError):
        backend.circuit_to_qua(wrong_direction)

    # Swapping the gate's own argument order compensates, landing back on
    # the calibrated physical (0, 1) direction -- which fails only for the
    # missing cx macro now, not for direction.
    right_direction = Circuit(2, wire_names=["q1", "q0"])
    right_direction.add(gates.CNOT(1, 0))
    from qibo_qm_provider.backend.circuit_conversion import (
        qibo_circuit_to_qiskit,
        validate_two_qubit_connectivity,
    )

    qc = qibo_circuit_to_qiskit(right_direction, qubit_dict=backend.qiskit_backend.qubit_dict)
    validate_two_qubit_connectivity(qc, backend.qiskit_backend.qubit_pair_dict)  # must not raise


# --------------------------------------------------------------------------- #
# execute_circuit's default transpile step (issue #4)
#
# Bug #3's root cause: a plain H reaching the OpenQASM3 exporter with no
# registered macro. `run()` is patched out below (it needs a live QM
# connection) purely to capture the QuantumCircuit execute_circuit hands it,
# proving the decomposition happened before compiling.
# --------------------------------------------------------------------------- #


def test_execute_circuit_transpiles_a_non_native_gate_by_default(add_basic_macros_installed):
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)
    _install_gpi2_macro(backend, add_basic_macros_installed)
    assert "H" not in backend.natives

    circuit = Circuit(1)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))

    with (
        patch.object(backend.qiskit_backend, "run") as mock_run,
        patch("qibo_qm_provider.backend.qibo_qm_backend.translate_measurements"),
    ):
        with pytest.warns(UserWarning, match="H"):
            backend.execute_circuit(circuit)

    (qc,), _ = mock_run.call_args
    assert "h" not in [instr.operation.name for instr in qc.data]


def test_execute_circuit_transpile_false_leaves_a_non_native_gate_untouched(add_basic_macros_installed):
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)
    _install_gpi2_macro(backend, add_basic_macros_installed)

    circuit = Circuit(1)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))

    with (
        patch.object(backend.qiskit_backend, "run") as mock_run,
        patch("qibo_qm_provider.backend.qibo_qm_backend.translate_measurements"),
    ):
        backend.execute_circuit(circuit, transpile=False)

    (qc,), _ = mock_run.call_args
    assert [instr.operation.name for instr in qc.data] == ["h", "measure"]


def test_execute_circuit_already_native_gate_is_unaffected_by_transpile(add_basic_macros_installed):
    import warnings

    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)
    assert "CZ" in backend.natives

    circuit = Circuit(2)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.M(0, 1))

    with (
        patch.object(backend.qiskit_backend, "run") as mock_run,
        patch("qibo_qm_provider.backend.qibo_qm_backend.translate_measurements"),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            backend.execute_circuit(circuit)

    # Scoped to UserWarning -- this package's own decomposition-notice
    # channel (see default_transpile.default_transpile) -- rather than
    # every warning of any kind, since the automatic layout step now calls
    # qiskit.compiler.transpile(), which can trigger unrelated third-party
    # DeprecationWarnings (e.g. lazy plugin discovery) with no bearing on
    # whether *this* already-native, already-correctly-directed circuit was
    # left alone.
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    (qc,), _ = mock_run.call_args
    assert [instr.operation.name for instr in qc.data] == ["cz", "measure", "measure"]


def test_execute_circuit_non_native_non_decomposable_gate_raises_actionable_error(add_basic_macros_installed):
    """Without GPI2/U3 (add_basic_macros' gpi2 removed here) H genuinely
    cannot be decomposed, and that must surface as UnsupportedGateError, not
    a bare upstream DecompositionError."""
    from qibo import Circuit, gates
    from qibo_qm_provider.exceptions import UnsupportedGateError

    for qubit in add_basic_macros_installed.active_qubits:
        qubit.macros.pop("gpi2")
    backend = QiboQMBackend(add_basic_macros_installed)
    assert "GPI2" not in backend.natives and "U3" not in backend.natives

    circuit = Circuit(1)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))

    with pytest.raises(UnsupportedGateError, match="H"):
        backend.execute_circuit(circuit)


def test_execute_circuits_rejects_non_circuit_initial_state(backend):
    from qibo import Circuit, gates

    circuit = Circuit(1)
    circuit.add(gates.M(0))
    with pytest.raises(ValueError):
        backend.execute_circuits([circuit], initial_state=[0, 1])


def test_execute_circuits_rejects_symbolic_circuit(add_basic_macros_installed):
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=sp.Symbol("theta")))

    with pytest.raises(ValueError, match="circuit_to_qua"):
        backend.execute_circuits([circuit])


def test_execute_circuits_rejects_reversed_asymmetric_direction(add_basic_macros_installed):
    """The connectivity check must fire before execute_circuits ever
    attempts to submit a job (no live QM connection is set up in this
    fixture). ``transpile=False`` keeps the CNOT from being decomposed into
    (reorderable) CZs first."""
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)

    circuit = Circuit(2)
    circuit.add(gates.CNOT(1, 0))
    circuit.add(gates.M(0, 1))

    with pytest.raises(UnsupportedConnectivityError):
        backend.execute_circuits([circuit], transpile=False)


def test_execute_circuits_reorders_reversed_cz_in_every_circuit(add_basic_macros_installed):
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)
    circuits = []
    for _ in range(2):
        circuit = Circuit(2)
        circuit.add(gates.CZ(1, 0))
        circuit.add(gates.M(0, 1))
        circuits.append(circuit)

    with (
        patch.object(backend.qiskit_backend, "run") as mock_run,
        patch("qibo_qm_provider.backend.qibo_qm_backend.translate_measurements"),
    ):
        backend.execute_circuits(circuits, transpile=False)

    (qcs,), _ = mock_run.call_args
    for qc in qcs:
        cz_instr = next(instr for instr in qc.data if instr.operation.name == "cz")
        assert [qc.find_bit(q).index for q in cz_instr.qubits] == [0, 1]


def test_execute_circuits_submits_one_job_for_the_whole_batch(add_basic_macros_installed):
    """The point of execute_circuits over calling execute_circuit in a loop:
    every circuit is submitted together as a single QMBackend.run(...) job,
    not one job per circuit."""
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)

    circuits = [Circuit(2) for _ in range(3)]
    for circuit in circuits:
        circuit.add(gates.CZ(0, 1))
        circuit.add(gates.M(0, 1))

    with (
        patch.object(backend.qiskit_backend, "run") as mock_run,
        patch("qibo_qm_provider.backend.qibo_qm_backend.translate_measurements"),
    ):
        backend.execute_circuits(circuits)

    assert mock_run.call_count == 1
    (qcs,), kwargs = mock_run.call_args
    assert len(qcs) == 3
    assert kwargs["memory"] is True


def test_execute_circuits_translates_each_circuit_with_its_own_experiment_index(
    add_basic_macros_installed,
):
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)

    circuits = [Circuit(2) for _ in range(3)]
    for circuit in circuits:
        circuit.add(gates.CZ(0, 1))
        circuit.add(gates.M(0, 1))

    with (
        patch.object(backend.qiskit_backend, "run") as mock_run,
        patch(
            "qibo_qm_provider.backend.qibo_qm_backend.translate_measurements"
        ) as mock_translate,
    ):
        outcomes = backend.execute_circuits(circuits)

    assert len(outcomes) == 3
    assert [call.kwargs["experiment_index"] for call in mock_translate.call_args_list] == [
        0,
        1,
        2,
    ]


def test_execute_circuits_transpiles_non_native_gates_by_default(add_basic_macros_installed):
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)
    _install_gpi2_macro(backend, add_basic_macros_installed)
    assert "H" not in backend.natives

    circuits = []
    for _ in range(2):
        circuit = Circuit(1)
        circuit.add(gates.H(0))
        circuit.add(gates.M(0))
        circuits.append(circuit)

    with (
        patch.object(backend.qiskit_backend, "run") as mock_run,
        patch("qibo_qm_provider.backend.qibo_qm_backend.translate_measurements"),
    ):
        with pytest.warns(UserWarning, match="H"):
            backend.execute_circuits(circuits)

    (qcs,), _ = mock_run.call_args
    for qc in qcs:
        assert "h" not in [instr.operation.name for instr in qc.data]


def test_execute_circuits_prepends_initial_state_to_every_circuit(add_basic_macros_installed):
    from qibo import Circuit, gates

    backend = QiboQMBackend(add_basic_macros_installed)

    initial_state = Circuit(2)
    initial_state.add(gates.I(0))

    circuits = [Circuit(2) for _ in range(2)]
    for circuit in circuits:
        circuit.add(gates.CZ(0, 1))
        circuit.add(gates.M(0, 1))

    with (
        patch.object(backend.qiskit_backend, "run") as mock_run,
        patch("qibo_qm_provider.backend.qibo_qm_backend.translate_measurements"),
    ):
        backend.execute_circuits(circuits, initial_state=initial_state)

    (qcs,), _ = mock_run.call_args
    for qc in qcs:
        assert [instr.operation.name for instr in qc.data] == ["id", "cz", "measure", "measure"]


def test_update_target_delegates_to_wrapped_backend(backend, dummy_machine):
    # Manually install a macro (bypassing register_gate) then confirm the
    # thin update_target() delegate actually resyncs the registry. "foo" has
    # no Qibo-gate meaning, so this is checked against the wrapped backend's
    # raw operation-name view, not the Qibo-native-filtered `natives`.
    from quam.components.macro import PulseMacro

    q0 = dummy_machine.qubits["q0"]
    assert "foo" not in backend.qiskit_backend.target.operation_names
    q0.macros["foo"] = PulseMacro(pulse="x180")

    backend.update_target()

    assert "foo" in backend.qiskit_backend.target.operation_names


def test_register_gate_installs_macro_on_single_qubit_and_resyncs(backend, dummy_machine):
    from quam.components.macro import PulseMacro

    assert "foo" not in backend.qiskit_backend.target.operation_names

    backend.register_gate("foo", 0, PulseMacro(pulse="x180"))

    assert dummy_machine.qubits["q0"].macros["foo"] is not None
    # resynced automatically, no manual update_target() needed
    assert "foo" in backend.qiskit_backend.target.operation_names


def test_register_gate_installs_macro_on_qubit_pair(backend, dummy_machine):
    from quam.components.macro import PulseMacro

    pair = dummy_machine.qubit_pairs["q0-q1"]
    backend.register_gate("bar", (0, 1), PulseMacro(pulse="#/qubits/q0/z/operations/const"))

    assert pair.macros["bar"] is not None
    assert "bar" in backend.qiskit_backend.target.operation_names


def test_register_gate_overwrites_an_existing_name_without_warning(backend, dummy_machine):
    """Since qiskit-qm-provider 0.3.5 (operation_key cache fix), overwriting an
    existing macro name reaches the compiler, so there is nothing to warn about."""
    import warnings

    from quam.components.macro import PulseMacro

    backend.register_gate("foo", 0, PulseMacro(pulse="x180"))
    replacement = PulseMacro(pulse="x180")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend.register_gate("foo", 0, replacement)
    assert dummy_machine.qubits["q0"].macros["foo"] is replacement


def test_natives_match_across_backends_for_same_machine(add_basic_macros_installed):
    """Issue #6: both backends share the Enum∩QuAM-macro natives definition."""
    from qibo_qm_provider import QiboQMBackend, QiboQMPlatformBackend

    qm = QiboQMBackend(add_basic_macros_installed)
    qm.qiskit_backend.update_target()
    platform = QiboQMPlatformBackend.from_machine(add_basic_macros_installed)

    assert set(qm.natives) == set(platform.natives) == {"I", "Z", "RZ", "GPI2", "M", "CZ"}
