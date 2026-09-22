"""Tests for qibo_qm_provider.backend.qibo_qm_platform_backend.QiboQMPlatformBackend
and the qibo_qm_provider.MetaBackend entrypoint.

Does not test actual pulse execution/``.connect()`` -- those are inherited
unchanged from qibolab's ``QibolabBackend`` and are exercised by qibolab's
own test suite. The fixtures used here (dummy_machine/add_basic_macros_installed)
are Octave-less IQ machines with no `network`, so Platform.instruments comes
back empty for them regardless (see test_quam_wiring.py for the
working-instruments path, using the mw_fem_machine fixture). ``execute_circuit``
itself *is* tested here (see the "default transpile" section below), since
this class overrides it to add a default gate-decomposition step -- the
inherited compile/execute call it delegates to afterwards is what is left
untested. Otherwise this file only proves QiboQMPlatformBackend's own
additions: construction, refresh(), and entrypoint wiring.
"""

from unittest.mock import patch

import pytest

import qibo_qm_provider
from qibo_qm_provider import QiboQMPlatformBackend


def test_from_machine_sets_machine_and_platform(add_basic_macros_installed):
    backend = QiboQMPlatformBackend.from_machine(add_basic_macros_installed)

    assert backend.machine is add_basic_macros_installed
    assert set(backend.qubits) == {"q0", "q1"}


def test_refresh_picks_up_machine_mutation(add_basic_macros_installed):
    backend = QiboQMPlatformBackend.from_machine(add_basic_macros_installed)

    add_basic_macros_installed.qubits["q0"].xy.operations["x180"].amplitude = 0.42
    backend.refresh()

    # dummy_machine's ports are bare tuples with no output_mode, so the
    # default 0.5 V "direct" max_voltage applies: 0.42 V -> 0.84 dimensionless.
    _, pulse = backend.platform.natives.single_qubit["q0"].RX[0]
    assert pulse.amplitude == pytest.approx(0.84)


def test_refresh_without_machine_raises(add_basic_macros_installed):
    platform = QiboQMPlatformBackend.from_machine(add_basic_macros_installed).platform
    bare = QiboQMPlatformBackend(platform=platform)

    assert bare.machine is None
    with pytest.raises(RuntimeError, match="from_iqcc/from_local/from_machine"):
        bare.refresh()


def test_meta_backend_load_wraps_a_platform_object_directly(add_basic_macros_installed):
    """Cheapest correctness check of the qibo.set_backend("qibo_qm_provider",
    ...) entrypoint wiring: no $QIBOLAB_PLATFORMS filesystem setup needed,
    since MetaBackend.load accepts an already-built Platform object just
    like QibolabBackend itself does."""
    platform = QiboQMPlatformBackend.from_machine(add_basic_macros_installed).platform

    backend = qibo_qm_provider.MetaBackend.load(platform=platform)

    assert isinstance(backend, QiboQMPlatformBackend)
    assert backend.platform is platform
    assert backend.machine is None  # MetaBackend.load has no machine to pass through


def test_set_backend_resolves_to_qibo_qm_platform_backend(add_basic_macros_installed):
    import qibo
    from qibo.backends import _Global

    platform = QiboQMPlatformBackend.from_machine(add_basic_macros_installed).platform
    try:
        qibo.set_backend("qibo_qm_provider", platform=platform)
        assert isinstance(_Global.backend(), QiboQMPlatformBackend)
    finally:
        qibo.set_backend("numpy")  # don't leak the global backend into other tests


# ---------------------------------------------------------------------------
# sequence_to_qua_macro / circuit_to_qua_macro
#
# Uses mw_fem_machine (working-instruments path -- see its docstring) rather
# than add_basic_macros_installed, since these methods need real QuAM
# channels to emit QUA against. mw_fem_machine only has a "measure" macro
# installed by default (no "x"/"sx"), so an "x" PulseMacro pointing at the
# already-registered "x180" pulse is added where an RX native is needed.
# ---------------------------------------------------------------------------


def test_sequence_to_qua_macro_requires_machine(mw_fem_machine):
    import warnings

    from qibolab._core.pulses.envelope import Rectangular
    from qibolab._core.pulses.pulse import Pulse
    from qibolab._core.sequence import PulseSequence

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        platform = QiboQMPlatformBackend.from_machine(mw_fem_machine).platform
    bare = QiboQMPlatformBackend(platform=platform)

    sequence = PulseSequence([("mw0/drive", Pulse(duration=40, amplitude=0.2, envelope=Rectangular()))])
    with pytest.raises(RuntimeError, match="from_iqcc/from_local/from_machine"):
        bare.sequence_to_qua_macro(sequence)


def test_sequence_to_qua_macro_emits_against_backend_machine(mw_fem_machine):
    import warnings

    from qm import qua
    from qibolab._core.pulses.envelope import Rectangular
    from qibolab._core.pulses.pulse import Pulse
    from qibolab._core.sequence import PulseSequence

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backend = QiboQMPlatformBackend.from_machine(mw_fem_machine)

    sequence = PulseSequence([("mw0/drive", Pulse(duration=40, amplitude=0.2, envelope=Rectangular()))])
    with qua.program():
        macro = backend.sequence_to_qua_macro(sequence)
        macro()  # should not raise


def test_circuit_to_qua_macro_compiles_and_emits(mw_fem_machine):
    """Uses a bare M(0) circuit -- mw_fem_machine only has an MZ native
    (via its "measure" macro), not RX (qibolab's own Compiler.rules has no
    rule for gates.X directly; exercising a native-gate rule beyond
    measurement_rule would need an RX-native-bearing machine, already
    covered indirectly by test_sequence_to_qua_macro's Pulse-level tests)."""
    import warnings

    from qm import qua
    from qibo.models import Circuit
    from qibo.gates import M

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backend = QiboQMPlatformBackend.from_machine(mw_fem_machine)
    circuit = Circuit(1, wire_names=["mw0"])
    circuit.add(M(0))

    with qua.program():
        macro = backend.circuit_to_qua_macro(circuit)
        macro()

    assert len(macro.measurement_map) == 1
    assert len(macro.acquisitions) == 1


# ---------------------------------------------------------------------------
# execute_circuit's default transpile step (issue #4)
#
# qibolab's own Compiler.compile never checks a gate against the platform's
# native gates and decomposes it if not -- it documents that the circuit it
# receives must already "respect the platform's connectivity and native
# gates". QibolabBackend.execute_circuit is patched out below (it needs a
# live platform.connect()/execute()) purely to capture the circuit this
# override hands it, proving the decomposition happened before delegating.
# ---------------------------------------------------------------------------


def test_execute_circuit_transpiles_a_non_native_gate_by_default(add_basic_macros_installed):
    from qibo import Circuit, gates

    backend = QiboQMPlatformBackend.from_machine(add_basic_macros_installed)
    assert "H" not in backend.natives  # H is not native on this fixture

    circuit = Circuit(1)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))

    with patch("qibolab._core.backends.QibolabBackend.execute_circuit") as mock_execute:
        with pytest.warns(UserWarning, match="H"):
            backend.execute_circuit(circuit)

    (received_circuit,), _ = mock_execute.call_args
    assert "H" not in [type(g).__name__ for g in received_circuit.queue]


def test_execute_circuit_transpile_false_leaves_a_non_native_gate_untouched(add_basic_macros_installed):
    """qibolab's own Compiler.compile raises its own error for an unknown
    gate; this override must not intercept that when transpile=False."""
    from qibo import Circuit, gates

    backend = QiboQMPlatformBackend.from_machine(add_basic_macros_installed)

    circuit = Circuit(1)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))

    with patch("qibolab._core.backends.QibolabBackend.execute_circuit") as mock_execute:
        backend.execute_circuit(circuit, transpile=False)

    (received_circuit,), _ = mock_execute.call_args
    assert [type(g).__name__ for g in received_circuit.queue] == ["H", "M"]


def test_execute_circuit_already_native_gate_is_unaffected_by_transpile(add_basic_macros_installed):
    import warnings

    from qibo import Circuit, gates

    backend = QiboQMPlatformBackend.from_machine(add_basic_macros_installed)
    assert "CZ" in backend.natives

    circuit = Circuit(2)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.M(0, 1))

    with patch("qibolab._core.backends.QibolabBackend.execute_circuit") as mock_execute:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            backend.execute_circuit(circuit)

    assert not caught
    (received_circuit,), _ = mock_execute.call_args
    assert [type(g).__name__ for g in received_circuit.queue] == ["CZ", "M"]


def test_execute_circuit_combines_initial_state_before_transpiling(add_basic_macros_installed):
    """A gate arriving only via initial_state must be transpiled too -- the
    override must combine before calling default_transpile, not after."""
    from qibo import Circuit, gates

    backend = QiboQMPlatformBackend.from_machine(add_basic_macros_installed)

    initial_state = Circuit(1)
    initial_state.add(gates.H(0))
    circuit = Circuit(1)
    circuit.add(gates.M(0))

    with patch("qibolab._core.backends.QibolabBackend.execute_circuit") as mock_execute:
        with pytest.warns(UserWarning, match="H"):
            backend.execute_circuit(circuit, initial_state=initial_state)

    (received_circuit,), received_kwargs = mock_execute.call_args
    assert "H" not in [type(g).__name__ for g in received_circuit.queue]
    assert received_kwargs["initial_state"] is None
