"""Tests for qibo_qm_provider.backend.default_transpile.default_transpile.

Regression coverage for issue #4: a plain ``H`` (bug #3's root cause) must be
decomposed into a target's native gates by default, while a gate this
package treats as a first-class machine native (``GPI``, ``PRX``, ``MS``,
none of which are members of qibo.transpiler.unroller.NativeGates) must
never be handed to Qibo's generic decomposer, which raises a bare
``KeyError``/``DecompositionError`` for them.
"""

import warnings

import pytest
from qibo import Circuit, gates

from qibo_qm_provider.backend.default_transpile import default_transpile
from qibo_qm_provider.exceptions import UnsupportedGateError

# A target vocabulary rich enough to decompose any standard gate.
FULL_NATIVE_GATES = ["I", "Z", "RZ", "GPI2", "U3", "CZ", "iSWAP", "CNOT", "M"]


def test_already_native_gate_is_left_untouched_and_does_not_warn():
    circuit = Circuit(1)
    circuit.add(gates.GPI(0, phi=0.3))
    circuit.add(gates.M(0))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = default_transpile(circuit, already_native={"GPI"}, decomposition_targets=FULL_NATIVE_GATES)

    assert not caught
    assert [type(g).__name__ for g in out.queue] == ["GPI", "M"]


def test_non_native_gate_is_decomposed_and_warns():
    """Bug #3's root cause: a plain H has no macro on many machines and must
    be decomposed by default."""
    circuit = Circuit(1)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))

    with pytest.warns(UserWarning, match="H"):
        out = default_transpile(circuit, already_native=set(), decomposition_targets=FULL_NATIVE_GATES)

    names = [type(g).__name__ for g in out.queue]
    assert "H" not in names
    assert names[-1] == "M"
    assert set(names) <= set(FULL_NATIVE_GATES)


def test_warn_on_change_false_suppresses_the_warning():
    circuit = Circuit(1)
    circuit.add(gates.H(0))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        default_transpile(
            circuit, already_native=set(), decomposition_targets=FULL_NATIVE_GATES, warn_on_change=False
        )
    assert not caught


def test_measurement_gate_is_never_decomposed():
    circuit = Circuit(2)
    circuit.add(gates.M(0, 1))

    out = default_transpile(circuit, already_native=set(), decomposition_targets=[])

    assert [type(g).__name__ for g in out.queue] == ["M"]


def test_deferred_gate_passes_through_for_its_own_specific_error_downstream():
    """Align/GIVENS/RBS/... have no representation on any lowering path in
    this package -- the specific, actionable error for them belongs to
    build_qiskit_circuit_directly (gate_map.DEFERRED_GATES), not a generic
    transpile failure. This step must not intercept them."""
    circuit = Circuit(2)
    circuit.add(gates.Align(0, delay=10))
    circuit.add(gates.M(0, 1))

    out = default_transpile(circuit, already_native=set(), decomposition_targets=FULL_NATIVE_GATES)

    assert [type(g).__name__ for g in out.queue] == ["Align", "M"]


def test_wire_names_are_preserved():
    circuit = Circuit(2, wire_names=["q3", "q1"])
    circuit.add(gates.H(0))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.M(0, 1))

    out = default_transpile(circuit, already_native={"CZ"}, decomposition_targets=FULL_NATIVE_GATES)

    assert out.wire_names == ["q3", "q1"]


def test_no_decomposition_rule_raises_actionable_error_not_a_bare_keyerror():
    """PRX/U1q/MS/GPI are not qibo.transpiler.unroller.NativeGates members,
    so Qibo's own decomposition tables have no entry for them at all
    (verified: a bare KeyError). Not already native and not decomposable
    must raise UnsupportedGateError naming the gate, not the raw KeyError."""
    circuit = Circuit(1)
    circuit.add(gates.PRX(0, 0.1, 0.2))

    with pytest.raises(UnsupportedGateError, match="PRX"):
        default_transpile(circuit, already_native=set(), decomposition_targets=FULL_NATIVE_GATES)


def test_missing_single_qubit_native_family_raises_actionable_error():
    """qibo's decomposer refuses outright (DecompositionError, not returned
    but raised) when neither U3 nor GPI2 is offered as a single-qubit
    target -- must still surface as UnsupportedGateError, not propagate."""
    circuit = Circuit(1)
    circuit.add(gates.H(0))

    with pytest.raises(UnsupportedGateError, match="H"):
        default_transpile(circuit, already_native=set(), decomposition_targets=["I", "RZ", "CZ", "M"])


def test_no_native_gates_at_all_raises_actionable_error():
    circuit = Circuit(1)
    circuit.add(gates.H(0))

    with pytest.raises(UnsupportedGateError, match="H"):
        default_transpile(circuit, already_native=set(), decomposition_targets=[])


def test_decomposition_targets_outside_nativegates_enum_are_ignored():
    """QibolabBackend.natives can contain names with no NativeGates member at
    all (e.g. "GPI", "Align") -- passing them as decomposition_targets must
    not raise (NativeGates[...] would KeyError on an unknown member); they
    are simply not usable as decomposition output."""
    circuit = Circuit(1)
    circuit.add(gates.I(0))
    circuit.add(gates.M(0))

    out = default_transpile(
        circuit, already_native={"I", "M"}, decomposition_targets=["GPI", "Align", "I", "M"]
    )

    assert [type(g).__name__ for g in out.queue] == ["I", "M"]
