"""Tests for qibo_qm_provider.qibolab_bridge.pulse_sequence_import.

Uses qibolab's own ``create_platform("dummy")`` (a real, calibrated 5-qubit
platform shipped by qibolab for testing) as a realistic source of natives,
against lightweight fake QuAM-shaped objects (rather than the full
``dummy_machine`` fixture) since the importer only needs `.xy`/`.resonator`/
`.z`/`.coupler`/`.macros`/`.operations`-shaped attributes, not a fully wired
QMBackend-compatible machine.
"""

import warnings

import pytest
from qibolab import create_platform

from qibo_qm_provider.qibolab_bridge import import_qibolab_natives_as_macros


class _FakeChannel:
    def __init__(self):
        self.operations = {}


class _FakeQubit:
    def __init__(self):
        self.xy = _FakeChannel()
        self.resonator = _FakeChannel()
        self.z = _FakeChannel()
        self.macros = {}


class _FakePair:
    def __init__(self):
        self.qubit_control = None
        self.qubit_target = None
        self.coupler = _FakeChannel()
        self.macros = {}


class _FakeMachine:
    def __init__(self, qubit_ids, pair_ids):
        self.qubits = {str(q): _FakeQubit() for q in qubit_ids}
        self.qubit_pairs = {f"{c}-{t}": _FakePair() for c, t in pair_ids}


@pytest.fixture
def platform():
    return create_platform("dummy")


@pytest.fixture
def fake_machine(platform):
    return _FakeMachine(platform.qubits, platform.pairs)


def test_single_qubit_natives_installed_as_macros(platform, fake_machine):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # expected VirtualZ-skip warnings from CZ natives
        installed = import_qibolab_natives_as_macros(platform, fake_machine, qubits=[0, 1])

    assert "0:x" in installed
    assert "0:sx" in installed
    assert "0:measure" in installed
    q0 = fake_machine.qubits["0"]
    assert "x" in q0.macros
    assert "sx" in q0.macros
    assert "measure" in q0.macros
    assert q0.xy.operations  # at least one pulse registered


def test_two_qubit_cz_native_installed_on_coupler(platform, fake_machine):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        installed = import_qibolab_natives_as_macros(platform, fake_machine, qubits=[0, 1, 2])

    assert "0-2:cz" in installed
    pair = fake_machine.qubit_pairs["0-2"]
    assert "cz" in pair.macros
    assert pair.coupler.operations


def test_virtual_z_legs_are_skipped_with_warning(platform, fake_machine):
    """CZ natives mix Pulse and VirtualZ instructions; VirtualZ has no
    single-channel QuAM Pulse equivalent and must be skipped, not crash."""
    with pytest.warns(UserWarning, match="VirtualZ"):
        import_qibolab_natives_as_macros(platform, fake_machine, qubits=[0, 1, 2])


def test_qubits_filter_restricts_import(platform, fake_machine):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        installed = import_qibolab_natives_as_macros(platform, fake_machine, qubits=[0])

    assert all(key.startswith("0:") for key in installed)
    assert fake_machine.qubits["1"].macros == {}


def test_missing_quam_qubit_is_skipped_not_raised(platform):
    machine = _FakeMachine(qubit_ids=[0], pair_ids=[])  # only "0" exists in machine.qubits
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        installed = import_qibolab_natives_as_macros(platform, machine, qubits=[0, 1])

    assert all(key.startswith("0:") for key in installed)
