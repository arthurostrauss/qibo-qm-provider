"""Tests for qibo_qm_provider.qibolab_bridge.native_import.

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


def test_readout_macro_reads_threshold_from_acquisition_config(fake_machine):
    """The qibolab Readout instruction carries no threshold/angle of its own
    -- that calibration lives on the platform's acquisition-channel config
    (``Platform.parameters.configs``) and must be plumbed through explicitly
    (see native_import.py's ``acquisition_configs`` parameter). Exercises
    ``_install_native_as_macro`` directly with a hand-built rectangular-probe
    Readout, since the "dummy" platform's own MZ probe envelope is Gaussian
    and would fall back to a plain WaveformPulse instead of a
    threshold-carrying SquareReadoutPulse."""
    from types import SimpleNamespace

    from qibolab._core.native import Native
    from qibolab._core.pulses.envelope import Rectangular
    from qibolab._core.pulses.pulse import Acquisition, Pulse, Readout

    from qibo_qm_provider.qibolab_bridge.native_import import _install_native_as_macro

    readout = Readout(
        acquisition=Acquisition(duration=1000), probe=Pulse(duration=1000, amplitude=0.1, envelope=Rectangular())
    )
    native = Native([("0/acquisition", readout)])
    acq_config = SimpleNamespace(threshold=0.42, iq_angle=1.23)

    q0 = fake_machine.qubits["0"]
    _install_native_as_macro(native, "measure", q0, "test_0", {"0/acquisition": acq_config})

    readout_pulse = q0.resonator.operations["test_0_measure"]
    assert readout_pulse.threshold == pytest.approx(0.42)
    assert readout_pulse.integration_weights_angle == pytest.approx(1.23)


def test_missing_quam_qubit_is_skipped_not_raised(platform):
    machine = _FakeMachine(qubit_ids=[0], pair_ids=[])  # only "0" exists in machine.qubits
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        installed = import_qibolab_natives_as_macros(platform, machine, qubits=[0, 1])

    assert all(key.startswith("0:") for key in installed)
