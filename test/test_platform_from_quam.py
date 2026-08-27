"""Tests for qibo_qm_provider.qibolab_bridge.platform_from_quam and
_quam_platform_conversion.

Uses the real ``dummy_machine``/``add_basic_macros_installed`` fixtures from
``conftest.py`` (a fully in-memory ``FluxTunableQuam``, no network/hardware
dependency) as the source, and asserts directly against qibolab's real
dataclass shapes -- no fake qibolab objects needed, mirroring
``test_qibolab_bridge.py``'s use of a real qibolab platform on the other
side of that conversion.
"""

import json

import pytest
from qibolab import Platform
from qibolab._core.pulses.envelope import Gaussian, Rectangular
from quam.components.pulses import DragCosinePulse, GaussianPulse, SquarePulse

from qibo_qm_provider.exceptions import AmplitudeOutOfRangeError, UnsupportedEnvelopeError
from qibo_qm_provider.qibolab_bridge._quam_platform_conversion import (
    _build_couplers,
    _build_native_gates,
    _build_qubits,
    _quam_envelope_to_qibolab_pulse,
)
from qibo_qm_provider.qibolab_bridge.platform_from_quam import create_iqcc, create_local, quam_to_qibolab_platform

# ---------------------------------------------------------------------------
# _quam_envelope_to_qibolab_pulse
# ---------------------------------------------------------------------------
#
# Amplitudes below are QuAM volts divided by the default 0.5 V "direct"
# max_voltage (no channel is passed, so _quam_envelope_to_qibolab_pulse falls
# back to that default) -- e.g. 0.1 V -> 0.2, matching qibolab's own
# channel_max_voltage re-multiplication at playback time.


def test_square_pulse_converts_to_rectangular():
    pulse = _quam_envelope_to_qibolab_pulse(SquarePulse(length=40, amplitude=0.1))
    assert pulse.duration == 40
    assert pulse.amplitude == pytest.approx(0.2)
    assert isinstance(pulse.envelope, Rectangular)


def test_gaussian_pulse_converts_losslessly():
    pulse = _quam_envelope_to_qibolab_pulse(GaussianPulse(length=100, amplitude=0.2, sigma=25))
    assert pulse.duration == 100
    assert pulse.amplitude == pytest.approx(0.4)
    assert isinstance(pulse.envelope, Gaussian)
    assert pulse.envelope.rel_sigma == pytest.approx(0.25)


def test_square_pulse_amplitude_scales_with_max_voltage():
    """An amplified channel (2.5 V max) needs a smaller dimensionless
    amplitude than a direct one (0.5 V max) to reach the same real voltage."""
    pulse = _quam_envelope_to_qibolab_pulse(SquarePulse(length=40, amplitude=0.1), max_voltage=2.5)
    assert pulse.amplitude == pytest.approx(0.04)


def test_amplitude_exceeding_max_voltage_raises():
    with pytest.raises(AmplitudeOutOfRangeError):
        _quam_envelope_to_qibolab_pulse(SquarePulse(length=40, amplitude=0.6), max_voltage=0.5)


def test_drag_cosine_pulse_falls_back_to_custom_envelope():
    """DragCosinePulse has no sigma-equivalent field, unlike qibolab's Drag
    envelope which requires rel_sigma -- rather than raising, this falls
    back to QuAM's own calculate_waveform() samples via qibolab's Custom
    envelope, so execution correctness doesn't depend on qibolab having a
    symbolic envelope class matching every QuAM pulse shape."""
    drag = DragCosinePulse(length=40, amplitude=0.1, alpha=0.1, anharmonicity=-2e8, axis_angle=0.0)
    pulse = _quam_envelope_to_qibolab_pulse(drag)

    assert pulse.duration == 40
    assert pulse.amplitude == 1.0  # samples are already amplitude-scaled, not double-scaled
    assert len(pulse.envelope.i_) == 40
    assert len(pulse.envelope.q_) == 40


def test_unrecognized_pulse_type_is_unsupported():
    with pytest.raises(UnsupportedEnvelopeError):
        _quam_envelope_to_qibolab_pulse(object())


# ---------------------------------------------------------------------------
# _build_qubits / _build_couplers
# ---------------------------------------------------------------------------


def test_build_qubits_uses_quam_names_verbatim(dummy_machine):
    qubits = _build_qubits(dummy_machine)
    assert set(qubits) == {"q0", "q1"}
    assert qubits["q0"].drive == "q0/drive"
    assert qubits["q0"].probe == "q0/probe"
    assert qubits["q0"].acquisition == "q0/acquisition"
    assert qubits["q0"].flux == "q0/flux"


def test_build_couplers_omits_coupler_less_pairs(dummy_machine):
    """dummy_machine's q0-q1 pair has coupler=None -- a fully valid real
    case (e.g. CZ activated by flux-tuning alone) -- so it contributes no
    coupler entry, independent of whether it has a working CZ native."""
    couplers = _build_couplers(dummy_machine)
    assert couplers == {}


# ---------------------------------------------------------------------------
# _build_native_gates / quam_to_qibolab_platform
# ---------------------------------------------------------------------------


def test_native_gates_match_quam_pulse_fields(add_basic_macros_installed):
    """dummy_machine's ports are bare ("con1", n) tuples with no output_mode,
    so every channel falls back to the 0.5 V "direct" max_voltage -- QuAM
    volts are therefore doubled here (0.1 V -> 0.2), matching qibolab's
    channel_max_voltage re-multiplication at playback time."""
    natives = _build_native_gates(add_basic_macros_installed)

    rx_channel, rx_pulse = natives.single_qubit["q0"].RX[0]
    assert rx_channel == "q0/drive"
    assert rx_pulse.duration == 40
    assert rx_pulse.amplitude == pytest.approx(0.2)

    rx90_channel, rx90_pulse = natives.single_qubit["q0"].RX90[0]
    assert rx90_pulse.amplitude == pytest.approx(0.1)

    mz_channel, mz_readout = natives.single_qubit["q0"].MZ[0]
    assert mz_channel == "q0/acquisition"
    assert mz_readout.probe.duration == 100
    assert mz_readout.probe.amplitude == pytest.approx(0.2)


def test_rz_macro_contributes_no_native(add_basic_macros_installed):
    """VirtualZMacro (rz) has no .pulse reference -- qibolab's
    SingleQubitNatives has no RZ field either (virtual-Z is a compiler-level
    frame operation), so this is expected, not a gap."""
    natives = _build_native_gates(add_basic_macros_installed)
    assert not hasattr(natives.single_qubit["q0"], "RZ")


def test_cz_native_uses_moving_qubit_flux_channel(add_basic_macros_installed):
    """The installed CZGate's flux_pulse_qubit resolves directly to a real
    QuAM Pulse object (verified empirically, not a string needing a
    secondary channel/.operations lookup) and plays on whichever qubit
    pair.moving_qubit selects -- "control" (q0) by default here."""
    natives = _build_native_gates(add_basic_macros_installed)
    pair = add_basic_macros_installed.qubit_pairs["q0-q1"]
    cz_macro = pair.macros.get("cz")
    if cz_macro is None:
        pytest.skip("cz macro not installed on this environment (see add_basic_macros_installed docstring)")

    cz_natives = natives.two_qubit[("q0", "q1")]
    channel, pulse = cz_natives.CZ[0]
    assert channel == "q0/flux"
    assert pulse.duration == 40
    assert pulse.amplitude == pytest.approx(0.2)  # 0.1 V / 0.5 V default direct max_voltage


def test_quam_to_qibolab_platform_builds_full_topology(add_basic_macros_installed):
    """dummy_machine has no `network` and uses Octave-less IQ channels
    (XYDriveIQ/ReadoutResonatorIQ), neither of which _quam_wiring can build
    instruments from -- quam_to_qibolab_platform warns and falls back to an
    instruments-less Platform rather than raising (see
    test_quam_wiring.py for the machines/assertions that exercise the
    working-instruments path, using the mw_fem_machine fixture)."""
    with pytest.warns(UserWarning, match="Could not build QmController"):
        platform = quam_to_qibolab_platform(add_basic_macros_installed, name="qibo-qm-local")

    assert isinstance(platform, Platform)
    assert platform.name == "qibo-qm-local"
    assert platform.nqubits == 2
    assert set(platform.qubits) == {"q0", "q1"}
    assert platform.pairs == [("q0", "q1")]
    assert platform.instruments == {}
    assert platform.settings.relaxation_time == add_basic_macros_installed.thermalization_time


# ---------------------------------------------------------------------------
# create_local
# ---------------------------------------------------------------------------


def test_create_local_round_trips_from_disk(dummy_machine, tmp_path):
    """Uses to_dict() + a manual JSON write rather than machine.save(),
    which depends on a global qualibrate config context not set up in this
    test environment -- to_dict()/load() round-trips without it."""
    (tmp_path / "state.json").write_text(json.dumps(dummy_machine.to_dict()))

    platform = create_local(state_path=str(tmp_path))

    assert set(platform.qubits) == {"q0", "q1"}
    assert platform.name == "qibo-qm-local"


def test_create_local_requires_state_path():
    with pytest.raises(ValueError, match="state_path"):
        create_local()


def test_create_local_resolves_relative_state_path_against_folder(dummy_machine, tmp_path):
    (tmp_path / "quam_state").mkdir()
    (tmp_path / "quam_state" / "state.json").write_text(json.dumps(dummy_machine.to_dict()))
    (tmp_path / "quam_source.json").write_text(json.dumps({"state_path": "quam_state"}))

    platform = create_local(folder=tmp_path)

    assert set(platform.qubits) == {"q0", "q1"}


# ---------------------------------------------------------------------------
# create_iqcc (mocked -- no existing IQCC-mocking infra in this repo, see
# qibolab_platform_from_quam_plan.md)
# ---------------------------------------------------------------------------


def test_create_iqcc_happy_path(dummy_machine, monkeypatch, tmp_path):
    def fake_get_machine_from_iqcc(backend_name, api_token=None, quam_state_folder_path=None, quam_cls=None):
        assert backend_name == "arbel"
        return dummy_machine, None

    monkeypatch.setattr(
        "qiskit_qm_provider.providers.iqcc_cloud_provider.get_machine_from_iqcc",
        fake_get_machine_from_iqcc,
    )

    platform = create_iqcc("qibo-qm-iqcc-arbel", state_path=str(tmp_path))

    assert set(platform.qubits) == {"q0", "q1"}
    assert platform.name == "qibo-qm-iqcc-arbel"


def test_create_iqcc_wraps_value_error(monkeypatch, tmp_path):
    def fake_get_machine_from_iqcc(*args, **kwargs):
        raise ValueError("Error code 404 Not Found")

    monkeypatch.setattr(
        "qiskit_qm_provider.providers.iqcc_cloud_provider.get_machine_from_iqcc",
        fake_get_machine_from_iqcc,
    )

    with pytest.raises(ValueError, match="arbel"):
        create_iqcc("qibo-qm-iqcc-arbel", state_path=str(tmp_path))


def test_create_iqcc_wraps_connection_error(monkeypatch, tmp_path):
    def fake_get_machine_from_iqcc(*args, **kwargs):
        raise ConnectionError("Error code 502 Bad Gateway")

    monkeypatch.setattr(
        "qiskit_qm_provider.providers.iqcc_cloud_provider.get_machine_from_iqcc",
        fake_get_machine_from_iqcc,
    )

    with pytest.raises(ValueError, match="arbel"):
        create_iqcc("qibo-qm-iqcc-arbel", state_path=str(tmp_path))
