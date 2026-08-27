"""Tests for qibo_qm_provider.qibolab_bridge._quam_wiring and the
instrument-wiring half of platform_from_quam.quam_to_qibolab_platform.

Uses the ``mw_fem_machine`` fixture (real OPX1000 MW-FEM/LF-FEM port
objects) for the MW-FEM/LF-FEM wiring itself, and ``dummy_machine``/
``add_basic_macros_installed`` (Octave-less IQ, bare tuple ports) to prove
the graceful fallback for machines this converter cannot wire.
"""

import warnings

import pytest
from qibolab._core.components import AcquisitionChannel, DcChannel, IqChannel
from qibolab._core.instruments.qm import QmController
from qibolab._core.instruments.qm.components import MwFemOscillatorConfig, OpxOutputConfig, QmAcquisitionConfig

from qibo_qm_provider.exceptions import MissingQuamAttributeError, UnsupportedWiringError
from qibo_qm_provider.qibolab_bridge._quam_platform_conversion import _build_couplers, _build_qubits
from qibo_qm_provider.qibolab_bridge._quam_wiring import build_qm_wiring
from qibo_qm_provider.qibolab_bridge.platform_from_quam import DEFAULT_QM_PORT, _build_qm_controller, quam_to_qibolab_platform

# ---------------------------------------------------------------------------
# build_qm_wiring -- channel/config shape
# ---------------------------------------------------------------------------


def test_channel_ids_match_build_qubits(mw_fem_machine):
    """Platform.channels (derived from these instruments) must line up with
    Platform.qubits (derived independently by _build_qubits) or a Qibocal
    sweeper addressing platform.qubits[q].probe would find no such channel."""
    channels, _, _ = build_qm_wiring(mw_fem_machine)
    qubits = _build_qubits(mw_fem_machine)

    expected = {getattr(q, attr) for q in qubits.values() for attr in ("drive", "probe", "acquisition", "flux")}
    assert set(channels) == expected


def test_channel_types(mw_fem_machine):
    channels, _, _ = build_qm_wiring(mw_fem_machine)
    assert isinstance(channels["mw0/drive"], IqChannel)
    assert isinstance(channels["mw0/probe"], IqChannel)
    assert isinstance(channels["mw0/acquisition"], AcquisitionChannel)
    assert isinstance(channels["mw0/flux"], DcChannel)
    assert channels["mw0/acquisition"].probe == "mw0/probe"


def test_fems_fully_populated_as_mw_and_lf(mw_fem_machine):
    """fems must be explicit for every device -- QmController.fems defaults
    to a defaultdict returning "opx1" for any missing key, which would
    silently mistype an MW-FEM channel while its element is still built
    MW-FEM-style (a real, confirmed footgun in qibolab's own driver)."""
    _, _, fems = build_qm_wiring(mw_fem_machine)
    assert fems == {"con1/1": "MW", "con1/5": "LF"}


def test_iq_config_frequency_is_absolute_rf(mw_fem_machine):
    """qibolab's IqConfig.frequency is the absolute RF frequency -- it
    derives intermediate_frequency itself as `config.frequency -
    lo_config.frequency` (qibolab._core.instruments.qm.config.config).
    Passing QuAM's signed IF straight through would be wrong by twice the LO."""
    channels, configs, _ = build_qm_wiring(mw_fem_machine)
    xy = mw_fem_machine.qubits["mw0"].xy

    drive_config = configs["mw0/drive"]
    lo_config = configs[channels["mw0/drive"].lo]
    assert drive_config.frequency == pytest.approx(xy.upconverter_frequency + xy.intermediate_frequency)
    assert drive_config.frequency - lo_config.frequency == pytest.approx(xy.intermediate_frequency)


def test_multiplexed_readout_shares_one_lo_config(mw_fem_machine):
    """mw0 and mw1's resonators share one physical MW-FEM port -- their
    probe channels must reference the *same* LO config id, and configuring
    both must not trip qibolab's MwFemOutput.update() consistency asserts
    (band/sampling_rate/power must match, which they do since it's the same
    physical port)."""
    channels, configs, fems = build_qm_wiring(mw_fem_machine)

    assert channels["mw0/probe"].lo == channels["mw1/probe"].lo
    lo_id = channels["mw0/probe"].lo
    assert isinstance(configs[lo_id], MwFemOscillatorConfig)

    controller = QmController(address="1.2.3.4:9510", channels=channels, fems=fems)
    # Exercises qibolab's own MwFemOutput.update() assertions internally --
    # would raise AssertionError if the two qubits' shared-port configs
    # disagreed on band/sampling_rate/power.
    controller.configure_channel("mw0/acquisition", configs)
    controller.configure_channel("mw1/acquisition", configs)


def test_acquisition_config_uses_readout_pulse_fields(mw_fem_machine):
    _, configs, _ = build_qm_wiring(mw_fem_machine)
    acq_config = configs["mw0/acquisition"]
    ro = mw_fem_machine.qubits["mw0"].resonator.operations["readout"]

    assert isinstance(acq_config, QmAcquisitionConfig)
    assert acq_config.delay == 372
    assert acq_config.threshold == pytest.approx(ro.threshold)
    assert acq_config.iq_angle == pytest.approx(ro.integration_weights_angle)
    assert acq_config.kernel is None


# ---------------------------------------------------------------------------
# Flux wiring -- operating point and filter loss
# ---------------------------------------------------------------------------


def test_flux_offset_comes_from_joint_offset_not_port(mw_fem_machine):
    """The static config offset must come from FluxLine.joint_offset (the
    calibrated operating point QuAM biases to at runtime), not from the
    port's own offset field (which QuAM always emits as 0.0, since it biases
    in QUA instead) -- a qibolab Platform has no runtime prologue to do that
    biasing, so the operating point must be static or every Qibocal protocol
    would run at the wrong flux bias."""
    _, configs, _ = build_qm_wiring(mw_fem_machine)
    assert configs["mw0/flux"].offset == pytest.approx(0.05)
    assert isinstance(configs["mw0/flux"], OpxOutputConfig)


def test_feedforward_and_exponential_together_warns_only_on_affected_qubit(mw_fem_machine):
    """mw0's flux port has both exponential_filter and feedforward_filter
    (the combination that risks double-applying the exponential correction,
    see _quam_wiring's module docstring); mw1's has neither -- the warning
    must fire once, naming mw0, and not at all for mw1."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_qm_wiring(mw_fem_machine)

    double_apply_warnings = [w for w in caught if "double-apply" in str(w.message)]
    assert len(double_apply_warnings) == 1
    assert "'mw0'" in str(double_apply_warnings[0].message)
    assert not any("'mw1'" in str(w.message) and "double-apply" in str(w.message) for w in caught)


def test_delay_and_crosstalk_loss_warns(mw_fem_machine):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_qm_wiring(mw_fem_machine)

    assert any("delay=61" in str(w.message) for w in caught)


def test_feedforward_taps_are_transferred(mw_fem_machine):
    """QuAM's raw feedforward_filter taps are now transferred as a
    FiniteImpulseResponseFilter (previously silently dropped) -- the
    resulting config's "feedforward" key is the normalized *convolution* of
    those taps with the exponential filter's own FIR approximation (per
    OpxOutputConfig.filter()'s cluster-agnostic feedforward property, which
    combines every registered filter), so it is not bit-identical to the raw
    taps, but the taps are a genuine, present contribution to it -- unlike
    before, where they had no representation in the config at all."""
    from qibolab._core.components.filters import FiniteImpulseResponseFilter

    _, configs, _ = build_qm_wiring(mw_fem_machine)
    quam_port = mw_fem_machine.qubits["mw0"].z.opx_output

    fir_filters = [f for f in configs["mw0/flux"].filters if isinstance(f, FiniteImpulseResponseFilter)]
    assert len(fir_filters) == 1
    assert fir_filters[0].coefficients == list(quam_port.feedforward_filter)

    derived = configs["mw0/flux"].filter("LF")["feedforward"]
    assert derived != list(quam_port.feedforward_filter)  # convolved with the exponential terms + normalized


# ---------------------------------------------------------------------------
# Unsupported wiring
# ---------------------------------------------------------------------------


def test_iq_channel_machine_raises_unsupported_wiring(add_basic_macros_installed):
    """dummy_machine's XYDriveIQ/ReadoutResonatorIQ channels have no MW-FEM
    equivalent supported by this converter."""
    with pytest.raises(UnsupportedWiringError):
        build_qm_wiring(add_basic_macros_installed)


def test_quam_to_qibolab_platform_falls_back_without_raising(add_basic_macros_installed):
    """quam_to_qibolab_platform must keep working for machines it cannot
    wire (this package's own previous behavior) -- warn and fall back to an
    instruments-less Platform rather than raising where it used to succeed."""
    with pytest.warns(UserWarning, match="Could not build QmController"):
        platform = quam_to_qibolab_platform(add_basic_macros_installed, name="qibo-qm-local")

    assert platform.instruments == {}
    assert platform.parameters.configs == {}
    assert set(platform.qubits) == {"q0", "q1"}  # topology/native-gates unaffected


def test_missing_network_host_falls_back_with_warning(mw_fem_machine):
    """A machine with fully-supported MW-FEM/LF-FEM wiring but no
    machine.network["host"] (e.g. never connected to a real cluster) must
    fall back the same way unsupported wiring does, not raise -- both are
    "instruments cannot be built" cases from the caller's perspective."""
    mw_fem_machine.network = {}
    with pytest.warns(UserWarning, match="Could not build QmController"):
        platform = quam_to_qibolab_platform(mw_fem_machine, name="qibo-qm-local")
    assert platform.instruments == {}


# ---------------------------------------------------------------------------
# _build_qm_controller -- address/cluster assembly
# ---------------------------------------------------------------------------


def test_build_qm_controller_uses_network_host_and_cluster(mw_fem_machine):
    controller, configs = _build_qm_controller(mw_fem_machine)
    assert controller.address == f"1.2.3.4:{DEFAULT_QM_PORT}"
    assert controller.cluster_name == "test-cluster"
    assert set(controller.channels) == {"mw0/drive", "mw0/probe", "mw0/acquisition", "mw0/flux",
                                         "mw1/drive", "mw1/probe", "mw1/acquisition", "mw1/flux"}
    assert configs  # non-empty


def test_build_qm_controller_synthesizes_missing_port_with_warning(mw_fem_machine):
    """mw_fem_machine's network deliberately carries no "port" entry (matches
    real arbel state) -- QmController.connect() does
    self.address.split(":") unconditionally, so a port must always be
    synthesized, with a warning rather than a silent guess."""
    with pytest.warns(UserWarning, match="no 'port' entry"):
        controller, _ = _build_qm_controller(mw_fem_machine)
    assert controller.address.endswith(f":{DEFAULT_QM_PORT}")


def test_build_qm_controller_respects_explicit_port_override(mw_fem_machine):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        controller, _ = _build_qm_controller(mw_fem_machine, port=12345)
    assert controller.address == "1.2.3.4:12345"


def test_build_qm_controller_requires_network_host():
    from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import FluxTunableQuam

    machine = FluxTunableQuam()
    with pytest.raises(MissingQuamAttributeError, match="host"):
        _build_qm_controller(machine)


# ---------------------------------------------------------------------------
# End-to-end: quam_to_qibolab_platform with working instruments
# ---------------------------------------------------------------------------


def test_quam_to_qibolab_platform_builds_working_instruments(mw_fem_machine):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # expected filter-loss/port warnings
        platform = quam_to_qibolab_platform(mw_fem_machine, name="qibo-qm-local")

    assert list(platform.instruments) == ["qm"]
    qubits = _build_qubits(mw_fem_machine)
    expected_channels = {getattr(q, attr) for q in qubits.values() for attr in ("drive", "probe", "acquisition", "flux")}
    assert set(platform.channels) == expected_channels

    # every channel has a matching config -- required for QmController.play,
    # which re-registers every DC channel on every execution regardless of
    # whether it appears in the sequence (controller.py: "register DC
    # elements so that all qubits are sweetspot even when not used").
    for channel_id in platform.channels:
        assert channel_id in platform.parameters.configs
    assert _build_couplers(mw_fem_machine) == platform.couplers == {}
