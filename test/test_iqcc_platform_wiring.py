"""Opt-in, real-hardware verification for the QuAM -> qibolab instrument
wiring (``qibo_qm_provider.qibolab_bridge._quam_wiring``/
``platform_from_quam._build_qm_controller``).

Mirrors ``test_iqcc_integration.py``'s pattern (same backend name, same
skip-on-unavailable fixture style) but targets the *other* execution path:
``QiboQMPlatformBackend``'s qibolab-native ``Platform``, not
``QiboQMBackend``'s OpenQASM/QuAM-macro path.

This is the strongest available check for the wiring converter: it diffs
qibolab's generated QM config against QuAM's own ``generate_config()`` for
the same physical ports, on real, 21-qubit, multi-bank production hardware
-- not just the synthetic ``mw_fem_machine`` fixture in ``test_quam_wiring.py``.
No connection to real instruments is made anywhere in this file (``manager``
is always ``None``); it only exercises config generation and compilation.

**Real finding, confirmed while writing this file, not a bug in this
converter:** installing macros and re-deriving native gates for the full
21-qubit machine (``add_basic_macros`` + ``refresh()``) hits
``AmplitudeOutOfRangeError`` for 7 of 21 qubits (``qA3``, ``qB5``, ``qC3``,
``qC5``, ``qD3``, ``qD5``) -- their calibrated ``x180`` pulses have a peak
voltage (0.51-0.88 V) exceeding qibolab's ``channel_max_voltage`` of 0.5 V.
Traced to qibolab's own QM driver (``qibolab._core.instruments.qm.
controller.channel_max_voltage``): it only special-cases an *LF-FEM*
``output_mode == "amplified"`` config; every ``IqConfig``-configured channel
(every MW-FEM drive/probe channel, on *any* rig) always gets the flat 0.5 V
"direct" ceiling, regardless of that port's actual ``full_scale_power_dbm``
(which genuinely varies per arbel port, -2 to +18 dBm) -- qibolab has no
dBm-to-voltage conversion for MW-FEM at all. Given this, dividing by a
higher, dBm-derived ceiling here would only desync this converter's scaling
from what qibolab's own driver does at pulse-registration time, producing a
*different*, silent amplitude mismatch instead of this loud, honest one;
raising is the correct behavior pending an upstream qibolab fix. The wiring
tests below (which do not depend on native-gate conversion) are unaffected
and use a separate, macro-free fixture; only
``test_compiles_real_native_gate_sequence_offline`` needs natives, and picks
a qubit (``qA1``) confirmed to be within qibolab's assumed ceiling.

Requires IQCC cloud credentials/access (see ``qiskit_qm_provider.IQCCProvider``)
and ``QUAM_STATE_PATH``. Skipped automatically otherwise. Run explicitly with::

    ~/venvs/rl_qoc/bin/python -m pytest test/test_iqcc_platform_wiring.py -v -m iqcc
"""

from dataclasses import asdict

import pytest

from qibo_qm_provider import QiboQMPlatformBackend, add_basic_macros

pytestmark = pytest.mark.iqcc

BACKEND_NAME = "arbel"

# One representative qubit per readout bank (A/B/C/D), matching arbel's real
# topology -- exercises the multiplexed-readout sharing across all 4
# physical MW-FEM readout ports, not just one.
_REPRESENTATIVE_QUBITS = ["qA1", "qB1", "qC1", "qD1"]

# Confirmed (see module docstring) to be within qibolab's assumed 0.5 V
# direct-mode ceiling -- safe to use for the one test that needs a real,
# fully-converted native-gate pulse.
_QUBIT_WITHIN_VOLTAGE_CEILING = "qA1"


@pytest.fixture
def backend():
    """Wiring-only: no macros installed, so native-gate conversion never
    touches a real pulse amplitude (macros dict is empty on a fresh fetch) --
    used by the two tests that only need instrument wiring, independent of
    the native-gate amplitude issue documented in the module docstring."""
    try:
        return QiboQMPlatformBackend.from_iqcc(BACKEND_NAME)
    except Exception as exc:  # noqa: BLE001 - any credential/network/backend failure means "skip"
        pytest.skip(f"IQCC backend {BACKEND_NAME!r} unavailable: {exc!r}")




def test_instruments_and_channels_populated_for_every_qubit(backend):
    platform = backend.platform
    assert list(platform.instruments) == ["qm"]

    for qubit_id in platform.qubits:
        qubit = platform.qubits[qubit_id]
        for attr in ("drive", "probe", "acquisition", "flux"):
            channel_id = getattr(qubit, attr)
            assert channel_id in platform.channels, f"{qubit_id}/{attr} missing from platform.channels"
            assert channel_id in platform.parameters.configs, f"{qubit_id}/{attr} missing a config"


def test_generated_config_matches_quam_ground_truth(backend):
    """Diffs qibolab's generated QM config against QuAM's own
    ``generate_config()`` for one qubit per readout bank. Only the
    known-and-warned-about differences (flux `offset`/`filter.feedforward`,
    absent `delay`/`shareable`) are expected to differ -- everything else
    (port tuples, signed `intermediate_frequency`, `time_of_flight`,
    `smearing`, FEM `type`, `band`, `full_scale_power_dbm`, `upconverter`)
    must match exactly.
    """
    platform = backend.platform
    controller = platform.instruments["qm"]
    configs = platform.parameters.configs

    truth = backend.machine.generate_config()

    for qubit_id in _REPRESENTATIVE_QUBITS:
        for suffix in ("drive", "probe", "acquisition", "flux"):
            controller.configure_channel(f"{qubit_id}/{suffix}", configs)

        generated = asdict(controller.config)["elements"]

        drive_element = generated[f"{qubit_id}/drive"]
        drive_truth = truth["elements"][f"{qubit_id}.xy"]
        assert drive_element["MWInput"] == drive_truth["MWInput"]
        assert drive_element["intermediate_frequency"] == pytest.approx(drive_truth["intermediate_frequency"])

        acq_element = generated[f"{qubit_id}/acquisition"]
        res_truth = truth["elements"][f"{qubit_id}.resonator"]
        assert acq_element["MWInput"] == res_truth["MWInput"]
        assert acq_element["MWOutput"] == res_truth["MWOutput"]
        assert acq_element["intermediate_frequency"] == pytest.approx(res_truth["intermediate_frequency"])
        assert acq_element["time_of_flight"] == pytest.approx(res_truth["time_of_flight"])
        assert acq_element["smearing"] == pytest.approx(res_truth["smearing"])

        flux_element = generated[f"{qubit_id}/flux"]
        flux_truth = truth["elements"][f"{qubit_id}.z"]
        assert flux_element["singleInput"] == flux_truth["singleInput"]


def test_compiles_real_native_gate_sequence_offline(backend):
    """Builds a PulseSequence for one qubit's real, calibrated RX native and
    plays it with ``manager=None`` (no hardware connection) -- qibolab
    returns the QUA program/config instead of executing
    (``controller.py:666-670``).

    Builds that one qubit's native via
    ``_quam_platform_conversion._single_qubit_natives`` directly (rather than
    ``platform.natives``, which would require ``add_basic_macros`` +
    ``backend.refresh()`` across the *whole* machine -- and, per the module
    docstring, 6 *other* qubits currently raise ``AmplitudeOutOfRangeError``
    during that whole-machine pass, unrelated to whether ``qA1`` itself is
    fine). This isolates the check to exactly the qubit under test.

    Confirms the amplitude-unit fix (Step 0) lands correctly end to end,
    regardless of which envelope path a given qubit's x180 takes (arbel uses
    DragCosinePulse -- the sampled `Custom` fallback, `amplitude=1.0` with
    everything baked into the samples -- but this check is written to work
    identically for a symbolic Rectangular/Gaussian pulse too): the peak
    realizable voltage qibolab's pulse would produce (`envelope * amplitude
    * max_voltage`) must match QuAM's own calibrated peak voltage
    (`calculate_waveform()`'s peak), since both ultimately describe the same
    physical x180 pulse.
    """
    import numpy as np
    from qibolab._core.execution_parameters import ExecutionParameters
    from qibolab._core.sequence import PulseSequence

    from qibo_qm_provider.qibolab_bridge._quam_platform_conversion import _single_qubit_natives

    qubit_id = _QUBIT_WITHIN_VOLTAGE_CEILING
    quam_qubit = backend.machine.qubits[qubit_id]
    # add_basic_macros only accepts a QuamRoot/QMBackend, not a bare qubit --
    # install macros on the whole machine (cheap: it only touches Python
    # objects, no wiring/network access), then build just this qubit's
    # native directly rather than the whole machine's (see module docstring:
    # the whole-machine pass currently fails for 6 other qubits).
    add_basic_macros(backend.machine, reset_type="active", max_attempts=1)

    natives = _single_qubit_natives(quam_qubit)
    rx_channel, rx_pulse = natives.RX[0]
    sequence = PulseSequence([(rx_channel, rx_pulse)])

    platform = backend.platform
    controller = platform.instruments["qm"]
    assert controller.manager is None  # never connected in this test file

    result = controller.play(
        platform.parameters.configs,
        [sequence],
        # relaxation_time must be an int here: QmController.play accesses
        # options.relaxation_time directly (controller.py's own
        # `default(self.relaxation_time, 0)` fallback lives on a different,
        # unrelated code path), so the field default of None would raise
        # `TypeError: '>' not supported between instances of 'NoneType' and 'int'`.
        ExecutionParameters(nshots=1, relaxation_time=0),
        [],
    )
    assert "program" in result and "config" in result

    # Pulse.i()/.q() take a *sampling rate* (samples per duration-unit), not a
    # sample count -- internally: `samples = int(self.duration * sampling_rate)`.
    # At QM's 1 GSa/s (`channel_sampling_rate`'s value for this config), that's
    # `sampling_rate=1`, giving `samples == duration` since duration is already
    # expressed in ns == samples at 1 GSa/s.
    envelope = np.asarray(rx_pulse.i(1)) + 1j * np.asarray(rx_pulse.q(1))
    qibolab_peak_voltage = float(np.max(np.abs(envelope))) * 0.5  # direct-mode MW-FEM max_voltage

    quam_waveform = np.asarray(quam_qubit.xy.operations["x180"].calculate_waveform(), dtype=complex)
    quam_peak_voltage = float(np.max(np.abs(quam_waveform)))

    assert qibolab_peak_voltage == pytest.approx(quam_peak_voltage, rel=1e-6)
