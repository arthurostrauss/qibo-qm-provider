"""Tests for qibo_qm_provider.qibolab_bridge._quam_pulses -- the shared,
bidirectional QuAM <-> qibolab pulse-envelope conversion.

QuAM -> qibolab tests for Rectangular/Gaussian/generic-fallback already live
in test_platform_from_quam.py (via the _quam_envelope_to_qibolab_pulse
backward-compat alias re-exported from _quam_platform_conversion). This
module focuses on the qibolab -> QuAM direction (new coverage: all 7
qibolab envelope kinds, previously 1 of 7 -- rectangular only, since gaussian
had no forward-direction converter at all before this module existed) and
round-trip amplitude preservation across both directions.
"""

import numpy as np
import pytest
from qibolab._core.pulses.envelope import Custom, Drag, Exponential, Gaussian, GaussianSquare, Rectangular, Snz
from qibolab._core.pulses.pulse import Acquisition, Pulse, Readout
from quam.components.pulses import GaussianPulse, SquarePulse, SquareReadoutPulse, WaveformPulse

from qibo_qm_provider.exceptions import AmplitudeOutOfRangeError
from qibo_qm_provider.qibolab_bridge._quam_pulses import (
    quam_envelope_to_qibolab_pulse,
    quam_pulse_from_qibolab_pulse,
    quam_readout_pulse_from_qibolab_readout,
)

_ENVELOPES = [
    Rectangular(),
    Gaussian(rel_sigma=0.2),
    Drag(rel_sigma=0.2, beta=0.5),
    GaussianSquare(risefall=5, sigma=3.0),
    Snz(t_idling=4, b_amplitude=0.5),
    Exponential(tau=0.1, upsilon=0.2),
    Custom(i_=np.linspace(0, 1, 40), q_=np.zeros(40)),
]


@pytest.mark.parametrize("envelope", _ENVELOPES, ids=lambda e: e.kind)
def test_qibolab_to_quam_round_trips_amplitude(envelope):
    """Every one of qibolab's 7 envelope kinds converts to a QuAM Pulse and
    back with its real, played waveform preserved -- not just its nominal
    `amplitude` field, since the sampled-fallback kinds bake amplitude into
    the samples themselves rather than carrying a separate field."""
    pulse = Pulse(duration=40, amplitude=0.3, envelope=envelope)
    quam_pulse = quam_pulse_from_qibolab_pulse(pulse, "test", max_voltage=0.5)
    back = quam_envelope_to_qibolab_pulse(quam_pulse, max_voltage=0.5)

    orig_waveform = pulse.i(1.0) + 1j * pulse.q(1.0)
    back_waveform = back.i(1.0) + 1j * back.q(1.0)
    assert np.allclose(orig_waveform, back_waveform, atol=1e-6)


def test_rectangular_converts_to_symbolic_square_pulse():
    pulse = Pulse(duration=40, amplitude=0.3, envelope=Rectangular())
    quam_pulse = quam_pulse_from_qibolab_pulse(pulse, "test", max_voltage=0.5)
    assert isinstance(quam_pulse, SquarePulse)
    assert quam_pulse.amplitude == pytest.approx(0.15)  # 0.3 * 0.5 V max_voltage
    assert quam_pulse.length == 40


def test_gaussian_converts_to_symbolic_gaussian_pulse():
    pulse = Pulse(duration=100, amplitude=0.4, envelope=Gaussian(rel_sigma=0.25))
    quam_pulse = quam_pulse_from_qibolab_pulse(pulse, "test", max_voltage=0.5)
    assert isinstance(quam_pulse, GaussianPulse)
    assert quam_pulse.amplitude == pytest.approx(0.2)
    assert quam_pulse.sigma == pytest.approx(25)


def test_drag_falls_back_to_waveform_pulse():
    """Not quam's deprecated DragGaussianPulse -- see _quam_pulses'
    module-level comment on why the sampled fallback is used instead."""
    pulse = Pulse(duration=40, amplitude=0.3, envelope=Drag(rel_sigma=0.2, beta=0.5))
    quam_pulse = quam_pulse_from_qibolab_pulse(pulse, "test", max_voltage=0.5)
    assert isinstance(quam_pulse, WaveformPulse)
    assert quam_pulse.length == 40
    assert quam_pulse.waveform_Q is not None  # Drag has a nonzero Q component


def test_qibolab_amplitude_exceeding_range_raises():
    pulse = Pulse(duration=40, amplitude=1.5, envelope=Rectangular())
    with pytest.raises(AmplitudeOutOfRangeError):
        quam_pulse_from_qibolab_pulse(pulse, "test", max_voltage=0.5)


def test_readout_with_rectangular_probe_converts_to_square_readout_pulse():
    probe = Pulse(duration=1000, amplitude=0.1, envelope=Rectangular())
    readout = Readout(acquisition=Acquisition(duration=1000), probe=probe)
    quam_pulse = quam_readout_pulse_from_qibolab_readout(readout, "readout", max_voltage=0.5)
    assert isinstance(quam_pulse, SquareReadoutPulse)
    assert quam_pulse.amplitude == pytest.approx(0.05)
    assert quam_pulse.length == 1000


def test_readout_with_non_rectangular_probe_falls_back():
    probe = Pulse(duration=1000, amplitude=0.1, envelope=Gaussian(rel_sigma=0.2))
    readout = Readout(acquisition=Acquisition(duration=1000), probe=probe)
    quam_pulse = quam_readout_pulse_from_qibolab_readout(readout, "readout", max_voltage=0.5)
    assert isinstance(quam_pulse, GaussianPulse)
    assert not isinstance(quam_pulse, SquareReadoutPulse)
