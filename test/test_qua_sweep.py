"""Tests for qibo_qm_provider.qibolab_bridge.qua_sweep.

Uses the ``mw_fem_machine`` fixture (same as ``test_qua_macros.py``) --
mw0's drive/resonator channels have real, distinct LOs (5.0 GHz drive,
7.0 GHz resonator, both shared with mw1's resonator), which is exactly what
the frequency-sweep validation tests need without any extra fixture setup.
"""

import numpy as np
import pytest
from qibolab._core.pulses.envelope import Rectangular
from qibolab._core.pulses.pulse import Delay, Pulse, VirtualZ
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper
from qm import generate_qua_script, qua

from qibo_qm_provider.qibolab_bridge.naming import channel_id
from qibo_qm_provider.qibolab_bridge.qua_sweep import build_sweep_macro


def _script(prog) -> str:
    return generate_qua_script(prog)


def test_no_sweepers_behaves_like_a_bare_sequence(mw_fem_machine):
    pulse = Pulse(duration=40, amplitude=0.2, envelope=Rectangular())
    sequence = PulseSequence([(channel_id("mw0", "drive"), pulse)])

    with qua.program() as prog:
        run = build_sweep_macro(mw_fem_machine, sequence, [])
        run()

    script = _script(prog)
    assert "play(" in script
    assert "for_(" not in script and "for_each_" not in script


def test_amplitude_sweep_registers_rescaled_reference_pulse(mw_fem_machine):
    pulse = Pulse(duration=40, amplitude=0.2, envelope=Rectangular())
    sequence = PulseSequence([(channel_id("mw0", "drive"), pulse)])
    values = np.linspace(-0.9, 0.9, 5)
    sweeper = Sweeper(parameter=Parameter.amplitude, values=values, pulses=[pulse])
    n_ops_before = len(mw_fem_machine.qubits["mw0"].xy.operations)

    with qua.program() as prog:
        run = build_sweep_macro(mw_fem_machine, sequence, [[sweeper]])
        run()

    # exactly one new, dedicated reference operation registered (not reusing
    # x180/x90/y90/-y90, and not clobbering any of them).
    assert len(mw_fem_machine.qubits["mw0"].xy.operations) == n_ops_before + 1
    from qibolab._core.instruments.qm.program.sweepers import sweeper_amplitude

    from qibo_qm_provider.qibolab_bridge.quam_pulses import max_voltage_for_channel

    new_ops = {
        name: op
        for name, op in mw_fem_machine.qubits["mw0"].xy.operations.items()
        if name.startswith("qua_sweep_ref_")
    }
    assert len(new_ops) == 1
    (reference_op,) = new_ops.values()
    max_voltage = max_voltage_for_channel(mw_fem_machine.qubits["mw0"].xy)
    # reference_op.amplitude is QuAM's own (volts) convention; sweeper_amplitude
    # is qibolab's dimensionless [-1, 1] convention -- convert before comparing.
    assert reference_op.amplitude == pytest.approx(sweeper_amplitude(values) * max_voltage)
    assert reference_op.length == 40  # shape (duration) untouched

    script = _script(prog)
    assert "for_(" in script
    assert "amplitude_scale=" in script


def test_duration_interpolated_sweep_registers_minimum_duration_reference(mw_fem_machine):
    pulse = Pulse(duration=80, amplitude=0.2, envelope=Rectangular())
    sequence = PulseSequence([(channel_id("mw0", "drive"), pulse)])
    values = np.array([40.0, 80.0, 120.0])
    sweeper = Sweeper(parameter=Parameter.duration_interpolated, values=values, pulses=[pulse])

    with qua.program() as prog:
        run = build_sweep_macro(mw_fem_machine, sequence, [[sweeper]])
        run()

    new_ops = {
        name: op
        for name, op in mw_fem_machine.qubits["mw0"].xy.operations.items()
        if name.startswith("qua_sweep_ref_")
    }
    assert len(new_ops) == 1
    (reference_op,) = new_ops.values()
    assert reference_op.length == 40  # min(values), not the original 80

    script = _script(prog)
    assert "for_(" in script


def test_relative_phase_sweep_emits_frame_rotation(mw_fem_machine):
    pulse = Pulse(duration=40, amplitude=0.2, envelope=Rectangular())
    sequence = PulseSequence([(channel_id("mw0", "drive"), pulse)])
    sweeper = Sweeper(parameter=Parameter.relative_phase, values=np.linspace(0, 2 * np.pi, 4), pulses=[pulse])

    with qua.program() as prog:
        run = build_sweep_macro(mw_fem_machine, sequence, [[sweeper]])
        run()

    script = _script(prog)
    assert "frame_rotation_2pi(" in script
    assert "for_(" in script


def test_phase_sweeper_requires_virtualz_and_emits_frame_rotation(mw_fem_machine):
    vz = VirtualZ(phase=0.0)
    sequence = PulseSequence([(channel_id("mw0", "drive"), vz)])
    sweeper = Sweeper(parameter=Parameter.phase, values=np.linspace(0, 2 * np.pi, 4), pulses=[vz])

    with qua.program() as prog:
        run = build_sweep_macro(mw_fem_machine, sequence, [[sweeper]])
        run()

    script = _script(prog)
    assert "frame_rotation_2pi(" in script


def test_frequency_sweep_emits_update_frequency(mw_fem_machine):
    lo = mw_fem_machine.qubits["mw0"].xy.upconverter_frequency
    values = lo + np.array([-1e6, 0.0, 1e6])
    sweeper = Sweeper(parameter=Parameter.frequency, values=values, channels=[channel_id("mw0", "drive")])
    sequence = PulseSequence([(channel_id("mw0", "drive"), Pulse(duration=40, amplitude=0.2, envelope=Rectangular()))])

    with qua.program() as prog:
        run = build_sweep_macro(mw_fem_machine, sequence, [[sweeper]])
        run()

    script = _script(prog)
    assert "update_frequency(" in script


def test_frequency_sweep_rejects_mismatched_lo_channels(mw_fem_machine):
    """mw0's drive (5.0 GHz) and resonator (7.0 GHz, shared_resonator_out)
    have different LOs -- sweeping frequency across both in one Sweeper
    must raise, matching qibolab's own find_lo_frequencies."""
    values = np.array([-1e6, 0.0, 1e6])
    sweeper = Sweeper(
        parameter=Parameter.frequency,
        values=values,
        channels=[channel_id("mw0", "drive"), channel_id("mw0", "probe")],
    )
    sequence = PulseSequence([(channel_id("mw0", "drive"), Pulse(duration=40, amplitude=0.2, envelope=Rectangular()))])

    with qua.program():
        with pytest.raises(ValueError, match="different LO"):
            run = build_sweep_macro(mw_fem_machine, sequence, [[sweeper]])
            run()


def test_frequency_sweep_rejects_out_of_bandwidth_values(mw_fem_machine):
    lo = mw_fem_machine.qubits["mw0"].xy.upconverter_frequency
    values = lo + np.array([0.0, 1e9])  # far beyond the 400 MHz bandwidth
    sweeper = Sweeper(parameter=Parameter.frequency, values=values, channels=[channel_id("mw0", "drive")])
    sequence = PulseSequence([(channel_id("mw0", "drive"), Pulse(duration=40, amplitude=0.2, envelope=Rectangular()))])

    with qua.program():
        with pytest.raises(ValueError, match="bandwidth"):
            run = build_sweep_macro(mw_fem_machine, sequence, [[sweeper]])
            run()


def test_offset_sweep_emits_clamped_set_dc_offset(mw_fem_machine):
    sweeper = Sweeper(parameter=Parameter.offset, values=np.array([-0.6, 0.0, 0.6]), channels=[channel_id("mw0", "flux")])
    sequence = PulseSequence([(channel_id("mw0", "flux"), Delay(duration=16))])

    with qua.program() as prog:
        run = build_sweep_macro(mw_fem_machine, sequence, [[sweeper]])
        run()

    script = _script(prog)
    assert "set_dc_offset(" in script
    assert "for_(" in script


def test_unsupported_parameter_raises_not_implemented(mw_fem_machine):
    pulse = Pulse(duration=40, amplitude=0.2, envelope=Rectangular())
    sequence = PulseSequence([(channel_id("mw0", "drive"), pulse)])
    sweeper = Sweeper(parameter=Parameter.duration, values=np.array([40.0, 80.0]), pulses=[pulse])

    with qua.program():
        with pytest.raises(NotImplementedError, match="duration"):
            build_sweep_macro(mw_fem_machine, sequence, [[sweeper]])


def test_parallel_sweepers_use_for_each(mw_fem_machine):
    """Two sweepers in the same ParallelSweepers group (co-swept) must
    compile via qua.for_each_, not two independent for_ loops."""
    pulse = Pulse(duration=40, amplitude=0.2, envelope=Rectangular())
    vz = VirtualZ(phase=0.0)
    sequence = PulseSequence(
        [
            (channel_id("mw0", "drive"), pulse),
            (channel_id("mw0", "drive"), vz),
        ]
    )
    amp_sweeper = Sweeper(parameter=Parameter.amplitude, values=np.linspace(-0.9, 0.9, 4), pulses=[pulse])
    phase_sweeper = Sweeper(parameter=Parameter.phase, values=np.linspace(0, 2 * np.pi, 4), pulses=[vz])

    with qua.program() as prog:
        run = build_sweep_macro(mw_fem_machine, sequence, [[amp_sweeper, phase_sweeper]])
        run()

    script = _script(prog)
    assert "for_each_(" in script
