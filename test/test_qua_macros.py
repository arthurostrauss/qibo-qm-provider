"""Tests for qibo_qm_provider.qibolab_bridge.qua_macros.

Uses the ``mw_fem_machine`` fixture (real OPX1000 MW-FEM/LF-FEM port
objects, already-registered x180/x90/y90/-y90/readout operations) -- the
same fixture ``test_quam_wiring.py`` uses for the working-instruments path,
since ``sequence_to_qua_macro`` needs real QuAM channels to call
``.play()``/``.measure()``/``.wait()``/``.align()`` on. QUA emission itself
(``channel.play(...)`` etc.) requires an active ``with qm.qua.program():``
scope -- provided by each test that actually calls the returned macro;
``build_operation_index``/``_resolve_operation`` are pure Python and need
no such scope.
"""

import pytest
from qibolab._core.pulses.envelope import Gaussian, Rectangular
from qibolab._core.pulses.pulse import Acquisition, Align, Delay, Pulse, Readout, VirtualZ
from qibolab._core.sequence import PulseSequence
from qm import qua

from qibo_qm_provider.qibolab_bridge.naming import channel_id
from qibo_qm_provider.qibolab_bridge.qua_macros import build_operation_index, sequence_to_qua_macro


def test_build_operation_index_finds_existing_pulse_by_shape(mw_fem_machine):
    """x180/x90/y90/-y90 are all Rectangular, duration-40 pulses on mw0's
    drive channel -- they collide onto the *same* shape key (amplitude is
    deliberately excluded from it, see qua_macros._shape_key), so only one
    survives as the index's canonical reference for that shape. Which one
    survives is an implementation detail (dict-iteration order); what must
    hold is that whichever one it is correctly anchors amplitude_scale for
    any newly-resolved pulse of that same shape."""
    index = build_operation_index(mw_fem_machine)
    drive_shapes = index[channel_id("mw0", "drive")]

    key = ("rectangular", 40.0, ())
    assert key in drive_shapes
    op_name, template = drive_shapes[key]
    assert op_name in mw_fem_machine.qubits["mw0"].xy.operations
    assert template.envelope.kind == "rectangular"
    assert template.duration == 40.0


def test_sequence_to_qua_macro_reuses_matching_operation(mw_fem_machine):
    pulse = Pulse(duration=40, amplitude=0.2, envelope=Rectangular())
    sequence = PulseSequence([(channel_id("mw0", "drive"), pulse)])
    n_ops_before = len(mw_fem_machine.qubits["mw0"].xy.operations)

    with qua.program():
        macro = sequence_to_qua_macro(mw_fem_machine, sequence)
        macro()

    assert len(mw_fem_machine.qubits["mw0"].xy.operations) == n_ops_before  # matched, nothing new


def test_sequence_to_qua_macro_registers_missing_pulse(mw_fem_machine):
    pulse = Pulse(duration=64, amplitude=0.3, envelope=Gaussian(rel_sigma=0.25))
    sequence = PulseSequence([(channel_id("mw0", "drive"), pulse)])
    n_ops_before = len(mw_fem_machine.qubits["mw0"].xy.operations)

    with qua.program():
        macro = sequence_to_qua_macro(mw_fem_machine, sequence)
        macro()

    assert len(mw_fem_machine.qubits["mw0"].xy.operations) == n_ops_before + 1


def test_sequence_to_qua_macro_register_missing_false_raises(mw_fem_machine):
    pulse = Pulse(duration=64, amplitude=0.3, envelope=Gaussian(rel_sigma=0.25))
    sequence = PulseSequence([(channel_id("mw0", "drive"), pulse)])

    with qua.program():
        macro = sequence_to_qua_macro(mw_fem_machine, sequence, register_missing=False)
        with pytest.raises(ValueError, match="No matching QuAM operation"):
            macro()


def test_sequence_to_qua_macro_emits_delay_virtualz_align_without_raising(mw_fem_machine):
    drive_pulse = Pulse(duration=40, amplitude=0.2, envelope=Rectangular())
    delay = Delay(duration=16)
    vz = VirtualZ(phase=0.5)
    align = Align()

    sequence = PulseSequence(
        [
            (channel_id("mw0", "drive"), drive_pulse),
            (channel_id("mw0", "drive"), delay),
            (channel_id("mw0", "drive"), vz),
            (channel_id("mw0", "drive"), align),
            (channel_id("mw1", "drive"), align),
        ]
    )

    with qua.program():
        macro = sequence_to_qua_macro(mw_fem_machine, sequence)
        macro()


def test_sequence_to_qua_macro_readout_populates_acquisitions(mw_fem_machine):
    probe = Pulse(duration=1000, amplitude=0.1, envelope=Rectangular())
    readout = Readout(acquisition=Acquisition(duration=1000), probe=probe)
    sequence = PulseSequence([(channel_id("mw0", "acquisition"), readout)])

    with qua.program():
        macro = sequence_to_qua_macro(mw_fem_machine, sequence)
        macro()

    assert readout.acquisition.id in macro.acquisitions
    i_var, q_var = macro.acquisitions[readout.acquisition.id]
    assert i_var is not None and q_var is not None


def test_sequence_to_qua_macro_amplitude_parameter_override(mw_fem_machine):
    pulse = Pulse(duration=40, amplitude=0.2, envelope=Rectangular())
    sequence = PulseSequence([(channel_id("mw0", "drive"), pulse)])

    with qua.program():
        amp_var = qua.declare(qua.fixed)
        macro = sequence_to_qua_macro(mw_fem_machine, sequence, parameters={"amp": (pulse.id, "amplitude")})
        macro(amp_var)  # a live QUA variable overrides the shape-matched amplitude_scale


def test_sequence_to_qua_macro_phase_parameter_override(mw_fem_machine):
    """Regression test: overriding a pulse's phase with a live QUA variable
    used to crash with QmQuaException, because the phase was tested for
    truthiness (``if phase:``) instead of ``is not None`` -- ``bool()`` on a
    QUA variable is not allowed. See symbolic_circuit_lowering.md."""
    pulse = Pulse(duration=40, amplitude=0.2, envelope=Rectangular())
    sequence = PulseSequence([(channel_id("mw0", "drive"), pulse)])

    with qua.program():
        phase_var = qua.declare(qua.fixed)
        macro = sequence_to_qua_macro(mw_fem_machine, sequence, parameters={"phase": (pulse.id, "phase")})
        macro(phase_var)  # a live QUA variable overrides the pulse's (absent) relative_phase


def test_sequence_to_qua_macro_invalid_parameter_field_raises(mw_fem_machine):
    pulse = Pulse(duration=40, amplitude=0.2, envelope=Rectangular())
    sequence = PulseSequence([(channel_id("mw0", "drive"), pulse)])

    with pytest.raises(ValueError, match="field"):
        sequence_to_qua_macro(mw_fem_machine, sequence, parameters={"bad": (pulse.id, "frequency")})
