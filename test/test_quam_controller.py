"""Offline tests for ``qibolab_bridge.quam_controller.QuamQmController`` --
the local (non-cloud) counterpart to ``test_iqcc_qm_controller.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from qibolab._core.execution_parameters import AcquisitionType, ExecutionParameters
from qibolab._core.pulses.envelope import Rectangular
from qibolab._core.pulses.pulse import Acquisition, Pulse, Readout
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper

from qibo_qm_provider.qibolab_bridge import QuamQmController
from qibo_qm_provider.qibolab_bridge.naming import channel_id
from qibo_qm_provider.qibolab_bridge.quam_wiring import build_qm_wiring


class _FakeResult:
    def __init__(self, data):
        self._data = data

    def fetch_all(self):
        return self._data


class _FakeResultHandles:
    def wait_for_all_values(self):
        pass

    def get(self, name):
        return _FakeResult(np.zeros(1))


class _FakeJob:
    def __init__(self):
        self.result_handles = _FakeResultHandles()


class _FakePendingJob:
    def wait_for_execution(self):
        return _FakeJob()


class _FakeQueue:
    def add_compiled(self, program_id):
        return _FakePendingJob()


class _FakeQuantumMachine:
    def __init__(self, config):
        self.config = config
        self.queue = _FakeQueue()

    def compile(self, program):
        return "fake-program-id"


class _FakeQuantumMachinesManager:
    def __init__(self):
        self.opened_configs = []

    def open_qm(self, config, *args, **kwargs):
        self.opened_configs.append(config)
        return _FakeQuantumMachine(config)


def _sequence(mw_fem_machine):
    drive_pulse = Pulse(duration=40, amplitude=0.2, envelope=Rectangular())
    probe_pulse = Pulse(duration=1000, amplitude=0.1, envelope=Rectangular())
    readout = Readout(acquisition=Acquisition(duration=1000), probe=probe_pulse)
    return PulseSequence(
        [
            (channel_id("mw0", "drive"), drive_pulse),
            (channel_id("mw0", "acquisition"), readout),
        ]
    )


def _controller(mw_fem_machine):
    channels, _, fems = build_qm_wiring(mw_fem_machine)
    return QuamQmController(address="1.2.3.4:9510", channels=channels, fems=fems, machine=mw_fem_machine)


def test_play_without_manager_returns_program_and_generate_config(mw_fem_machine):
    """``manager is None`` fallback: the returned "config" must be exactly
    ``machine.generate_config()`` -- never a from-scratch qibolab
    ``Configuration`` -- confirming config authority moved for the local
    controller too, not just the IQCC one."""
    controller = _controller(mw_fem_machine)
    sequence = _sequence(mw_fem_machine)
    options = ExecutionParameters(nshots=1, relaxation_time=0, acquisition_type=AcquisitionType.INTEGRATION)

    with pytest.warns(UserWarning, match="Not connected"):
        result = controller.play({}, [sequence], options, [])

    assert set(result) == {"program", "config"}
    assert result["config"] == mw_fem_machine.generate_config()


def test_play_rejects_raw_acquisition(mw_fem_machine):
    controller = _controller(mw_fem_machine)
    sequence = _sequence(mw_fem_machine)
    options = ExecutionParameters(nshots=1, relaxation_time=0, acquisition_type=AcquisitionType.RAW)

    with pytest.raises(NotImplementedError, match="RAW"):
        controller.play({}, [sequence], options, [])


def test_play_with_discrimination_acquisition_compiles_offline(mw_fem_machine):
    """``AcquisitionType.DISCRIMINATION`` compiles through the full
    program-building path (manager=None) -- ``mw0``'s ``"readout"`` op
    carries a real ``threshold``/``integration_weights_angle`` (see
    ``conftest.py``), so this exercises ``ShotsAcquisition`` end to end,
    not just the ``INTEGRATION`` path ``test_play_without_manager_returns_
    program_and_generate_config`` already covers.
    """
    controller = _controller(mw_fem_machine)
    sequence = _sequence(mw_fem_machine)
    options = ExecutionParameters(nshots=3, relaxation_time=0, acquisition_type=AcquisitionType.DISCRIMINATION)

    with pytest.warns(UserWarning, match="Not connected"):
        result = controller.play({}, [sequence], options, [])

    assert set(result) == {"program", "config"}


def test_play_executes_through_a_fake_local_manager_and_fetches_results(mw_fem_machine):
    """Full offline exercise of the compile-then-queue-then-wait tail
    (distinct from IQCCQmController's cloud execute() tail), against a fake
    local QuantumMachinesManager standing in only for the network boundary."""
    controller = _controller(mw_fem_machine)
    controller.manager = _FakeQuantumMachinesManager()
    sequence = _sequence(mw_fem_machine)
    options = ExecutionParameters(nshots=1, relaxation_time=0, acquisition_type=AcquisitionType.INTEGRATION)

    results = controller.play({}, [sequence], options, [])

    assert len(controller.manager.opened_configs) == 1
    assert controller.manager.opened_configs[0] == mw_fem_machine.generate_config()
    assert results


def test_play_continues_past_an_all_empty_batch_and_keeps_later_results(mw_fem_machine, monkeypatch):
    """Regression for the empty-unrolled-sequence hazard qibolab's own
    ``QmController.play()`` has (a bare ``return {}``, verified against the
    installed qibolab): an all-empty batch must ``continue``, not abort the
    whole call -- otherwise every later, real batch's results would be
    silently dropped. ``_batch`` is monkeypatched because qibolab's real
    batching (grouped by resource bounds) never isolates an empty sequence
    into its own later batch -- but the hazard is in the loop body, which
    this exercises directly."""
    import qibo_qm_provider.qibolab_bridge.quam_controller as quam_controller_module

    controller = _controller(mw_fem_machine)
    controller.manager = _FakeQuantumMachinesManager()
    real_sequence = _sequence(mw_fem_machine)
    monkeypatch.setattr(
        quam_controller_module,
        "_batch",
        lambda sequences: iter([[PulseSequence([])], [real_sequence]]),
    )
    options = ExecutionParameters(nshots=1, relaxation_time=0, acquisition_type=AcquisitionType.INTEGRATION)

    results = controller.play({}, [real_sequence], options, [])

    assert results
    assert len(controller.manager.opened_configs) == 1


def test_play_with_amplitude_sweep_compiles_offline(mw_fem_machine):
    """Compile-only check (manager=None) that a real sweep threads through
    the new play() without raising -- numeric correctness is covered by
    test_qua_sweep.py; this is the integration point (shot loop + sweep
    loop + acquisition download all composed together)."""
    controller = _controller(mw_fem_machine)
    drive_pulse = Pulse(duration=40, amplitude=0.2, envelope=Rectangular())
    sequence = PulseSequence([(channel_id("mw0", "drive"), drive_pulse)])
    sweeper = Sweeper(parameter=Parameter.amplitude, values=np.linspace(-0.9, 0.9, 5), pulses=[drive_pulse])
    options = ExecutionParameters(nshots=3, relaxation_time=0, acquisition_type=AcquisitionType.INTEGRATION)

    with pytest.warns(UserWarning, match="Not connected"):
        result = controller.play({}, [sequence], options, [[sweeper]])

    assert "program" in result
