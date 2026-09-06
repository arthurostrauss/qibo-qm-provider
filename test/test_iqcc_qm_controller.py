"""Offline tests for ``qibolab_bridge.iqcc_controller.IQCCQmController`` and
its dispatch in ``platform_from_quam._resolve_controller_class``.

None of this needs real IQCC credentials or ``iqcc_cloud_client`` installed
(it genuinely isn't, in this project's own ``.venv`` -- confirmed while
writing these tests): the dispatch fallback is exercised against that real
absence, and the "picks IQCCQmController"/``connect()``/``play()`` paths are
exercised against a minimal fake ``iqcc_cloud_client.qmm_cloud`` module
(``_install_fake_iqcc_cloud_client``) standing in for the real one -- this
package's own code never imports anything beyond
``CloudQuantumMachinesManager``'s ``__init__``/``open_qm`` and
``CloudQuantumMachine``'s ``execute``, so the fake only needs to shadow that.
"""

from __future__ import annotations

import sys
import types

import pytest
from qibolab._core.execution_parameters import AcquisitionType, ExecutionParameters
from qibolab._core.pulses.envelope import Rectangular
from qibolab._core.pulses.pulse import Acquisition, Pulse, Readout
from qibolab._core.sequence import PulseSequence
from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import FluxTunableQuam

from qibo_qm_provider.qibolab_bridge import IQCCQmController, QuamQmController
from qibo_qm_provider.qibolab_bridge.naming import channel_id
from qibo_qm_provider.qibolab_bridge.platform_from_quam import _resolve_controller_class
from qibo_qm_provider.qibolab_bridge.quam_wiring import build_qm_wiring


class _FakeCloudResult:
    def __init__(self, data):
        self._data = data

    def fetch_all(self):
        return self._data


class _FakeCloudResultHandles:
    """Lenient stand-in: any handle name fetches the same dummy value --
    this package never needs to assert on QM's internal operation-naming
    scheme, only that the plumbing from ``execute()`` to ``fetch_results``
    is wired correctly end to end."""

    def wait_for_all_values(self):
        pass

    def get(self, name):
        import numpy as np

        return _FakeCloudResult(np.zeros(1))


class _FakeCloudJob:
    def __init__(self):
        self.result_handles = _FakeCloudResultHandles()


class _FakeCloudQuantumMachine:
    def __init__(self, config):
        self.config = config
        self.executed = []

    def execute(self, program, terminal_output=False, options=None):
        self.executed.append((program, terminal_output, options))
        return _FakeCloudJob()


class _FakeCloudQuantumMachinesManager:
    def __init__(self, backend):
        self.backend = backend
        self.opened_configs = []
        self._qm = None

    def open_qm(self, config, *args, **kwargs):
        self.opened_configs.append(config)
        self._qm = _FakeCloudQuantumMachine(config)
        return self._qm


def _install_fake_iqcc_cloud_client(monkeypatch):
    """Register a minimal fake ``iqcc_cloud_client.qmm_cloud`` module so
    ``IQCCQmController``'s lazy imports resolve without the real package."""
    fake_package = types.ModuleType("iqcc_cloud_client")
    fake_qmm_cloud = types.ModuleType("iqcc_cloud_client.qmm_cloud")
    fake_qmm_cloud.CloudQuantumMachinesManager = _FakeCloudQuantumMachinesManager
    monkeypatch.setitem(sys.modules, "iqcc_cloud_client", fake_package)
    monkeypatch.setitem(sys.modules, "iqcc_cloud_client.qmm_cloud", fake_qmm_cloud)
    return fake_qmm_cloud


def _iqcc_machine() -> FluxTunableQuam:
    machine = FluxTunableQuam()
    machine.network = {
        "host": "10.1.1.6",
        "cluster_name": "test-cluster",
        "qmm_class": "iqcc_cloud_client.qmm_cloud.CloudQuantumMachinesManager",
        "qmm_settings": {"backend": "arbel"},
        "use_custom_qmm": True,
    }
    return machine


def test_resolve_controller_class_falls_back_without_iqcc_cloud_client():
    """The real state of this project's own ``.venv``: ``iqcc_cloud_client``
    is not installed, so an IQCC-shaped ``network`` config must still fall
    back to a plain ``QuamQmController`` (local execution, still carrying
    ``initialize_qpu()``) rather than raising -- the real connection failure
    belongs at ``connect()`` time, not here."""
    assert "iqcc_cloud_client" not in sys.modules
    assert _resolve_controller_class(_iqcc_machine()) is QuamQmController


def test_resolve_controller_class_ignores_missing_qmm_class():
    machine = FluxTunableQuam()
    machine.network = {"host": "1.2.3.4"}
    assert _resolve_controller_class(machine) is QuamQmController


def test_resolve_controller_class_picks_iqcc_controller(monkeypatch):
    _install_fake_iqcc_cloud_client(monkeypatch)
    assert _resolve_controller_class(_iqcc_machine()) is IQCCQmController


def test_connect_delegates_to_machine_connect(monkeypatch):
    _install_fake_iqcc_cloud_client(monkeypatch)
    machine = _iqcc_machine()
    controller = IQCCQmController(address="10.1.1.6:9510", channels={}, machine=machine)

    controller.connect()

    assert isinstance(controller.manager, _FakeCloudQuantumMachinesManager)
    assert controller.manager.backend == "arbel"


def test_connect_rejects_a_non_cloud_manager(monkeypatch):
    _install_fake_iqcc_cloud_client(monkeypatch)
    machine = _iqcc_machine()
    machine.connect = lambda: object()  # e.g. a misconfigured/local machine
    controller = IQCCQmController(address="10.1.1.6:9510", channels={}, machine=machine)

    with pytest.raises(TypeError, match="CloudQuantumMachinesManager"):
        controller.connect()


def test_disconnect_rejects_keep_dc_offsets_off(monkeypatch):
    _install_fake_iqcc_cloud_client(monkeypatch)
    machine = _iqcc_machine()
    controller = IQCCQmController(
        address="10.1.1.6:9510", channels={}, machine=machine, keep_dc_offsets_on=False
    )
    controller.connect()

    with pytest.raises(NotImplementedError, match="close_all_quantum_machines"):
        controller.disconnect()


def test_play_executes_through_the_cloud_manager_and_fetches_results(mw_fem_machine):
    """Full offline exercise of the overridden ``play()``: real channel/pulse/
    acquisition registration (via ``build_qm_wiring``'s real output, same as
    ``QmController.play`` would use), against a fake manager standing in only
    for the network boundary -- ``manager.open_qm(...).execute(...)`` -- to
    confirm the "compile, then queue" replacement is wired correctly end to
    end, including result fetching through ``CloudResultHandles``.
    """
    channels, configs, fems = build_qm_wiring(mw_fem_machine)
    controller = IQCCQmController(
        address="1.2.3.4:9510",
        cluster_name="test-cluster",
        channels=channels,
        fems=fems,
        machine=mw_fem_machine,
    )
    controller.manager = _FakeCloudQuantumMachinesManager(backend="arbel")

    drive_pulse = Pulse(duration=40, amplitude=0.2, envelope=Rectangular())
    probe_pulse = Pulse(duration=1000, amplitude=0.1, envelope=Rectangular())
    readout = Readout(acquisition=Acquisition(duration=1000), probe=probe_pulse)
    sequence = PulseSequence(
        [
            (channel_id("mw0", "drive"), drive_pulse),
            (channel_id("mw0", "acquisition"), readout),
        ]
    )
    options = ExecutionParameters(
        nshots=1, relaxation_time=0, acquisition_type=AcquisitionType.INTEGRATION
    )

    results = controller.play(configs, [sequence], options, [])

    assert len(controller.manager.opened_configs) == 1
    assert len(controller.manager._qm.executed) == 1
    assert results  # fetch_results/Acquisition.fetch ran against the fake handles

    # The strongest regression check that config authority actually moved:
    # what was sent to open_qm() must be exactly machine.generate_config()
    # (captured after play(), so it reflects any op build_sweep_macro
    # registered) -- never a from-scratch qibolab Configuration.
    assert controller.manager.opened_configs[0] == mw_fem_machine.generate_config()


def test_play_with_discrimination_acquisition_through_the_cloud_manager(mw_fem_machine):
    """Same cloud-execution path as ``test_play_executes_through_the_cloud_
    manager_and_fetches_results``, but ``AcquisitionType.DISCRIMINATION`` --
    ``mw0``'s ``"readout"`` op carries a real ``threshold``/
    ``integration_weights_angle`` (see ``conftest.py``), exercising
    ``ShotsAcquisition`` through the IQCC cloud tail too, not just the
    local ``QuamQmController`` one."""
    channels, configs, fems = build_qm_wiring(mw_fem_machine)
    controller = IQCCQmController(
        address="1.2.3.4:9510",
        cluster_name="test-cluster",
        channels=channels,
        fems=fems,
        machine=mw_fem_machine,
    )
    controller.manager = _FakeCloudQuantumMachinesManager(backend="arbel")

    drive_pulse = Pulse(duration=40, amplitude=0.2, envelope=Rectangular())
    probe_pulse = Pulse(duration=1000, amplitude=0.1, envelope=Rectangular())
    readout = Readout(acquisition=Acquisition(duration=1000), probe=probe_pulse)
    sequence = PulseSequence(
        [
            (channel_id("mw0", "drive"), drive_pulse),
            (channel_id("mw0", "acquisition"), readout),
        ]
    )
    options = ExecutionParameters(
        nshots=1, relaxation_time=0, acquisition_type=AcquisitionType.DISCRIMINATION
    )

    results = controller.play(configs, [sequence], options, [])

    assert len(controller.manager.opened_configs) == 1
    assert results  # fetch_results/ShotsAcquisition.fetch ran against the fake handles
