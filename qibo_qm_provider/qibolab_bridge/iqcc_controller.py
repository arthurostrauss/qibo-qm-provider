"""``IQCCQmController``: a ``QuamQmController`` that executes through IQCC's
cloud manager instead of a local ``QuantumMachinesManager``.

Subclasses :class:`~.quam_controller.QuamQmController` (not ``QmController``
directly) so that the whole QuAM-macro execution path -- ``initialize_qpu()``,
``build_sweep_macro``, ``qua_config(machine)`` as the sole config authority --
propagates through cloud execution the same way it does through local
execution; only :meth:`connect`/:meth:`disconnect`/:meth:`play` below are
IQCC's own concern: the cloud manager's execution shape.

``qibolab._core.instruments.qm.controller.QmController`` (unmodified) makes
two hardcoded assumptions about ``self.manager``:

* ``connect()`` always builds a plain ``qm.QuantumMachinesManager`` from
  ``self.address`` (an ``"host:port"`` string) -- there is no hook for a
  different manager class at all.
* ``play()`` then assumes that manager's full *local* surface:
  ``manager.open_qm(config)`` returns a ``QuantumMachine`` with
  ``.compile(program) -> program_id`` and
  ``.queue.add_compiled(program_id) -> QmPendingJob``.

``iqcc_cloud_client.qmm_cloud.CloudQuantumMachinesManager``/
``CloudQuantumMachine`` expose none of that -- only a single, synchronous
``CloudQuantumMachine.execute(program, options) -> CloudJob``, with no
``compile``/``queue`` at all. Verified directly (not assumed) that everything
*around* that gap already works unmodified against the cloud classes:
``CloudQuantumMachinesManager.open_qm(config)`` matches
``QuamQmController.play()``'s call exactly, and
:func:`~.qua_acquisition.IntegratedAcquisition.fetch` only ever calls
``handles.get(name).fetch_all()``, which ``CloudResultHandles``/
``CloudResult`` already implement (``wait_for_all_values`` is a documented
no-op on the cloud side too). So the only code that needs replacing is the
"compile, then queue, then wait" sandwich in the middle -- not the program-
building steps before it or the result-fetching after it.

Why this is IQCC-specific, not a generic "cloud-capable" base class
--------------------------------------------------------------------
A different cloud provider's manager could have a completely different
execution/result-handle shape (async submission, a different handle
protocol, ...). Branching on "is this a cloud manager" inside one shared
class would just move today's problem into a growing if/elif ladder instead
of a new, independent subclass the next provider can write on its own terms.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import Field
from qibolab._core.components import Config
from qibolab._core.execution_parameters import AcquisitionType, ExecutionParameters
from qibolab._core.identifier import Result
from qibolab._core.instruments.qm.controller import _batch, _unroll_sequences
from qibolab._core.pulses.pulse import PulseId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from .qua_acquisition import fetch_results
from .quam_controller import QuamQmController

if TYPE_CHECKING:
    from iqcc_cloud_client.qmm_cloud import CloudQuantumMachinesManager

__all__ = ["IQCCQmController"]


class IQCCQmController(QuamQmController):
    """A ``QuamQmController`` whose manager/execution come from IQCC's cloud SDK.

    Built the same way as a plain ``QuamQmController`` (``address``/
    ``channels``/``fems``/``machine`` -- ``address`` is kept only for
    ``Instrument.signature`` and is never used to build the manager here),
    plus two IQCC-specific extra fields:

    Attributes:
        machine: (Inherited from ``QuamQmController``.) The source QuAM
            object. ``connect()`` delegates to ``machine.connect()``
            (``quam_builder``'s own, unmodified config-driven resolution of
            ``network.qmm_class``/``network.qmm_settings``) rather than
            re-deriving IQCC connection details here -- the same resolution
            ``qiskit_qm_provider.QMBackend.qmm`` already relies on for the
            OpenQASM path, so a machine's own state decides local vs. IQCC,
            not this class.
        execute_options: Extra keyword arguments forwarded to
            ``CloudQuantumMachine.execute``'s ``options`` (e.g.
            ``{"timeout": 120}`` -- ``iqcc_cloud_client`` otherwise defaults
            to 60s via ``$IQCC_DEFAULT_TIMEOUT``). Copied before each call,
            since ``execute`` mutates the dict it is given.
        terminal_output: Forwarded to ``CloudQuantumMachine.execute`` verbatim.
    """

    execute_options: Dict[str, Any] = Field(default_factory=dict)
    terminal_output: bool = False

    def connect(self) -> None:
        """Connect via ``self.machine.connect()`` instead of ``self.address``.

        Raises:
            TypeError: If ``self.machine.connect()`` does not return a
                ``CloudQuantumMachinesManager`` -- a misconfigured or
                genuinely local machine should fail here, loudly and by
                name, rather than later inside :meth:`play` with an opaque
                ``AttributeError`` naming neither IQCC nor this controller.
        """
        from iqcc_cloud_client.qmm_cloud import CloudQuantumMachinesManager

        self._temporary_calibration()
        manager = self.machine.connect()
        if not isinstance(manager, CloudQuantumMachinesManager):
            raise TypeError(
                f"IQCCQmController.connect() expected self.machine.connect() to "
                f"return a CloudQuantumMachinesManager (self.machine.network's "
                f"'qmm_class' should point at "
                f"'iqcc_cloud_client.CloudQuantumMachinesManager'), got "
                f"{type(manager).__name__} instead. Use a plain QmController "
                f"for a locally-wired machine."
            )
        self.manager = manager

    def disconnect(self) -> None:
        """Disconnect. ``keep_dc_offsets_on=False`` has no cloud equivalent.

        Raises:
            NotImplementedError: If ``keep_dc_offsets_on`` was set to
                ``False`` -- the base implementation would otherwise call
                ``self.manager.close_all_quantum_machines()``, which
                ``CloudQuantumMachinesManager`` does not have.
        """
        self._reset_temporary_calibration()
        if self.manager is not None and not self.keep_dc_offsets_on:
            raise NotImplementedError(
                "IQCCQmController.disconnect(keep_dc_offsets_on=False) has no cloud "
                "equivalent -- CloudQuantumMachinesManager has no "
                "close_all_quantum_machines(). Leave keep_dc_offsets_on at its "
                "default (True) for an IQCC-connected controller."
            )
        self.manager = None

    def play(
        self,
        configs: Dict[str, Config],
        sequences: List[PulseSequence],
        options: ExecutionParameters,
        sweepers: List[ParallelSweepers],
    ) -> Dict[PulseId, Result] | Dict[str, object]:
        """Same program-building as ``QuamQmController.play()`` (shared via
        :meth:`~.quam_controller.QuamQmController._build_batch_program`),
        executed through ``CloudQuantumMachine.execute()`` instead of the
        local SDK's compile-then-queue-then-wait sequence.

        Not a call to ``super().play()`` with ``self.manager`` swapped in:
        a ``CloudQuantumMachine`` has no ``.compile()``/``.queue`` at all,
        so the compile/queue/wait tail has to be replaced outright rather
        than intercepted. Only the batching/empty-batch handling and the
        final "open, then execute" step are duplicated from the base
        class -- both routed through the same shared helper so they cannot
        drift from it independently.
        """
        if options.acquisition_type is AcquisitionType.RAW:
            raise NotImplementedError(
                f"{type(self).__name__}.play() does not support AcquisitionType.RAW yet -- "
                "a documented v2 follow-up, see the Option-A convergence plan (repo root)."
            )

        results: Dict[PulseId, Result] = {}
        for batched_sequences in _batch(sequences):
            if len(batched_sequences) == 0:
                continue
            elif len(batched_sequences) == 1:
                sequence = batched_sequences[0]
            else:
                sequence, _ = _unroll_sequences(batched_sequences, options.relaxation_time)
            if len(sequence) == 0:
                continue

            qua_program, config, run = self._build_batch_program(sequence, options, sweepers)

            if self.manager is None:
                assert not results, (
                    "self.manager cannot become None mid-call -- a disconnected "
                    "multi-batch play() must have returned on its first non-empty batch."
                )
                warnings.warn(
                    "Not connected to Quantum Machines. Returning program and config.",
                    stacklevel=2,
                )
                return {"program": qua_program, "config": config}

            cloud_qm = self.manager.open_qm(config)
            job = cloud_qm.execute(
                qua_program,
                terminal_output=self.terminal_output,
                options=dict(self.execute_options),
            )
            handles = job.result_handles
            handles.wait_for_all_values()
            results |= fetch_results(handles, run.acquisitions.values())
        return results
