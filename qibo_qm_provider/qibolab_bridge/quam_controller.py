"""``QuamQmController``: a ``qibolab`` ``QmController`` whose ``play()``
lowers a circuit's pulse sequence onto QuAM's own channel objects
(via :func:`~.qua_sweep.build_sweep_macro`/:func:`~.qua_macros.
sequence_to_qua_macro`) instead of building a QM ``Configuration`` from
scratch via qibolab's own ``Config`` classes -- so the wire config sent to
hardware is always ``machine.generate_config()`` (:func:`~.qm_config_source.
qua_config`), QuAM's own always-correct generator, never re-derived.

This retires the from-scratch config-authoring path this package used
until now (``qibolab._core.instruments.qm.program.instructions.program`` +
qibolab's own ``Config``/``Acquisition`` classes, built up in
``quam_wiring.py`` and consumed via ``configure_channel``/
``register_pulses``/``register_acquisitions``). That path caused two real
bugs this session, both structural, not incidental: a flux-filter
double-application (qibolab's ``OpxOutputConfig.filter()`` convolves real
FIR taps together with each ``ExponentialFilter``'s own derived
approximation, exceeding hardware's 48-tap limit) and a TWPA
``initialize_qpu()`` failure (qibolab's ``Config`` classes have no
representation for QM's ``sticky`` output mode at all). Both become
structurally impossible here, not patched, because there is no second
config-authoring code path left to diverge from QuAM's own -- see the
Option-A convergence plan (repo root) for the full design discussion.

``quam_wiring.build_qm_wiring``'s channel/config wiring is **not** retired
by this change -- ``Platform.channels``/``Platform.parameters.configs``
still need it, for Qibocal ``Sweeper.channels`` addressing and topology
inspection, independent of who authors the wire bytes. This controller's
``play()`` simply never reads ``configs`` for that purpose anymore; the
parameter stays in the signature only because ``Controller.play()``'s
abstract signature requires it.

Supported real-time sweeps (v1): ``phase``/``relative_phase``,
``amplitude``, ``duration_interpolated``, ``frequency``, ``offset`` --
see :mod:`~.qua_sweep` for the full parameter-type breakdown and why
each is handled the way it is. Non-interpolated ``duration`` sweeps and
``RAW`` acquisition are an explicit, documented v2 follow-up
(:func:`~.qua_sweep.build_sweep_macro` raises ``NotImplementedError`` for
the former; :meth:`play` raises for the latter) -- not silently ignored.
``AcquisitionType.DISCRIMINATION`` is supported (:class:`~.qua_acquisition.
ShotsAcquisition`), sourcing ``threshold``/``angle`` from the QuAM readout
``Pulse`` actually played.

``initialize_qpu()`` is called once per built QUA program -- i.e. once per
*batch* (see :meth:`play`'s own docstring), matching qibolab's own
``QmController.play()``'s per-batch granularity -- regardless of how many
shots/sweep points that program executes. (Confirmed with the user this
session, before this class existed in its current form.)
"""

from __future__ import annotations

import warnings
from typing import Dict, List

from qm import generate_qua_script, qua
from qm.qua import declare, for_
from qibolab._core.components import Config
from qibolab._core.execution_parameters import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab._core.identifier import Result
from qibolab._core.instruments.qm import QmController
from qibolab._core.instruments.qm.controller import _batch, _unroll_sequences
from qibolab._core.pulses.pulse import PulseId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers
from quam.core import QuamRoot

from .qm_config_source import qua_config
from .qua_acquisition import fetch_results
from .qua_sweep import build_sweep_macro

__all__ = ["QuamQmController"]


class QuamQmController(QmController):
    """A ``QmController`` backed by a QuAM machine, executing through
    :func:`~.qua_sweep.build_sweep_macro` -- see the module docstring.

    Attributes:
        machine: The QuAM root providing ``initialize_qpu()`` and every
            channel/operation :func:`~.qua_sweep.build_sweep_macro` plays
            on.
    """

    machine: QuamRoot

    def play(
        self,
        configs: Dict[str, Config],
        sequences: List[PulseSequence],
        options: ExecutionParameters,
        sweepers: List[ParallelSweepers],
    ) -> Dict[PulseId, Result] | Dict[str, object]:
        """Play ``sequences`` against ``self.machine``, batched the same
        way qibolab's own ``QmController.play()`` batches them
        (``_batch``/``_unroll_sequences``, unchanged -- both operate on
        qibolab ``PulseSequence`` objects only, agnostic to how the
        resulting sequence gets turned into QUA).

        Per batch: one QUA program is built (``self.machine.
        initialize_qpu()``, once, then the shot loop wrapping one
        :func:`~.qua_sweep.build_sweep_macro`-built macro call), compiled
        against ``qua_config(self.machine)``, and executed.

        Raises:
            NotImplementedError: If ``options.acquisition_type`` is
                ``AcquisitionType.RAW`` (a documented v2 follow-up -- see
                the module docstring) or if any sweeper's parameter is
                unsupported (raised by :func:`~.qua_sweep.build_sweep_macro`
                itself).
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
                return {}

            with qua.program() as qua_program:
                self.machine.initialize_qpu()
                n = declare(int)
                run = build_sweep_macro(
                    self.machine,
                    sequence,
                    sweepers,
                    register_missing=True,
                    average=options.averaging_mode is AveragingMode.CYCLIC,
                    relaxation_time=options.relaxation_time,
                    acquisition_type=options.acquisition_type,
                )
                with for_(n, 0, n < options.nshots, n + 1):
                    run()
                # INTEGRATION's results_shape reserves a leading "2" (I/Q) axis
                # that is never a real buffer axis (I and Q are two separate
                # streams, not a buffer dim); DISCRIMINATION's does not have
                # one at all -- direct port of qibolab's own
                # `options.results_shape(sweepers)[::-1][int(has_iq):]`
                # (instructions.py's `program()`).
                has_iq = options.acquisition_type is AcquisitionType.INTEGRATION
                buffer_dims = options.results_shape(sweepers)[::-1][int(has_iq):]
                with qua.stream_processing():
                    for acquisition in run.acquisitions.values():
                        acquisition.download(*buffer_dims)

            config = qua_config(self.machine)

            if self.script_file_name is not None:
                script = generate_qua_script(qua_program, config)
                with open(self.script_file_name, "w") as file:
                    file.write(script)

            if self.manager is None:
                warnings.warn(
                    "Not connected to Quantum Machines. Returning program and config.",
                    stacklevel=2,
                )
                return {"program": qua_program, "config": config}

            opened_qm = self.manager.open_qm(config)
            program_id = opened_qm.compile(qua_program)
            pending_job = opened_qm.queue.add_compiled(program_id)
            job = pending_job.wait_for_execution()
            handles = job.result_handles
            handles.wait_for_all_values()
            results |= fetch_results(handles, run.acquisitions.values())
        return results
