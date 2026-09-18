"""QuAM-targeted acquisition/result-fetching: declare, measure, download,
fetch -- the multi-shot streaming/averaging machinery
:func:`~.qua_macros.sequence_to_qua_macro`'s bare ``channel.measure()`` call
does not provide on its own.

A direct, verified port of qibolab's own
``qibolab._core.instruments.qm.program.acquisition`` (``IntegratedAcquisition``,
plus ``ShotsAcquisition`` for ``AcquisitionType.DISCRIMINATION`` -- ``RAW`` is
still deferred, see this package's Option-A convergence plan), retargeted at
QuAM's own ``Channel.measure(operation, amplitude_scale=..., qua_vars=(i, q))``
instead of a raw ``qua.measure(...)`` call -- confirmed to accept pre-declared
QUA variables and perform the identical dual-demod arithmetic qibolab's own
classes use, so only the *declare/save/download/fetch* steps around that call
needed adding, not the measurement itself.

``collect``/``split`` (shape math) and ``assign_variables_to_element`` (the
QM-recommended workaround forcing acquisition variables onto the right
element thread) are unmodified in qibolab -- imported directly from
``qibolab._core.instruments.qm.program.acquisition`` rather than
reimplemented, since a copy would just be a second place to keep in sync with
the original.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

import numpy as np
from qibolab._core.instruments.qm.program.acquisition import (
    _collect as collect,
    _split as split,
    assign_variables_to_element,
)
from qibolab._core.pulses.pulse import PulseId

if TYPE_CHECKING:
    from qm.qua._dsl import _ResultSource, _Variable

__all__ = [
    "IntegratedAcquisition",
    "ShotsAcquisition",
    "collect",
    "split",
    "assign_variables_to_element",
    "fetch_results",
]


@dataclass
class _AcquisitionBase:
    """Shared ``(operation, element)`` identity/bookkeeping for one readout
    group -- mirrors qibolab's own ``Acquisition`` base class, which the
    same two properties are lifted from verbatim.
    """

    operation: str
    element: str
    average: bool = False
    keys: List[PulseId] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Identifier used to save/fetch this group's streams.

        ``"/"`` is replaced the same way qibolab's own ``Acquisition.name``
        does (QUA/OPX1000 stream-processing ``save`` rejects it).
        """
        return f"{self.operation}_{self.element}".replace("/", "|")

    @property
    def npulses(self) -> int:
        return len(self.keys)


@dataclass
class IntegratedAcquisition(_AcquisitionBase):
    """One physical ``(operation, element)`` readout group's declare/
    measure/download/fetch state, for ``AcquisitionType.INTEGRATION`` --
    see :class:`~.qua_macros.AcquisitionGroup`'s docstring (superseded by
    this class) for why grouping by ``(operation, element)`` rather than by
    individual ``Readout`` matters. ``average`` selects
    ``AveragingMode.CYCLIC`` (``True``) vs. ``SINGLESHOT`` (``False``),
    mirroring qibolab's own ``create_acquisition``.
    """

    i: "_Variable" = None
    q: "_Variable" = None
    istream: "_ResultSource" = None
    qstream: "_ResultSource" = None

    def declare(self) -> None:
        """Declare this group's QUA variables/streams. Call once, before
        the shot loop -- never per shot."""
        from qm.qua import declare, declare_stream, fixed

        self.i = declare(fixed)
        self.q = declare(fixed)
        self.istream = declare_stream()
        self.qstream = declare_stream()
        assign_variables_to_element(self.element, self.i, self.q)

    def measure(self, channel, amplitude_scale=None) -> None:
        """Play ``self.operation`` on ``channel`` (a QuAM ``Channel``
        object), reusing this group's pre-declared ``(i, q)`` variables,
        and save each into this group's streams. Call once per shot (per
        ``Readout`` occurrence) inside the shot loop.
        """
        from qm import qua

        channel.measure(self.operation, amplitude_scale=amplitude_scale, qua_vars=(self.i, self.q))
        qua.save(self.i, self.istream)
        qua.save(self.q, self.qstream)

    def download(self, *dimensions: int) -> None:
        """Inside ``with qua.stream_processing():``, buffer/average and
        save this group's streams. Call once, after the shot loop closes.

        ``dimensions`` are the sweep-dimension buffer sizes (e.g. from
        ``ExecutionParameters.results_shape(sweepers)``, reversed) --
        callers should compute these the same way qibolab's own
        ``program()`` does, not assume they transfer unchanged.
        """
        istream = self.istream
        qstream = self.qstream
        if self.npulses > 1:
            istream = istream.buffer(self.npulses)
            qstream = qstream.buffer(self.npulses)
        for dim in dimensions:
            istream = istream.buffer(dim)
            qstream = qstream.buffer(dim)
        if self.average:
            istream = istream.average()
            qstream = qstream.average()
        istream.save(f"{self.name}_I")
        qstream.save(f"{self.name}_Q")

    def fetch(self, handles) -> List[Any]:
        """Pull this group's downloaded streams back and split them into
        one array per readout occurrence (``self.keys`` order)."""
        ires = handles.get(f"{self.name}_I").fetch_all()
        qres = handles.get(f"{self.name}_Q").fetch_all()
        signal = collect(ires, qres, self.npulses)
        return split(signal, self.npulses)


@dataclass
class ShotsAcquisition(_AcquisitionBase):
    """One physical ``(operation, element)`` readout group's declare/
    measure/download/fetch state, for ``AcquisitionType.DISCRIMINATION``.

    ``threshold``/``angle`` are read straight off the QuAM readout
    ``Pulse`` actually played (its ``threshold``/``integration_weights_angle``
    fields -- the same ones ``quam_wiring._readout_pulse``'s
    threshold/iq_angle extraction already reads for the legacy, now-mostly-
    superseded ``QmAcquisitionConfig``), not sourced from a qibolab
    ``Config`` object: QuAM already carries this calibration data on the
    channel itself, so there is nothing to plumb through a separate
    parameter path here.

    The classification arithmetic (``i*cos(angle) - q*sin(angle) >
    threshold``) is a direct, verbatim port of qibolab's own
    ``ShotsAcquisition.measure()`` -- see the module docstring for why only
    the ``measure()``/``declare()`` calls it wraps are retargeted, not the
    arithmetic itself.
    """

    threshold: Optional[float] = None
    angle: float = 0.0
    i: "_Variable" = None
    q: "_Variable" = None
    shot: "_Variable" = None
    shots: "_ResultSource" = None

    def __post_init__(self) -> None:
        if self.threshold is None:
            raise ValueError(
                f"ShotsAcquisition for {self.operation!r} on {self.element!r} has no "
                "threshold -- the QuAM readout pulse actually played must set "
                "`threshold` for AcquisitionType.DISCRIMINATION (see MeasureMacro's "
                "own `discriminated` mode in quam_builder)."
            )
        self.cos = np.cos(self.angle)
        self.sin = np.sin(self.angle)

    def declare(self) -> None:
        from qm.qua import declare, declare_stream, fixed

        self.i = declare(fixed)
        self.q = declare(fixed)
        self.shot = declare(int)
        self.shots = declare_stream()
        assign_variables_to_element(self.element, self.i, self.q, self.shot)

    def measure(self, channel, amplitude_scale=None) -> None:
        """Play ``self.operation`` on ``channel``, then classify the shot
        against ``self.threshold`` in the rotated IQ plane. Call once per
        shot (per ``Readout`` occurrence) inside the shot loop.
        """
        from qm import qua

        channel.measure(self.operation, amplitude_scale=amplitude_scale, qua_vars=(self.i, self.q))
        qua.assign(
            self.shot,
            qua.Cast.to_int(self.i * self.cos - self.q * self.sin > self.threshold),
        )
        qua.save(self.shot, self.shots)

    def download(self, *dimensions: int) -> None:
        """Inside ``with qua.stream_processing():``, buffer/average and
        save this group's shot stream. Call once, after the shot loop
        closes. ``dimensions`` -- see ``IntegratedAcquisition.download``.
        """
        shots = self.shots
        if self.npulses > 1:
            shots = shots.buffer(self.npulses)
        for dim in dimensions:
            shots = shots.buffer(dim)
        if self.average:
            shots = shots.average()
        shots.save(f"{self.name}_shots")

    def fetch(self, handles) -> List[Any]:
        """Pull this group's downloaded shot stream back and split it into
        one array per readout occurrence (``self.keys`` order)."""
        shots = handles.get(f"{self.name}_shots").fetch_all()
        return split(shots, self.npulses)


def fetch_results(
    handles, acquisitions: Iterable[Union[IntegratedAcquisition, ShotsAcquisition]]
) -> Dict[PulseId, Any]:
    """Turn ``{acquisition: fetched arrays}`` into ``{PulseId: Result}`` --
    a package-local port of qibolab's own ``QmController``-module
    ``fetch_results`` free function (``controller.py:75-94``), operating on
    this module's acquisition objects instead of qibolab's ``Acquisition``.
    """
    from collections import defaultdict

    results: Dict[PulseId, list] = defaultdict(list)
    for acquisition in acquisitions:
        data = acquisition.fetch(handles)
        for key, value in zip(acquisition.keys, data):
            results[key].append(value)

    # collapse single element lists for back-compatibility, same as qibolab's own.
    return {key: value[0] if len(value) == 1 else value for key, value in results.items()}
