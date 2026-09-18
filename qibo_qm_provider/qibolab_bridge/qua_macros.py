"""Convert a qibolab ``PulseSequence`` into a reusable, embeddable QUA macro.

The qibolab-native analogue of ``qiskit_qm_provider.QMBackend.
schedule_to_qua_macro``: instead of a Qiskit Pulse ``Schedule``, this takes a
qibolab ``PulseSequence`` (already-built or ``Compiler.compile``'d from a
Qibo ``Circuit``) and emits the equivalent sequence of QuAM channel calls
(``play``/``wait``/``align``/``frame_rotation_2pi``/``measure``) instead of
running a whole qibolab ``Platform.execute()`` program -- so the result
composes into the caller's own ``with program():`` block, exactly like
``schedule_to_qua_macro``'s callable does on the Qiskit side.

Instruction mapping is a direct, verified port of qibolab's own QUA emitter
(``qibolab._core.instruments.qm.program.instructions.play``) -- not a
reimplementation from scratch; only the *target* changes, from qibolab's own
generated QUA element names to QuAM's ``Channel`` methods.

Real-time parameters: this is deliberately **pulse-field-level**, matching
the granularity qibolab's own ``Sweeper.Parameter`` enum exposes (amplitude,
duration, phase) -- not circuit-level symbolic gate parameters. See
``qibolab_platform_from_quam_plan.md``'s "Real-time parameterization"
section for why qibolab's native compiler cannot offer the latter (verified:
its rule functions do eager numeric arithmetic on gate parameters, and raise
on a symbolic value).
"""

from __future__ import annotations

from inspect import Parameter as SigParameter, Signature
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

import numpy as np
from qibolab._core.execution_parameters import AcquisitionType
from qibolab._core.pulses.pulse import Align, Delay, Pulse as QibolabPulse, PulseId
from qibolab._core.pulses.pulse import Readout as QibolabReadout
from qibolab._core.pulses.pulse import VirtualZ

from . import naming
from .qua_acquisition import IntegratedAcquisition, ShotsAcquisition
from .quam_pulses import max_voltage_for_channel, quam_envelope_to_qibolab_pulse, quam_pulse_from_qibolab_pulse

if TYPE_CHECKING:
    from qibolab._core.sequence import PulseSequence
    from quam.core import QuamRoot

__all__ = ["build_operation_index", "sequence_to_qua_macro", "ParameterTarget"]

# A parameter binding names one macro-signature argument as a live override
# for one field ("amplitude", "duration", or "phase") of one pulse in the
# sequence, identified by that pulse's PulseId.
ParameterTarget = Tuple[PulseId, str]
_VALID_FIELDS = {"amplitude", "duration", "phase"}


def _freeze(value):
    """Make a pulse-envelope field hashable for use in a shape-key tuple --
    only ``Custom``'s ``i_``/``q_`` (numpy arrays) need this."""
    if isinstance(value, np.ndarray):
        return tuple(float(v) for v in value.tolist())
    return value


def _shape_key(pulse: QibolabPulse) -> tuple:
    """A pulse's identity for QuAM-operation matching, deliberately
    excluding ``amplitude`` -- qibolab's ``native.rotation()`` rescales
    amplitude for an arbitrary gate angle without touching ``envelope``, so
    two pulses of otherwise-identical shape but different rotation angles
    should resolve to the same QuAM operation, played with a different
    ``amplitude_scale``. ``relative_phase``/``chirp``/``id_`` are excluded
    for the same reason: they are per-use, not per-shape.
    """
    envelope_fields = tuple(
        sorted((k, _freeze(v)) for k, v in pulse.envelope.model_dump().items() if k != "kind")
    )
    return (pulse.envelope.kind, pulse.duration, envelope_fields)


def build_operation_index(machine: "QuamRoot") -> Dict[str, Dict[tuple, Tuple[str, QibolabPulse]]]:
    """Index every operation already registered on ``machine``'s channels by
    pulse shape (see :func:`_shape_key`), keyed by qibolab ``ChannelId``.

    Derived lazily from ``machine`` on every call -- there is no cached
    state to keep in sync with e.g. ``QiboQMPlatformBackend.refresh()``.
    An operation that fails to convert (an envelope kind/shape this
    package's QuAM -> qibolab converter cannot represent) is silently
    skipped, not raised -- it simply isn't offered as a reuse candidate.
    """
    index: Dict[str, Dict[tuple, Tuple[str, QibolabPulse]]] = {}
    for channel_id, channel in naming.iter_channels(machine):
        max_voltage = max_voltage_for_channel(channel)
        shapes: Dict[tuple, Tuple[str, QibolabPulse]] = {}
        for op_name, quam_pulse in (getattr(channel, "operations", None) or {}).items():
            try:
                qibolab_pulse = quam_envelope_to_qibolab_pulse(quam_pulse, max_voltage)
            except Exception:  # noqa: BLE001 - not every registered op need be a playable pulse.
                continue
            shapes[_shape_key(qibolab_pulse)] = (op_name, qibolab_pulse)
        index[channel_id] = shapes
    return index


def _resolve_operation(
    machine: "QuamRoot",
    index: Dict[str, Dict[tuple, Tuple[str, QibolabPulse]]],
    channel_id: str,
    pulse: QibolabPulse,
    *,
    register_missing: bool,
) -> Tuple[str, Optional[float]]:
    """Find (or, if allowed, register) a QuAM operation matching ``pulse``'s
    shape on ``channel_id``, returning ``(op_name, amplitude_scale)``.

    ``amplitude_scale`` is ``None`` when the matched/registered operation's
    own amplitude already equals ``pulse.amplitude`` (no play()-time scaling
    needed) -- notably, every freshly *registered* operation is built from
    ``pulse`` itself, so it always returns ``None`` here.
    """
    shapes = index.setdefault(channel_id, {})
    key = _shape_key(pulse)
    if key in shapes:
        op_name, template = shapes[key]
        if template.amplitude in (0, None) or template.amplitude == pulse.amplitude:
            return op_name, None
        return op_name, pulse.amplitude / template.amplitude

    if not register_missing:
        raise ValueError(
            f"No matching QuAM operation on channel {channel_id!r} for pulse shape "
            f"{key!r}, and register_missing=False."
        )
    channel = naming.resolve_channel(machine, channel_id)
    max_voltage = max_voltage_for_channel(channel)
    op_name = f"qua_macro_{abs(hash(pulse)):x}"
    channel.operations[op_name] = quam_pulse_from_qibolab_pulse(pulse, op_name, max_voltage)
    shapes[key] = (op_name, pulse)
    return op_name, None


def sequence_to_qua_macro(
    machine: "QuamRoot",
    sequence: "PulseSequence",
    *,
    parameters: Optional[Dict[str, ParameterTarget]] = None,
    operation_overrides: Optional[Dict[PulseId, str]] = None,
    relaxation_time: Optional[float] = None,
    register_missing: bool = True,
    average: bool = False,
    acquisition_type: AcquisitionType = AcquisitionType.INTEGRATION,
) -> Callable:
    """Convert a qibolab ``PulseSequence`` into a reusable QUA macro.

    Must be called inside an existing ``with program():`` block (matching
    ``qibolab._core.instruments.qm.program.instructions.play``'s own
    contract, which this function's instruction mapping is ported from).
    Unlike before this function also emits QUA immediately, at *build*
    time, not only when the returned callable is invoked: every ``Readout``
    in ``sequence`` is grouped into an :class:`~.qua_acquisition.
    IntegratedAcquisition` by ``(operation, element)`` (mirroring qibolab's
    own multiplexed-readout grouping,
    ``qibolab._core.instruments.qm.controller.QmController.
    register_acquisitions``), and each group's ``.declare()`` (stream/
    variable declaration) runs once, here, before the returned macro is
    ever invoked -- matching qibolab's own ``program()``, which declares
    every acquisition once before its shot loop, never per shot. Building
    the macro without ever invoking it (e.g. ``register_missing=False``
    raising before any ``Readout`` is reached) still requires a program
    scope if ``sequence`` contains a ``Readout`` at all, since the grouping
    pass runs unconditionally, before that raise.

    Args:
        machine: The QuAM root whose channels the sequence plays on.
        sequence: The pulse sequence to convert -- e.g. from
            ``natives.RX.create_sequence()`` or
            ``QiboQMPlatformBackend.circuit_to_qua_macro``'s
            ``self.compiler.compile(circuit, self.platform)`` call.
        parameters: Real-time hooks, ``{arg_name: (pulse_id, field)}`` where
            ``field`` is ``"amplitude"``, ``"duration"``, or ``"phase"``.
            Each key becomes a positional-or-keyword argument of the
            returned callable, bound to a live QUA scalar that overrides
            that field for that one pulse instruction at play time --
            e.g. ``parameters={"amp": (my_pulse.id, "amplitude")}`` lets
            ``qua_macro(declared_qua_var)`` sweep that pulse's amplitude
            without rebuilding the macro. This is pulse-field-level real
            time, not circuit-level symbolic parameters -- see the module
            docstring.
        relaxation_time: Trailing wait, in ns, after the sequence -- ported
            from qibolab's own ``play()``, which only emits it when > 0.
            Defaults to no trailing wait.
        register_missing: When ``True`` (default), a pulse with no matching
            existing QuAM operation is registered as a new one (mutating
            ``machine`` -- picked up by ``machine.generate_config()`` with
            no extra step, the same contract ``register_gate`` already has
            on the legacy backend). When ``False``, raises instead.
        operation_overrides: ``{pulse_id: op_name}``. Forces that one pulse
            instance (identified by its own ``PulseId``, same granularity
            as ``parameters``) to play/measure the named QuAM operation
            directly, bypassing :func:`_resolve_operation`'s shape-matching
            entirely for it. Needed for amplitude/duration_interpolated
            sweeps (see ``qua_sweep.py``): those pre-register a *dedicated*,
            rescaled reference operation for the swept pulse instance
            specifically (mirroring qibolab's own
            ``register_amplitude_sweeper_pulses``/
            ``register_duration_sweeper_pulses``) -- using the pulse's
            normally shape-matched operation instead would apply the real-
            time ``amplitude_scale``/``duration`` override to the *wrong*
            base waveform (e.g. an un-rescaled ``"x180"`` shared with other,
            un-swept occurrences of the same shape elsewhere in the
            sequence), not a bug ``parameters`` alone can fix since it only
            overrides the *field*, not which operation it applies to.
        average: Forwarded to every acquisition group this call creates --
            ``True`` selects ``AveragingMode.CYCLIC`` (averaged across
            shots at ``.download()`` time), ``False`` (default) selects
            ``SINGLESHOT`` (every shot kept), mirroring qibolab's own
            ``create_acquisition``.
        acquisition_type: Selects which acquisition class every ``Readout``
            group in ``sequence`` is built as -- ``INTEGRATION`` (default)
            for :class:`~.qua_acquisition.IntegratedAcquisition`,
            ``DISCRIMINATION`` for :class:`~.qua_acquisition.
            ShotsAcquisition` (``threshold``/``angle`` read straight off
            the QuAM readout ``Pulse`` actually played). ``RAW`` is not yet
            supported here.

    Returns:
        A callable accepting ``parameters``' names as its signature. Once
        built, ``callable.acquisitions`` holds
        ``{(operation, element): IntegratedAcquisition | ShotsAcquisition}``
        -- one entry per distinct operation/element pair played as a
        ``Readout`` in the sequence, already ``.declare()``d. A caller
        drives the full multi-shot lifecycle: this function only calls
        ``.declare()``; ``.download(*dims)``/``.fetch(handles)`` are the
        caller's responsibility, once per program, after the shot loop
        closes.

    Raises:
        NotImplementedError: If ``acquisition_type`` is
            ``AcquisitionType.RAW``.
    """
    from qm import qua

    if acquisition_type is AcquisitionType.RAW:
        raise NotImplementedError(
            "sequence_to_qua_macro does not support AcquisitionType.RAW yet "
            "(v2 follow-up, see the Option-A convergence plan)."
        )

    parameters = dict(parameters or {})
    for name, (_, target_field) in parameters.items():
        if target_field not in _VALID_FIELDS:
            raise ValueError(f"parameters[{name!r}]: field {target_field!r} must be one of {sorted(_VALID_FIELDS)}.")
    operation_overrides = dict(operation_overrides or {})

    index = build_operation_index(machine)
    sig = Signature([SigParameter(name, SigParameter.POSITIONAL_OR_KEYWORD) for name in parameters])

    def resolve_op_name(channel_id: str, pulse: QibolabPulse) -> Tuple[str, Optional[float]]:
        """``_resolve_operation``, except a pulse instance named in
        ``operation_overrides`` skips shape-matching entirely and plays the
        forced operation, with no shape-derived ``amplitude_scale`` (the
        caller is expected to override that too, via ``parameters``, if the
        forced operation's own amplitude isn't already what's wanted)."""
        forced = operation_overrides.get(pulse.id)
        if forced is not None:
            return forced, None
        return _resolve_operation(machine, index, channel_id, pulse, register_missing=register_missing)

    # Group every Readout by (operation, element) and declare each group's
    # acquisition once, now -- not per shot. Resolving op_name here (rather
    # than only inside qua_macro's per-call body below) also means any
    # missing operation this pass needs gets registered now; qua_macro's own
    # later re-resolution of the same (channel_id, pulse) shape is then a
    # pure cache hit, so nothing new gets registered mid-shot-loop.
    def make_acquisition(op_name: str, channel) -> "IntegratedAcquisition | ShotsAcquisition":
        if acquisition_type is AcquisitionType.DISCRIMINATION:
            readout_pulse = (getattr(channel, "operations", None) or {}).get(op_name)
            threshold = getattr(readout_pulse, "threshold", None)
            angle = getattr(readout_pulse, "integration_weights_angle", None) or 0.0
            return ShotsAcquisition(operation=op_name, element=channel.name, average=average, threshold=threshold, angle=angle)
        return IntegratedAcquisition(operation=op_name, element=channel.name, average=average)

    acquisitions: Dict[Tuple[str, str], "IntegratedAcquisition | ShotsAcquisition"] = {}
    for channel_id, instruction in sequence:
        if not isinstance(instruction, QibolabReadout):
            continue
        channel = naming.resolve_channel(machine, channel_id)
        op_name, _ = resolve_op_name(channel_id, instruction.probe)
        group = acquisitions.setdefault((op_name, channel.name), make_acquisition(op_name, channel))
        group.keys.append(instruction.acquisition.id)
    for group in acquisitions.values():
        group.declare()

    def qua_macro(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments

        def override(pulse_id: PulseId, field: str, default):
            for name, (target_id, target_field) in parameters.items():
                if target_id == pulse_id and target_field == field:
                    return values[name]
            return default

        processed_aligns = set()

        for channel_id, instruction in sequence:
            channel = naming.resolve_channel(machine, channel_id)

            if isinstance(instruction, QibolabPulse):
                op_name, amplitude_scale = resolve_op_name(channel_id, instruction)
                amplitude_scale = override(instruction.id, "amplitude", amplitude_scale)
                duration = override(instruction.id, "duration", None)
                phase = override(instruction.id, "phase", instruction.relative_phase or None)
                # `is not None`, not truthiness: `phase` may be a live QUA variable when
                # overridden via `parameters`, and `bool()` on one raises QmQuaException.
                # Mirrors qibolab's own instructions._play, which checks
                # `parameters.phase is not None` for the same reason.
                if phase is not None:
                    channel.frame_rotation_2pi(phase / (2 * np.pi))
                channel.play(op_name, amplitude_scale=amplitude_scale, duration=duration)
                if phase is not None:
                    qua.reset_frame(channel.name)  # mirrors qibolab's own instructions._play

            elif isinstance(instruction, QibolabReadout):
                op_name, amplitude_scale = resolve_op_name(channel_id, instruction.probe)
                amplitude_scale = override(instruction.probe.id, "amplitude", amplitude_scale)
                acquisitions[(op_name, channel.name)].measure(channel, amplitude_scale=amplitude_scale)

            elif isinstance(instruction, Delay):
                cycles = override(instruction.id, "duration", None)
                if cycles is None:
                    cycles = max(int(instruction.duration) // 4, 4)
                channel.wait(cycles)

            elif isinstance(instruction, VirtualZ):
                phase = override(instruction.id, "phase", instruction.phase)
                channel.frame_rotation_2pi(phase / (2 * np.pi))

            elif isinstance(instruction, Align) and instruction.id not in processed_aligns:
                other_ids = [cid for cid in sequence.pulse_channels(instruction.id) if cid != channel_id]
                other_names = [naming.resolve_channel(machine, cid).name for cid in other_ids]
                channel.align(*other_names)
                processed_aligns.add(instruction.id)

        if relaxation_time:
            qua.wait(int(relaxation_time) // 4)

    qua_macro.__signature__ = sig
    qua_macro.__name__ = "sequence_qua_macro"
    qua_macro.acquisitions = acquisitions
    return qua_macro
