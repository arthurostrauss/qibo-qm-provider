"""Real-time sweep support layered on top of :func:`~.qua_macros.
sequence_to_qua_macro`, for the QuAM-macro execution path.

Qibolab's own ``Sweeper.Parameter`` enum splits into two structurally
different kinds (``Parameter.channels()`` in ``qibolab._core.sweeper``),
and this module follows that split rather than fighting it -- see the
Option-A convergence plan (repo root plan discussion) for the full design
rationale:

- **Instruction-scoped** (``amplitude``, ``duration_interpolated``,
  ``relative_phase``/``phase``): a property of *one specific played pulse*
  (``Sweeper.pulses``, identified by that instruction's own ``PulseId``).
  These thread through ``sequence_to_qua_macro``'s existing ``parameters=``
  binding -- unchanged for phase; ``amplitude``/``duration_interpolated``
  additionally need a *dedicated, rescaled reference operation*
  pre-registered before the shot loop (mirroring qibolab's own
  ``register_amplitude_sweeper_pulses``/``register_duration_sweeper_pulses``,
  ``qibolab._core.instruments.qm.controller``), routed in via
  ``operation_overrides=`` -- using the pulse's normally shape-matched
  operation instead would apply the real-time override to the wrong base
  waveform (see ``sequence_to_qua_macro``'s own docstring for why).
- **Channel-scoped** (``frequency``, ``offset``): a property of *the
  channel itself* (``Sweeper.channels``, plain channel-id strings, no pulse
  identity at all) -- QM's own model, not something qibolab invented.
  QuAM already has native methods for both
  (``quam.components.channels.Channel.update_frequency``,
  ``SingleChannel.set_dc_offset`` -- confirmed directly against a real
  Arbel error trace and the installed ``quam`` source, not assumed), so
  these are called *directly* on the resolved QuAM channel, from the loop-
  builder below, entirely outside ``sequence_to_qua_macro`` -- which never
  needs to know these two sweep types exist at all.

Deferred to v2 (not implemented here, see the plan): non-interpolated
``duration`` (needs per-value pre-registered pulses + a QUA ``switch_``/
``case_`` dispatch). ``acquisition_type`` is accepted here only to forward
to :func:`~.qua_macros.sequence_to_qua_macro` -- the actual ``INTEGRATION``/
``DISCRIMINATION`` dispatch lives in ``qua_acquisition``/``qua_macros``, not
in any sweep-loop logic in this module.

Value normalization (``normalize_amplitude``/``normalize_duration``/
``normalize_phase``/``normalize_frequency``, ``sweeper_amplitude``) and the
QUA loop-declaration helper (``from_array``) are reused verbatim from
qibolab -- both are pure value/QUA-DSL transforms with zero dependency on
qibolab's own ``Config``/``Parameters`` classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np
from qibolab._core.execution_parameters import AcquisitionType
from qibolab._core.instruments.qm.program.loops import from_array
from qibolab._core.instruments.qm.program.sweepers import (
    FREQUENCY_BANDWIDTH,
    normalize_amplitude,
    normalize_duration,
    normalize_frequency,
    normalize_phase,
    sweeper_amplitude,
)
from qibolab._core.pulses.pulse import Pulse as QibolabPulse, PulseId
from qibolab._core.pulses.pulse import Readout as QibolabReadout
from qibolab._core.sweeper import Parameter

from . import naming
from .qua_macros import ParameterTarget, sequence_to_qua_macro
from .quam_pulses import max_voltage_for_channel, quam_pulse_from_qibolab_pulse

if TYPE_CHECKING:
    from qibolab._core.sequence import PulseSequence
    from qibolab._core.sweeper import ParallelSweepers, Sweeper
    from quam.core import QuamRoot

__all__ = ["build_sweep_macro", "SUPPORTED_PARAMETERS"]

_PULSE_SCOPED_FIELD = {
    Parameter.amplitude: "amplitude",
    Parameter.duration_interpolated: "duration",
    Parameter.relative_phase: "phase",
    Parameter.phase: "phase",
}
_CHANNEL_SCOPED = {Parameter.frequency, Parameter.offset}
SUPPORTED_PARAMETERS = set(_PULSE_SCOPED_FIELD) | _CHANNEL_SCOPED


def _rescale_target(item) -> Tuple[QibolabPulse, PulseId]:
    """The pulse to rescale, and the id ``sequence_to_qua_macro`` itself
    keys overrides by, for one ``Sweeper.pulses`` entry.

    Mirrors qibolab's own branching in ``controller._update_pulse_amplitude``:
    a bare ``Pulse`` targets itself; a ``Readout`` targets its own probe --
    and the override id for a ``Readout`` must be ``probe.id``, matching
    ``sequence_to_qua_macro``'s own Readout branch, which keys its
    ``amplitude``/``operation_overrides`` lookups by ``instruction.probe.id``,
    not ``instruction.id``.
    """
    if isinstance(item, QibolabReadout):
        return item.probe, item.probe.id
    return item, item.id


def _register_reference_pulse(
    machine: "QuamRoot",
    channel_id: str,
    pulse: QibolabPulse,
    *,
    amplitude: Optional[float] = None,
    duration: Optional[float] = None,
) -> str:
    """Register a new, dedicated QuAM operation for ``pulse`` with
    ``amplitude``/``duration`` overridden, returning the new op name.

    Mirrors qibolab's own ``register_amplitude_sweeper_pulses``/
    ``register_duration_sweeper_pulses`` (interpolated branch) -- a fresh
    operation, not a mutation of whatever op ``pulse``'s own shape would
    otherwise match, so other (un-swept) occurrences of the same shape
    elsewhere in the sequence are unaffected.
    """
    channel = naming.resolve_channel(machine, channel_id)
    max_voltage = max_voltage_for_channel(channel)
    updates = {}
    if amplitude is not None:
        updates["amplitude"] = amplitude
    if duration is not None:
        updates["duration"] = duration
    reference_pulse = pulse.model_copy(update=updates)
    op_name = f"qua_sweep_ref_{abs(hash((pulse.id, tuple(sorted(updates.items()))))):x}"
    channel.operations[op_name] = quam_pulse_from_qibolab_pulse(reference_pulse, op_name, max_voltage)
    return op_name


def _prepare_amplitude_sweep(machine: "QuamRoot", sequence: "PulseSequence", sweeper: "Sweeper") -> Dict[PulseId, str]:
    """Pre-register one rescaled reference pulse per swept pulse instance,
    at ``sweeper_amplitude(values)`` (QM's ``amp()`` scale-factor bound is
    +-1.99, ``qibolab._core.instruments.qm.program.sweepers.
    MAX_AMPLITUDE_FACTOR``) -- so the real-time ``amplitude_scale`` bound
    to it at play time never has to exceed that bound to reach every swept
    value. Returns ``{override_id: op_name}`` for
    ``sequence_to_qua_macro``'s ``operation_overrides``.
    """
    amplitude = sweeper_amplitude(np.asarray(sweeper.values))
    overrides: Dict[PulseId, str] = {}
    for item in sweeper.pulses:
        pulse, key = _rescale_target(item)
        channel_id = sequence.pulse_channels(item.id)[0]
        overrides[key] = _register_reference_pulse(machine, channel_id, pulse, amplitude=amplitude)
    return overrides


def _prepare_duration_interpolated_sweep(
    machine: "QuamRoot", sequence: "PulseSequence", sweeper: "Sweeper"
) -> Dict[PulseId, str]:
    """Pre-register the swept pulse at ``duration=min(sweeper.values)`` --
    QM's dynamic pulse duration can only *stretch*, never compress, an
    uploaded waveform, so the base reference has to already be at least as
    short as every value the sweep will stretch it to. Returns
    ``{override_id: op_name}`` for ``sequence_to_qua_macro``'s
    ``operation_overrides``.
    """
    min_duration = int(min(sweeper.values))
    overrides: Dict[PulseId, str] = {}
    for item in sweeper.pulses:
        pulse, key = _rescale_target(item)
        channel_id = sequence.pulse_channels(item.id)[0]
        overrides[key] = _register_reference_pulse(machine, channel_id, pulse, duration=min_duration)
    return overrides


def _validate_and_get_lo(machine: "QuamRoot", channel_ids, values: np.ndarray) -> int:
    """Mirrors qibolab's own ``find_lo_frequencies``: every channel a single
    ``Sweeper`` sweeps frequency on must share one LO (use parallel
    sweepers instead if not), and the resulting IF must stay within the
    instrument's bandwidth.

    Unlike qibolab's version, no probe -> acquire-element remap is needed:
    ``naming.resolve_channel`` already resolves a qibolab "probe" and
    "acquisition" channel id to the *same* QuAM ``resonator`` object (see
    ``naming.py``'s ``_CHANNEL_SUFFIX_TO_ATTR``, both map to ``"resonator"``)
    -- qibolab's own remap exists only because *its* channel model keeps
    probe/acquisition as two separate objects/QUA elements; QuAM's doesn't.
    """
    los = {naming.resolve_channel(machine, channel_id).upconverter_frequency for channel_id in channel_ids}
    if len(los) > 1:
        raise ValueError(
            "Cannot sweep frequency of channels using different LO with the same "
            "Sweeper object. Please use parallel sweepers instead."
        )
    lo_frequency = los.pop()
    max_if = float(np.max(np.abs(values - lo_frequency)))
    if max_if > FREQUENCY_BANDWIDTH:
        raise ValueError(
            f"Frequency sweep requires an intermediate frequency up to {max_if:g} Hz, "
            f"beyond the instrument bandwidth ({FREQUENCY_BANDWIDTH:g} Hz)."
        )
    return int(lo_frequency)


def _declare_and_normalize(machine: "QuamRoot", sweeper: "Sweeper"):
    """Declare this sweeper's QUA loop variable (``int`` for frequency/
    duration_interpolated, ``fixed`` otherwise -- mirrors qibolab's own
    ``INT_TYPE``) and return ``(variable, normalized_values)``."""
    from qm.qua import declare, fixed

    param = sweeper.parameter
    values = np.asarray(sweeper.values)
    if param in (Parameter.frequency, Parameter.duration_interpolated):
        variable = declare(int)
    else:
        variable = declare(fixed)

    if param is Parameter.amplitude:
        values = normalize_amplitude(values)
    elif param in (Parameter.relative_phase, Parameter.phase):
        values = normalize_phase(values)
    elif param is Parameter.duration_interpolated:
        values = normalize_duration(values)
    elif param is Parameter.frequency:
        lo_frequency = _validate_and_get_lo(machine, sweeper.channels, values)
        values = normalize_frequency(values, lo_frequency)
    # Parameter.offset: no normalization, values pass through unchanged --
    # matches qibolab's own NORMALIZERS (offset has no entry there either).
    return variable, values


def _dispatch_sweeper(machine: "QuamRoot", sweeper: "Sweeper", variable, arg_names: Dict[tuple, str], bound: dict) -> None:
    """Apply one sweeper's per-iteration effect, inside its own loop body.

    Channel-scoped parameters (frequency, offset) emit QUA immediately,
    directly on the resolved QuAM channel -- matching qibolab's own
    ``_frequency``/``_offset`` callbacks, which do the same instead of
    touching any pulse-field scratch state. Pulse-scoped parameters instead
    just record ``variable`` into ``bound``, consumed once, at the leaf,
    when the sequence macro is finally invoked.
    """
    from qm import qua

    param = sweeper.parameter
    if param is Parameter.frequency:
        for channel_id in sweeper.channels:
            naming.resolve_channel(machine, channel_id).update_frequency(variable)
    elif param is Parameter.offset:
        for channel_id in sweeper.channels:
            channel = naming.resolve_channel(machine, channel_id)
            max_offset = max_voltage_for_channel(channel)
            with qua.if_(variable >= max_offset):
                channel.set_dc_offset(max_offset)
            with qua.elif_(variable <= -max_offset):
                channel.set_dc_offset(-max_offset)
            with qua.else_():
                channel.set_dc_offset(variable)
    else:
        for item in sweeper.pulses:
            _, key = _rescale_target(item)
            bound[arg_names[(id(sweeper), key)]] = variable


def _sweep_recursive(
    machine: "QuamRoot",
    sweepers: List["ParallelSweepers"],
    arg_names: Dict[tuple, str],
    macro: Callable,
    bound: dict,
) -> None:
    from qm import qua

    if not sweepers:
        macro(**bound)
        return

    parallel = sweepers[0]
    declared = [_declare_and_normalize(machine, sweeper) for sweeper in parallel]
    variables = [v for v, _ in declared]
    values = [vals for _, vals in declared]

    loop = qua.for_each_(tuple(variables), tuple(values)) if len(parallel) > 1 else qua.for_(*from_array(variables[0], values[0]))
    with loop:
        for sweeper, variable in zip(parallel, variables):
            _dispatch_sweeper(machine, sweeper, variable, arg_names, bound)
        _sweep_recursive(machine, sweepers[1:], arg_names, macro, bound)


def build_sweep_macro(
    machine: "QuamRoot",
    sequence: "PulseSequence",
    sweepers: List["ParallelSweepers"],
    *,
    register_missing: bool = True,
    average: bool = False,
    relaxation_time: Optional[float] = None,
    acquisition_type: AcquisitionType = AcquisitionType.INTEGRATION,
) -> Callable[[], None]:
    """Build a zero-argument QUA macro playing ``sequence`` once per call,
    wrapped in the full nested sweep-loop ``sweepers`` describes -- the
    QuAM-macro-path analogue of qibolab's own ``instructions.sweep()``
    composed with ``play(args)``.

    Must be called inside an existing ``with program():`` block (the
    ``operation_overrides``/reference-pulse pre-registration this performs
    up front are pure Python, but ``sequence_to_qua_macro`` itself already
    requires program scope -- see its own docstring). The returned
    callable's own invocation (once, to emit the loop nest) must also stay
    inside that same scope.

    Args:
        acquisition_type: Forwarded verbatim to
            :func:`~.qua_macros.sequence_to_qua_macro`.

    Raises:
        NotImplementedError: If any sweeper's ``parameter`` is outside
            :data:`SUPPORTED_PARAMETERS` (non-interpolated ``duration`` --
            a v2 follow-up, not silently ignored), or if ``acquisition_type``
            is ``AcquisitionType.RAW`` (raised by ``sequence_to_qua_macro``).

    Returns:
        A zero-arg callable; call once, inside the program, to emit the
        whole nested-loop program. ``.acquisitions`` (same shape as
        ``sequence_to_qua_macro``'s) is available immediately, before the
        callable is ever invoked.
    """
    for parallel in sweepers:
        for sweeper in parallel:
            if sweeper.parameter not in SUPPORTED_PARAMETERS:
                raise NotImplementedError(
                    f"Sweeper parameter {sweeper.parameter!r} is not supported by the QuAM-macro "
                    f"execution path yet (v2 follow-up) -- supported: "
                    f"{sorted(p.name for p in SUPPORTED_PARAMETERS)}."
                )

    operation_overrides: Dict[PulseId, str] = {}
    for parallel in sweepers:
        for sweeper in parallel:
            if sweeper.parameter is Parameter.amplitude:
                operation_overrides.update(_prepare_amplitude_sweep(machine, sequence, sweeper))
            elif sweeper.parameter is Parameter.duration_interpolated:
                operation_overrides.update(_prepare_duration_interpolated_sweep(machine, sequence, sweeper))

    # One stable macro-argument name per (sweeper, pulse) pulse-scoped
    # binding, assigned up front so the same names can be used both to
    # build `parameters=` now and to invoke the macro later, once per
    # iteration, deep inside the loop nest.
    parameters: Dict[str, ParameterTarget] = {}
    arg_names: Dict[tuple, str] = {}
    counter = 0
    for parallel in sweepers:
        for sweeper in parallel:
            field = _PULSE_SCOPED_FIELD.get(sweeper.parameter)
            if field is None:
                continue
            for item in sweeper.pulses:
                _, key = _rescale_target(item)
                name = f"sweep_{counter}"
                counter += 1
                parameters[name] = (key, field)
                arg_names[(id(sweeper), key)] = name

    macro = sequence_to_qua_macro(
        machine,
        sequence,
        parameters=parameters,
        operation_overrides=operation_overrides,
        relaxation_time=relaxation_time,
        register_missing=register_missing,
        average=average,
        acquisition_type=acquisition_type,
    )

    def run() -> None:
        _sweep_recursive(machine, list(sweepers), arg_names, macro, {})

    run.acquisitions = macro.acquisitions
    return run
