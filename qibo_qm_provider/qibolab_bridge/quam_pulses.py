"""Private, bidirectional QuAM <-> qibolab pulse-envelope conversion.

Not re-exported from :mod:`qibo_qm_provider.qibolab_bridge` -- imported by
``quam_platform_conversion.py`` (QuAM -> qibolab, for native-gate/topology
inspection), ``native_import.py`` (qibolab -> QuAM, for the legacy
OpenQASM/QuAM-macro backend), and ``qua_macros.py`` (qibolab -> QuAM, for
the QUA-macro emitter). Split out so both directions of envelope conversion
exist exactly once, with one shared, reconciled amplitude-unit convention.

Amplitude units: QuAM ``Pulse`` amplitudes are volts; qibolab ``Pulse``
amplitudes are dimensionless, normalized to roughly ``[-1, 1]`` (qibolab
re-multiplies by the channel's maximum output voltage at playback time --
see ``qibolab._core.instruments.qm.controller.channel_max_voltage``). Both
conversion directions here go through the same ``max_voltage`` divide/
multiply, so a pulse converted one way and back preserves its real voltage.
"""

from __future__ import annotations

import numpy as np
from qibolab._core.pulses.envelope import Custom, Gaussian, Rectangular
from qibolab._core.pulses.pulse import Pulse as QibolabPulse
from qibolab._core.pulses.pulse import Readout as QibolabReadout
from quam.components.pulses import GaussianPulse, Pulse as QuamPulse, SquarePulse, SquareReadoutPulse, WaveformPulse

from ..exceptions import AmplitudeOutOfRangeError, UnsupportedEnvelopeError

__all__ = [
    "max_voltage_for_channel",
    "max_voltage_for_port",
    "quam_pulse_from_qibolab_pulse",
    "quam_readout_pulse_from_qibolab_readout",
    "quam_envelope_to_qibolab_pulse",
]

# qibolab's own defaults (qibolab._core.instruments.qm.controller):
# MAX_VOLTAGE = 0.5 for "direct" output_mode, MAX_VOLTAGE_AMPLIFIED = 2.5 for
# "amplified". Duplicated here (rather than imported) because this module
# converts amplitudes before any Controller/Config object exists to read
# output_mode off of.
MAX_VOLTAGE_DIRECT = 0.5
MAX_VOLTAGE_AMPLIFIED = 2.5

# qibolab-native sampling convention used throughout this module: pulse
# durations are in ns, and QM's FEM ports run at 1 GSa/s = 1 sample/ns, so
# `Pulse.i(sampling_rate=1.0)` yields exactly `int(duration)` samples.
_SAMPLES_PER_NS = 1.0


def max_voltage_for_port(port) -> float:
    """Maximum output voltage (V) for a QuAM output port.

    Mirrors qibolab's own ``channel_max_voltage``
    (``qibolab._core.instruments.qm.controller``): ``2.5`` V for an LF-FEM/
    OPX+ port with ``output_mode == "amplified"``, ``0.5`` V (QM's "direct"
    mode) otherwise -- including MW-FEM ports (which have no ``output_mode``
    field at all, since amplitude there is set via ``full_scale_power_dbm``
    instead) and the synthetic test fixture's bare ``("con1", n)`` tuples
    (``getattr`` on a tuple safely returns the default).
    """
    return MAX_VOLTAGE_AMPLIFIED if getattr(port, "output_mode", None) == "amplified" else MAX_VOLTAGE_DIRECT


def max_voltage_for_channel(channel) -> float:
    """Maximum output voltage (V) for a QuAM channel, single- or IQ-output.

    Reads ``channel.opx_output`` (``MWChannel``/``SingleChannel``, e.g.
    ``XYDriveMW``, ``FluxLine``) or, if absent, ``channel.opx_output_I``
    (``IQChannel``, e.g. ``XYDriveIQ`` -- the I and Q ports of one channel
    always share the same FEM module and ``output_mode``, so either suffices).
    """
    port = getattr(channel, "opx_output", None)
    if port is None:
        port = getattr(channel, "opx_output_I", None)
    return max_voltage_for_port(port)


def _check_amplitude_in_range(amplitude: float, label: str, max_voltage: float, real_voltage: float) -> None:
    if abs(amplitude) > 1.0:
        raise AmplitudeOutOfRangeError(
            f"{label} has amplitude {real_voltage} V, exceeding the channel's "
            f"maximum output voltage of {max_voltage} V ({abs(amplitude):.3g}x)."
        )


# ---------------------------------------------------------------------------
# QuAM -> qibolab
# ---------------------------------------------------------------------------


def quam_envelope_to_qibolab_pulse(quam_pulse: QuamPulse, max_voltage: float = MAX_VOLTAGE_DIRECT) -> QibolabPulse:
    """Convert a QuAM ``Pulse`` into a qibolab ``Pulse``.

    ``SquarePulse`` -> ``rectangular`` and ``GaussianPulse`` -> ``gaussian``
    are exact, symbolic (parameter-preserving) conversions -- the
    (near-)inverse of :func:`quam_pulse_from_qibolab_pulse`. Every other
    QuAM pulse type (``DragCosinePulse`` included) falls back to a generic,
    lossy-but-correct conversion: QuAM's own ``Pulse.calculate_waveform()``
    (every QuAM pulse implements it) is sampled directly and carried over
    as a qibolab ``Custom`` envelope (``envelope.i_``/``.q_`` raw sample
    arrays) -- rather than raising, since there is no reason execution
    correctness should depend on qibolab having a symbolic envelope class
    matching every QuAM pulse shape; QM only needs the right samples played.

    This does lose the *symbolic* parametrization (e.g. sweeping "sigma" or
    "alpha" isn't meaningful against a baked sample array the way it is
    against a ``Gaussian``/``Drag`` envelope's own fields) -- amplitude
    scaling still works (``Pulse.i() = pulse.amplitude * envelope.i(...)``,
    so the raw samples are stored with ``amplitude=1.0`` and are not
    double-scaled), just not shape-level resweeping. Acceptable: the
    symbolic envelope classes are how a handful of *known* shapes are
    represented, not a requirement placed on every future pulse family.

    ``max_voltage`` converts QuAM's volt-scale amplitude into qibolab's
    dimensionless ``[-1, 1]`` convention (qibolab multiplies
    ``pulse.amplitude`` by the channel's maximum output voltage at playback
    time -- see ``qibolab._core.instruments.qm.controller.
    channel_max_voltage`` -- so passing QuAM's raw volts through unchanged
    would silently halve every pulse's real amplitude on direct-mode
    channels). Callers should pass ``max_voltage_for_channel(channel)``;
    the default of 0.5 V (QM's "direct" output mode) only applies when no
    channel is available (e.g. calling this function directly, as the unit
    tests do).

    Raises:
        UnsupportedEnvelopeError: If ``calculate_waveform()`` itself fails,
            or returns something that can't be coerced into a real-valued
            array of the pulse's own declared length -- a genuine
            conversion failure, not merely an unrecognized envelope class.
        AmplitudeOutOfRangeError: If the QuAM amplitude, once divided by
            ``max_voltage``, would exceed qibolab's ``[-1, 1]`` range --
            raised rather than silently clipped, since clipping would play a
            different pulse than the one calibrated.
    """
    if isinstance(quam_pulse, SquarePulse):
        amplitude = quam_pulse.amplitude / max_voltage
        _check_amplitude_in_range(
            amplitude,
            f"QuAM pulse {getattr(quam_pulse, 'id', None)!r} of type {type(quam_pulse).__name__!r}",
            max_voltage,
            quam_pulse.amplitude,
        )
        return QibolabPulse(
            duration=quam_pulse.length,
            amplitude=amplitude,
            envelope=Rectangular(),
        )
    if isinstance(quam_pulse, GaussianPulse):
        amplitude = quam_pulse.amplitude / max_voltage
        _check_amplitude_in_range(
            amplitude,
            f"QuAM pulse {getattr(quam_pulse, 'id', None)!r} of type {type(quam_pulse).__name__!r}",
            max_voltage,
            quam_pulse.amplitude,
        )
        return QibolabPulse(
            duration=quam_pulse.length,
            amplitude=amplitude,
            envelope=Gaussian(rel_sigma=quam_pulse.sigma / quam_pulse.length),
        )
    return _generic_waveform_pulse(quam_pulse, max_voltage)


def _generic_waveform_pulse(quam_pulse: QuamPulse, max_voltage: float = MAX_VOLTAGE_DIRECT) -> QibolabPulse:
    """Fall back to a qibolab ``Custom`` envelope built from QuAM's own
    ``calculate_waveform()`` samples.

    Handles both return shapes ``calculate_waveform()`` actually produces
    (verified empirically): a full-length ``complex``/``float`` sample
    array (e.g. ``DragCosinePulse``) or a bare scalar for a
    constant-envelope pulse, broadcast to ``quam_pulse.length`` samples.
    Already-amplitude-scaled samples (``calculate_waveform()`` bakes in
    ``quam_pulse.amplitude``) are divided by ``max_voltage`` (same unit
    conversion as :func:`quam_envelope_to_qibolab_pulse`, see its docstring)
    and stored with qibolab ``amplitude=1.0`` to avoid double-scaling.
    """
    try:
        waveform = np.asarray(quam_pulse.calculate_waveform(), dtype=complex)
        if waveform.ndim == 0:
            waveform = np.full(quam_pulse.length, waveform.item())
        if len(waveform) != quam_pulse.length:
            raise ValueError(
                f"calculate_waveform() returned {len(waveform)} samples, "
                f"expected {quam_pulse.length}."
            )
    except Exception as exc:
        raise UnsupportedEnvelopeError(
            f"QuAM pulse {getattr(quam_pulse, 'id', None)!r} of type "
            f"{type(quam_pulse).__name__!r} could not be converted to a qibolab "
            f"Pulse: {exc}"
        ) from exc

    scaled = waveform / max_voltage
    peak = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    if peak > 1.0:
        raise AmplitudeOutOfRangeError(
            f"QuAM pulse {getattr(quam_pulse, 'id', None)!r} of type "
            f"{type(quam_pulse).__name__!r} has a peak sample amplitude of "
            f"{peak * max_voltage:.4g} V, exceeding the channel's maximum output "
            f"voltage of {max_voltage} V ({peak:.3g}x)."
        )

    return QibolabPulse(
        duration=quam_pulse.length,
        amplitude=1.0,
        envelope=Custom(i_=scaled.real, q_=scaled.imag),
    )


# ---------------------------------------------------------------------------
# qibolab -> QuAM
# ---------------------------------------------------------------------------

# Envelope kinds with an exact, symbolic QuAM Pulse equivalent. Everything
# else (drag, gaussian_square, snz, exponential, custom) falls back to a
# sampled QuAM WaveformPulse below -- deliberately not quam's
# DragGaussianPulse/FlatTopGaussianPulse, both of which are marked
# deprecated upstream (raise DeprecationWarning in __post_init__); the
# sampled fallback is execution-correct without depending on deprecated API.
_SYMBOLIC_ENVELOPE_KINDS = {"rectangular", "gaussian"}


def quam_pulse_from_qibolab_pulse(pulse: QibolabPulse, name: str, max_voltage: float = MAX_VOLTAGE_DIRECT) -> QuamPulse:
    """Build a QuAM ``Pulse`` matching one qibolab ``Pulse``'s envelope/
    duration/amplitude.

    ``max_voltage`` converts qibolab's dimensionless ``[-1, 1]`` amplitude
    into QuAM's volt-scale convention -- the inverse of
    :func:`quam_envelope_to_qibolab_pulse`'s divide, reconciled here so a
    pulse converted one way and back preserves its real voltage. Callers
    should pass ``max_voltage_for_channel(channel)`` for the *destination*
    QuAM channel.

    Raises:
        AmplitudeOutOfRangeError: If the resulting QuAM voltage would exceed
            ``max_voltage`` (i.e. the qibolab amplitude was already outside
            ``[-1, 1]``) -- raised rather than silently clipped.
    """
    kind = pulse.envelope.kind
    real_voltage = pulse.amplitude * max_voltage
    _check_amplitude_in_range(
        pulse.amplitude, f"qibolab pulse {name!r} (envelope {kind!r})", max_voltage, real_voltage
    )

    if kind == "rectangular":
        return SquarePulse(length=int(pulse.duration), amplitude=real_voltage, id=name)
    if kind == "gaussian":
        return GaussianPulse(
            length=int(pulse.duration),
            amplitude=real_voltage,
            sigma=pulse.envelope.rel_sigma * pulse.duration,
            id=name,
        )
    return _quam_waveform_pulse_from_qibolab_pulse(pulse, name, max_voltage)


def _quam_waveform_pulse_from_qibolab_pulse(pulse: QibolabPulse, name: str, max_voltage: float) -> WaveformPulse:
    """Universal sampled fallback for any qibolab envelope kind with no
    exact symbolic QuAM equivalent (``drag``, ``gaussian_square``, ``snz``,
    ``exponential``, ``custom``).

    Samples at 1 sample/ns (QM's 1 GSa/s FEM convention -- qibolab durations
    are in ns). ``Pulse.i()``/``.q()`` already bake in ``pulse.amplitude``,
    but on qibolab's dimensionless ``[-1, 1]`` scale -- unlike
    ``SquarePulse``/``GaussianPulse`` (which have a separate ``amplitude``
    field converted explicitly), ``WaveformPulse`` has no such field, so the
    ``max_voltage`` conversion has to happen on the samples themselves here,
    or every sampled-fallback pulse would be re-imported at ``1/max_voltage``
    of its real voltage the next time something reads it as a QuAM pulse.
    """
    i_samples = pulse.i(_SAMPLES_PER_NS) * max_voltage
    q_samples = pulse.q(_SAMPLES_PER_NS) * max_voltage
    waveform_q = q_samples.tolist() if np.any(q_samples) else None
    return WaveformPulse(waveform_I=i_samples.tolist(), waveform_Q=waveform_q, id=name)


def quam_readout_pulse_from_qibolab_readout(
    readout: QibolabReadout, name: str, max_voltage: float = MAX_VOLTAGE_DIRECT
) -> QuamPulse:
    """Build a QuAM readout-capable ``Pulse`` from a qibolab ``Readout``.

    Uses ``SquareReadoutPulse`` (carrying ``threshold``pulse-level fields
    QuAM's acquisition wiring reads) when the probe envelope is
    ``rectangular`` -- the common case for a QM readout pulse -- so an
    imported ``measure`` macro is acquisition-complete rather than needing a
    separate wiring step. Any other probe envelope falls back to the plain
    sampled ``WaveformPulse`` (via :func:`quam_pulse_from_qibolab_pulse`),
    which carries no ``threshold``/integration-weight fields.
    """
    probe = readout.probe
    if probe.envelope.kind == "rectangular":
        real_voltage = probe.amplitude * max_voltage
        _check_amplitude_in_range(probe.amplitude, f"qibolab readout pulse {name!r}", max_voltage, real_voltage)
        return SquareReadoutPulse(length=int(probe.duration), amplitude=real_voltage, id=name)
    return quam_pulse_from_qibolab_pulse(probe, name, max_voltage)
