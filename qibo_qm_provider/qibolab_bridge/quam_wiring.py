"""Private implementation of the QuAM -> qibolab instrument-wiring conversion.

Not re-exported from :mod:`qibo_qm_provider.qibolab_bridge` -- imported only
by :mod:`qibo_qm_provider.qibolab_bridge.platform_from_quam`, mirroring
:mod:`qibo_qm_provider.qibolab_bridge.quam_platform_conversion`'s convention
of splitting each conversion concern into its own independently-testable,
private module.

Scope: builds a qibolab :class:`~qibolab._core.instruments.qm.QmController`'s
``channels``/``fems`` plus the matching ``Platform.parameters.configs`` from a
QuAM object's channel/port graph, so ``Platform.channels`` (derived entirely
from ``Platform.instruments`` -- see
``qibolab._core.platform.platform.Platform.channels``) is populated and
Qibocal protocols, which address channels as ``Sweeper.channels`` targets, can
actually run.

Only OPX1000 **MW-FEM** (drive/probe/acquisition) and **LF-FEM** (flux)
wiring is supported -- this matches every real IQCC machine inspected while
building this converter (all MW-FEM + LF-FEM, no Octave). Octave/IQ-mixer
channels (``quam.components.channels.IQChannel``), OPX+ ports, and 2 GSa/s
ports raise :class:`~qibo_qm_provider.exceptions.UnsupportedWiringError`
rather than being silently mis-wired: a ``Platform`` that misconfigures real
hardware is worse than one that refuses to build. Callers (see
``platform_from_quam.quam_to_qibolab_platform``) catch this and fall back to
an instruments-less, inspection-only ``Platform``, preserving this package's
previous behavior for machines this module cannot wire.

Flux predistortion filters: QuAM's raw FIR ``feedforward_filter`` taps
**are** transferred, as a qibolab ``FiniteImpulseResponseFilter`` -- verified
against qibolab's own ``OpxOutputConfig.filter("LF")``, which for the
OPX1000 cluster type emits *both* a ``"feedforward"`` key (from every
``FiniteImpulseResponseFilter``/``ExponentialFilter.feedforward`` in
``filters``, convolved together) and an ``"exponential"`` key (from
``ExponentialFilter`` terms alone) -- so carrying QuAM's FIR taps as a
``FiniteImpulseResponseFilter`` *and* its exponential terms as
``ExponentialFilter`` reproduces both QM config keys without double-counting
either (the two keys are independent QM firmware inputs, not alternatives).
Port ``delay``/``crosstalk`` still have no qibolab field at all and are
dropped, with a warning.

One conversion is deliberately lossy, and warns rather than silently
dropping data:

- **Flux operating point.** QuAM emits a static ``0.0`` config offset and
  biases the qubit to its flux point (``joint``/``independent``/``min``/
  ``arbitrary``) at *runtime* in QUA (``set_all_fluxes``/``set_dc_offset``).
  A qibolab ``Platform`` has no such runtime prologue, so this converter
  writes the flux-point offset **statically** into ``OpxOutputConfig.offset``
  instead -- correct for qibolab's execution model, but a deliberate
  inversion of what QuAM's own ``generate_config()`` would emit, and it
  means the platform's idle point is fixed at conversion time and will not
  track a later ``machine.set_all_fluxes(...)`` call without rebuilding the
  platform.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from qibolab._core.components import AcquisitionChannel, Channel, Config, DcChannel, IqChannel
from qibolab._core.components.configs import IqConfig
from qibolab._core.components.filters import ExponentialFilter, FiniteImpulseResponseFilter
from qibolab._core.instruments.qm.components import MwFemOscillatorConfig, OpxOutputConfig, QmAcquisitionConfig
from quam.components.channels import MWChannel, SingleChannel

from ..exceptions import UnsupportedWiringError
from .naming import channel_id

if TYPE_CHECKING:
    from quam.core import QuamRoot

__all__: list[str] = []

# QuAM's own FEM-only sampling rate; quam_builder's pulse classes ignore
# `sampling_rate` when generating waveform samples (verified empirically), so
# a 2 GSa/s port would silently produce half-length waveforms -- rejected
# outright rather than guessed at.
_SUPPORTED_SAMPLING_RATE = 1e9

# FluxLine.flux_point -> the attribute holding that operating point's offset
# (all in volts). "zero" has no attribute at all -- it means 0.0 by
# definition.
_FLUX_POINT_OFFSET_ATTR = {
    "joint": "joint_offset",
    "independent": "independent_offset",
    "min": "min_offset",
    "arbitrary": "arbitrary_offset",
    "zero": None,
}


def _device(port, owner_label: str) -> str:
    """qibolab ``Channel.device`` for a QuAM FEM port: ``"{con}/{fem}"``.

    Only ports with a ``fem_id`` (OPX1000 MW-FEM/LF-FEM) are supported --
    OPX+ ports (``OPXPlusAnalogOutputPort``) and the synthetic test fixture's
    bare ``("con1", n)`` tuples have no ``fem_id`` and raise instead of being
    guessed at.
    """
    if not hasattr(port, "fem_id"):
        raise UnsupportedWiringError(
            f"{owner_label}: port {port!r} has no `fem_id` -- only OPX1000 FEM-wired "
            "QuAM ports (MW-FEM/LF-FEM) are supported by this converter; OPX+ ports "
            "and bare-tuple wiring are not."
        )
    return f"{port.controller_id}/{port.fem_id}"


def _path(port) -> str:
    """qibolab ``Channel.path`` for a QuAM port: the port number as a decimal
    string (``Channel.port`` does ``int(self.path)``)."""
    return str(port.port_id)


def _lo_id(port) -> str:
    """Config id for the oscillator/LO config of one physical MW-FEM output
    port, keyed by ``(controller, fem, port)`` rather than by qubit.

    This is what makes multiplexed readout (many qubits' resonators sharing
    one physical MW-FEM output/input port -- confirmed on real IQCC hardware,
    up to 6 qubits per port) share one LO config id structurally, rather than
    by incidentally-equal values: qibolab's ``MwFemOutput.update`` asserts
    matching ``band``/``sampling_rate``/``power`` across channels registered
    against the same physical port, so two different config ids for the same
    port would be a correctness bug waiting to happen, not just redundant.
    """
    return f"{port.controller_id}/{port.fem_id}/{port.port_id}/lo"


def _check_sampling_rate(port, owner_label: str) -> None:
    rate = getattr(port, "sampling_rate", _SUPPORTED_SAMPLING_RATE)
    if rate != _SUPPORTED_SAMPLING_RATE:
        raise UnsupportedWiringError(
            f"{owner_label}: port {port!r} is wired at {rate / 1e9:g} GSa/s -- only "
            f"{_SUPPORTED_SAMPLING_RATE / 1e9:g} GSa/s ports are supported (quam_builder's "
            "pulse classes ignore `sampling_rate` when generating waveform samples, so "
            "a faster port would silently produce half-length waveforms)."
        )


def _readout_pulse(qubit):
    """The QuAM readout ``Pulse`` used for shot threshold/rotation info,
    resolved via the ``measure`` macro (mirrors
    ``quam_platform_conversion._single_qubit_natives``).

    Returns ``None`` if no ``measure`` macro is installed, so acquisition
    channel *wiring* still succeeds without native-gate readout calibration
    data -- wiring and native-gate installation are independent concerns.
    """
    macro = (getattr(qubit, "macros", None) or {}).get("measure")
    pulse_name = getattr(macro, "pulse", None)
    if pulse_name is None:
        return None
    return qubit.resonator.operations.get(pulse_name)


def _flux_offset(flux_line) -> float:
    """The static DC offset to write for one flux line, chosen by its
    ``flux_point`` -- see the module docstring for why this is a deliberate
    inversion of what QuAM's own ``generate_config()`` emits."""
    flux_point = getattr(flux_line, "flux_point", "independent")
    attr = _FLUX_POINT_OFFSET_ATTR.get(flux_point)
    if attr is None:
        return 0.0
    return getattr(flux_line, attr, 0.0) or 0.0


def _wire_drive(qubit, channels: dict, configs: dict, fems: dict) -> None:
    xy = qubit.xy
    if not isinstance(xy, MWChannel):
        raise UnsupportedWiringError(
            f"Qubit {qubit.id!r}: drive channel {type(xy).__name__!r} is not an MW-FEM "
            "channel (quam.components.channels.MWChannel) -- Octave/IQ-mixer drive "
            "wiring is not supported by this converter."
        )
    port = xy.opx_output
    _check_sampling_rate(port, f"qubit {qubit.id!r} drive")

    device, path = _device(port, f"qubit {qubit.id!r} drive"), _path(port)
    lo_id = _lo_id(port)
    drive_id = channel_id(qubit.id, "drive")

    channels[drive_id] = IqChannel(device=device, path=path, lo=lo_id, mixer=None)
    configs.setdefault(
        lo_id,
        MwFemOscillatorConfig(
            frequency=xy.upconverter_frequency,
            power=port.full_scale_power_dbm,
            upconverter=xy.upconverter,
            band=port.band,
            sampling_rate=port.sampling_rate,
        ),
    )
    configs[drive_id] = IqConfig(frequency=xy.upconverter_frequency + xy.intermediate_frequency)
    fems[device] = "MW"


def _wire_probe_and_acquisition(qubit, channels: dict, configs: dict, fems: dict) -> None:
    resonator = qubit.resonator
    if not isinstance(resonator, MWChannel):
        raise UnsupportedWiringError(
            f"Qubit {qubit.id!r}: resonator channel {type(resonator).__name__!r} is not "
            "an MW-FEM channel (quam.components.channels.MWChannel) -- Octave/IQ-mixer "
            "readout wiring is not supported by this converter."
        )

    out_port, in_port = resonator.opx_output, resonator.opx_input
    _check_sampling_rate(out_port, f"qubit {qubit.id!r} probe")
    _check_sampling_rate(in_port, f"qubit {qubit.id!r} acquisition")

    probe_id, acq_id = channel_id(qubit.id, "probe"), channel_id(qubit.id, "acquisition")

    out_device, out_path = _device(out_port, f"qubit {qubit.id!r} probe"), _path(out_port)
    lo_id = _lo_id(out_port)
    channels[probe_id] = IqChannel(device=out_device, path=out_path, lo=lo_id, mixer=None)
    configs.setdefault(
        lo_id,
        MwFemOscillatorConfig(
            frequency=resonator.upconverter_frequency,
            power=out_port.full_scale_power_dbm,
            upconverter=resonator.upconverter,
            band=out_port.band,
            sampling_rate=out_port.sampling_rate,
        ),
    )
    configs[probe_id] = IqConfig(frequency=resonator.upconverter_frequency + resonator.intermediate_frequency)
    fems[out_device] = "MW"

    in_device, in_path = _device(in_port, f"qubit {qubit.id!r} acquisition"), _path(in_port)
    channels[acq_id] = AcquisitionChannel(device=in_device, path=in_path, probe=probe_id)

    ro = _readout_pulse(qubit)
    threshold = getattr(ro, "threshold", None) if ro is not None else None
    iq_angle = getattr(ro, "integration_weights_angle", None) if ro is not None else None
    if ro is not None:
        weights = getattr(ro, "integration_weights", None)
        if weights not in (None, [(1, ro.length)]):
            warnings.warn(
                f"Qubit {qubit.id!r}: readout pulse {getattr(ro, 'id', None)!r} has "
                f"non-constant integration_weights ({weights!r}) -- this converter only "
                "carries over threshold/iq_angle (a constant-weight `kernel=None` is "
                "used regardless), so shot discrimination will use uniform integration "
                "weights instead of the calibrated, non-constant ones.",
                stacklevel=2,
            )
    configs[acq_id] = QmAcquisitionConfig(
        delay=int(getattr(resonator, "time_of_flight", 140)),
        smearing=int(getattr(resonator, "smearing", 0) or 0),
        threshold=threshold,
        iq_angle=iq_angle,
        kernel=None,
    )
    fems[in_device] = "MW"


def _wire_flux(ch_id: str, flux_line, channels: dict, configs: dict, fems: dict, owner_label: str) -> None:
    if not isinstance(flux_line, SingleChannel):
        raise UnsupportedWiringError(
            f"{owner_label}: flux channel {type(flux_line).__name__!r} is not a single-output "
            "channel (quam.components.channels.SingleChannel) -- unsupported wiring shape."
        )
    port = flux_line.opx_output
    _check_sampling_rate(port, owner_label)

    device, path = _device(port, owner_label), _path(port)
    channels[ch_id] = DcChannel(device=device, path=path)

    delay, crosstalk = getattr(port, "delay", 0), getattr(port, "crosstalk", None)
    if delay or crosstalk:
        warnings.warn(
            f"{owner_label}: QuAM port delay={delay!r}/crosstalk={crosstalk!r} has no "
            "qibolab OpxOutputConfig equivalent and is dropped -- this can affect CZ timing "
            "alignment relative to QuAM's calibration.",
            stacklevel=2,
        )

    feedforward_taps = getattr(port, "feedforward_filter", None) or []
    exponential_terms = getattr(port, "exponential_filter", None) or []
    filters = []
    if feedforward_taps:
        filters.append(FiniteImpulseResponseFilter(coefficients=list(feedforward_taps)))
    filters += [ExponentialFilter(amplitude=a, tau=t) for a, t in exponential_terms]

    if feedforward_taps and exponential_terms:
        warnings.warn(
            f"{owner_label}: QuAM has both feedforward_filter taps and exponential_filter "
            "terms -- qibolab's OpxOutputConfig.filter() convolves every filter's "
            "`.feedforward` together for the 'feedforward' config key (including "
            "ExponentialFilter's own FIR approximation), while also emitting those same "
            "exponential terms natively under the 'exponential' key for the OPX1000 cluster "
            "type. This may double-apply the exponential correction on real hardware -- "
            "verify against machine.generate_config() before trusting flux pulse shape here.",
            stacklevel=2,
        )

    configs[ch_id] = OpxOutputConfig(
        offset=_flux_offset(flux_line),
        output_mode=getattr(port, "output_mode", "direct"),
        sampling_rate=port.sampling_rate,
        upsampling_mode=getattr(port, "upsampling_mode", "mw"),
        filters=filters,
    )
    fems[device] = "LF"


def build_qm_wiring(machine: "QuamRoot") -> tuple[dict[str, Channel], dict[str, Config], dict[str, str]]:
    """Build ``channels``, ``configs``, and ``fems`` for a
    :class:`~qibolab._core.instruments.qm.QmController` from a QuAM object's
    channel/port graph.

    Channel ids match :func:`qibo_qm_provider.qibolab_bridge.
    quam_platform_conversion._build_qubits`/``_build_couplers`` exactly
    (``{qubit}/drive``, ``{qubit}/probe``, ``{qubit}/acquisition``,
    ``{qubit}/flux``, ``coupler_{pair}/flux``), so ``Platform.channels``
    (derived from these instruments) lines up with ``Platform.qubits``/
    ``Platform.couplers`` (derived independently from QuAM's qubit/pair
    graph).

    Raises:
        UnsupportedWiringError: If any qubit/pair uses wiring this converter
            doesn't support (Octave/IQ-mixer channels, non-FEM ports, or a
            sampling rate other than 1 GSa/s). Raised for the whole machine
            rather than partially wiring it, so a caller never receives a
            ``Platform`` that silently omits some qubits' channels.
    """
    channels: dict[str, Channel] = {}
    configs: dict[str, Config] = {}
    fems: dict[str, str] = {}

    for name in machine.active_qubit_names:
        qubit = machine.qubits[name]
        if getattr(qubit, "xy", None) is not None:
            _wire_drive(qubit, channels, configs, fems)
        if getattr(qubit, "resonator", None) is not None:
            _wire_probe_and_acquisition(qubit, channels, configs, fems)
        if getattr(qubit, "z", None) is not None:
            _wire_flux(channel_id(name, "flux"), qubit.z, channels, configs, fems, owner_label=f"qubit {name!r} flux")

    for pair_name in getattr(machine, "active_qubit_pair_names", []):
        pair = machine.qubit_pairs[pair_name]
        coupler = getattr(pair, "coupler", None)
        if coupler is not None:
            _wire_flux(
                channel_id(pair_name, "flux", is_pair=True),
                coupler,
                channels,
                configs,
                fems,
                owner_label=f"pair {pair_name!r} coupler",
            )

    return channels, configs, fems
