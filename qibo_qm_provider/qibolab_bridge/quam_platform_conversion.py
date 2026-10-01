"""Private implementation of the QuAM -> qibolab ``Platform`` topology/
native-gate conversion.

Not re-exported from :mod:`qibo_qm_provider.qibolab_bridge` -- imported only
by :mod:`qibo_qm_provider.qibolab_bridge.platform_from_quam`. Split out from
that module so each conversion concern (topology, native gates, pulse
envelopes) is independently testable against the ``dummy_machine``/
``add_basic_macros_installed`` fixtures in ``test/conftest.py``.

Scope of this module: topology (qubits/couplers) and native gates only --
pulse-envelope conversion lives in ``quam_pulses.py`` (shared with the
qibolab -> QuAM direction), and ``Platform.instruments``/
``parameters.configs`` are built separately, by
:mod:`qibo_qm_provider.qibolab_bridge.quam_wiring` (see its module
docstring), from the machine's OPX1000 MW-FEM/LF-FEM channel/port wiring.
That module docstring also covers what happens when a machine's wiring
isn't supported (Octave/IQ-mixer channels, non-FEM ports): an
instruments-less ``Platform``, with a warning, rather than a guess that
could silently misconfigure real hardware. A ``Platform`` with no
instruments is still fully valid for topology/native-gate inspection and
for ``QiboQMPlatformBackend`` construction; it just cannot
``.execute()``/``.connect()``. When instruments *are* built,
``Platform.channels`` (derived from them) lines up with the channel-id
strings this module writes onto ``Qubit``/coupler entries, via the shared
:mod:`qibo_qm_provider.qibolab_bridge.naming` grammar both modules use.

Native gates are read from QuAM's high-level ``.macros`` dict (``x``,
``sx``, ``measure``, ``cz`` -- installed by ``add_basic_macros`` or
equivalent) first, for robustness to custom gate implementations. This
reuses the exact macro-name tables
:mod:`qibo_qm_provider.qibolab_bridge.naming` defines, so both conversion
directions stay consistent by construction.

When a single-qubit macro is absent, the machine's platform class (see
:mod:`~.quam_platforms`) may supply a fallback pulse from the raw
``.operations`` -- e.g. ``x180``/``x90``/``readout`` for quam_builder's
``FluxTunableQuam``. The generic platform class supplies none, so nothing is
assumed about an unknown machine's pulse names.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

from qibolab._core.native import SingleQubitNatives, TwoQubitNatives
from qibolab._core.parameters import NativeGates
from qibolab._core.pulses.pulse import Acquisition, Readout as QibolabReadout
from qibolab._core.qubits import Qubit, QubitMap

from ..exceptions import AmplitudeOutOfRangeError, UnsupportedEnvelopeError
from .naming import MACRO_NAME_TO_TWO_QUBIT_NATIVE, SINGLE_QUBIT_MACRO_NAMES, channel_id
from .quam_platforms import PulseFetcher, platform_class_for, resolve_single_qubit_pulse
from .quam_pulses import max_voltage_for_channel, quam_envelope_to_qibolab_pulse

if TYPE_CHECKING:
    from quam.core import QuamRoot

__all__: list[str] = []

# Re-exported for backward compatibility with call sites/tests written
# against this module's previous, pre-split location.
_quam_envelope_to_qibolab_pulse = quam_envelope_to_qibolab_pulse


def _build_qubits(machine: "QuamRoot") -> QubitMap:
    """Build a qibolab ``QubitMap`` from ``machine.active_qubit_names``.

    Qubit ids are the QuAM qubit names themselves, used verbatim (e.g.
    ``"q0"``, not ``0``) -- a deliberate deviation from
    ``native_import.py``'s ``str(qubit_id)`` convention, which only exists
    because that module goes the opposite direction (a bare qibolab int id
    -> a QuAM name it has to reconstruct) and has no better option. Going
    QuAM -> qibolab, the QuAM name is already in hand. This means this
    converter's output does not round-trip through
    ``import_qibolab_natives_as_macros``'s int-id convention if both
    directions are ever chained on the same machine -- accepted, not
    expected to be chained in practice.

    Channel ids follow the shared :func:`naming.channel_id` grammar, which
    is also what :func:`qibo_qm_provider.qibolab_bridge.quam_wiring.
    build_qm_wiring` registers into ``Platform.instruments`` -- so once
    instruments are built, ``Platform.channels`` lines up with these ids
    exactly.
    """
    qubits: dict = {}
    for name in machine.active_qubit_names:
        quam_qubit = machine.qubits[name]
        qubits[name] = Qubit(
            drive=channel_id(name, "drive") if getattr(quam_qubit, "xy", None) is not None else None,
            probe=channel_id(name, "probe") if getattr(quam_qubit, "resonator", None) is not None else None,
            acquisition=channel_id(name, "acquisition")
            if getattr(quam_qubit, "resonator", None) is not None
            else None,
            flux=channel_id(name, "flux") if getattr(quam_qubit, "z", None) is not None else None,
        )
    return qubits


def _build_couplers(machine: "QuamRoot") -> QubitMap:
    """Build a qibolab couplers ``QubitMap`` from ``machine.qubit_pairs``.

    Pairs whose ``.coupler`` is ``None`` (matches the ``dummy_machine``
    fixture, and a fully valid real case -- e.g. CZ activated purely by
    flux-tuning one qubit, no separate tunable-coupler element) contribute
    no coupler entry. This is independent of whether that pair has a
    working ``CZ`` native (see ``_build_native_gates``): a coupler-less pair
    can still have a calibrated ``CZ`` gate.
    """
    couplers: dict = {}
    for pair_name, pair in getattr(machine, "qubit_pairs", {}).items():
        if getattr(pair, "coupler", None) is not None:
            couplers[pair_name] = Qubit.coupler(pair_name)
    return couplers


def _single_qubit_natives(quam_qubit, fetch: Optional[PulseFetcher] = None) -> SingleQubitNatives:
    """Build ``SingleQubitNatives`` (``RX``, ``RX90``, ``MZ``) for one QuAM qubit.

    Each native's pulse is resolved by :func:`~.quam_platforms.
    resolve_single_qubit_pulse`: the ``x``/``sx``/``measure`` macro first,
    then ``fetch`` (the machine's platform-class fallback, see
    :mod:`~.quam_platforms`) when the macro is absent. Macros with no pulse
    reference (``VirtualZMacro`` for ``rz``, ``DelayMacro``, ``IdMacro``,
    ``ResetMacro``) have no qibolab field either: virtual-Z is a
    compiler-level frame operation, and ``SingleQubitNatives``' fields are
    ``RX, RX90, RX12, MZ, CP``.

    A pulse that fails to convert (``AmplitudeOutOfRangeError``,
    ``UnsupportedEnvelopeError`` -- see ``quam_pulses.py``) only drops that
    one native field, with a warning; it does not abort the whole qubit (let
    alone the whole platform). This matters in practice: a qubit's ``RX``
    pulse being miscalibrated/out-of-range must not take down its
    otherwise-fine ``RX90`` native too, since the two are independent pulses.
    """
    from qibolab._core.native import Native

    fields: dict = {}
    for native_field in SINGLE_QUBIT_MACRO_NAMES:
        resolved = resolve_single_qubit_pulse(quam_qubit, native_field, fetch)
        if resolved is None:
            continue
        pulse_name, channel, quam_pulse = resolved
        try:
            pulse = quam_envelope_to_qibolab_pulse(quam_pulse, max_voltage_for_channel(channel))
        except (AmplitudeOutOfRangeError, UnsupportedEnvelopeError) as exc:
            warnings.warn(
                f"Qubit {quam_qubit.id!r}: native {native_field!r} (pulse {pulse_name!r}) "
                f"could not be converted to a qibolab Pulse -- "
                f"omitting only this native gate for this qubit. {exc}",
                stacklevel=2,
            )
            continue

        if native_field == "MZ":
            ch_id = channel_id(quam_qubit.id, "acquisition")
            instruction = QibolabReadout(acquisition=Acquisition(duration=pulse.duration), probe=pulse)
        else:
            ch_id = channel_id(quam_qubit.id, "drive")
            instruction = pulse
        fields[native_field] = Native([(ch_id, instruction)])

    return SingleQubitNatives(**fields)


def _two_qubit_natives(pair) -> TwoQubitNatives:
    """Build ``TwoQubitNatives`` (``CZ`` only) for one QuAM qubit pair.

    ``CZGate.flux_pulse_qubit`` is a QuAM reference that resolves directly
    to the actual ``Pulse`` object when read (verified empirically: reading
    ``cz_macro.flux_pulse_qubit`` off an installed ``CZGate`` returns a real
    ``SquarePulse``, not a string) -- unlike single-qubit ``PulseMacro``,
    there is no separate channel/``.operations`` lookup step needed.

    The flux pulse plays on the *moving* qubit's flux channel
    (``pair.moving_qubit``, ``"control"`` or ``"target"`` -- selects which
    of ``pair.qubit_control``/``pair.qubit_target`` actually gets flux-pulsed
    for this gate), not a fixed convention -- this is exactly the
    information qibolab's own generic ``initialize_parameters`` default
    (which arbitrarily uses the pair's first qubit, since it doesn't have
    this information at scaffold time) lacks.
    """
    from qibolab._core.native import Native

    macros = getattr(pair, "macros", None) or {}
    fields: dict = {}
    for macro_name, native_field in MACRO_NAME_TO_TWO_QUBIT_NATIVE.items():
        macro = macros.get(macro_name)
        flux_pulse = getattr(macro, "flux_pulse_qubit", None)
        if flux_pulse is None or isinstance(flux_pulse, str):
            # A still-unresolved string reference (shouldn't happen once
            # attached to a live machine, but guarded rather than assumed).
            continue

        moving_qubit = pair.qubit_control if getattr(pair, "moving_qubit", "control") == "control" else pair.qubit_target
        try:
            pulse = quam_envelope_to_qibolab_pulse(flux_pulse, max_voltage_for_channel(moving_qubit.z))
        except (AmplitudeOutOfRangeError, UnsupportedEnvelopeError) as exc:
            warnings.warn(
                f"Pair {pair.name!r}: native {native_field!r} (macro {macro_name!r}) "
                f"could not be converted to a qibolab Pulse -- omitting only this "
                f"native gate for this pair. {exc}",
                stacklevel=2,
            )
            continue
        ch_id = channel_id(moving_qubit.id, "flux")
        fields[native_field] = Native([(ch_id, pulse)])

    return TwoQubitNatives(**fields)


def _build_native_gates(machine: "QuamRoot") -> NativeGates:
    """Build qibolab ``NativeGates`` (single- and two-qubit) from QuAM
    macros, with single-qubit fallbacks from ``machine``'s platform class
    (:func:`~.quam_platforms.platform_class_for`)."""
    fetch = platform_class_for(machine).fetch_single_qubit_pulse
    single_qubit = {
        name: _single_qubit_natives(machine.qubits[name], fetch) for name in machine.active_qubit_names
    }
    two_qubit = {
        pair_name: _two_qubit_natives(machine.qubit_pairs[pair_name])
        for pair_name in getattr(machine, "active_qubit_pair_names", [])
    }
    return NativeGates(single_qubit=single_qubit, two_qubit=two_qubit)
