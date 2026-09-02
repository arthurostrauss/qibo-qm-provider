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
equivalent), not the raw per-channel ``.operations`` dict, for robustness
to custom gate implementations. This reuses (inverted) the exact macro-name
tables :mod:`qibo_qm_provider.qibolab_bridge.naming` defines, so both
conversion directions stay consistent by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qibolab._core.native import SingleQubitNatives, TwoQubitNatives
from qibolab._core.parameters import NativeGates
from qibolab._core.pulses.pulse import Acquisition, Readout as QibolabReadout
from qibolab._core.qubits import Qubit, QubitMap

from ..exceptions import MissingQuamAttributeError
from .naming import MACRO_NAME_TO_SINGLE_QUBIT_NATIVE, MACRO_NAME_TO_TWO_QUBIT_NATIVE, channel_id
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


def _single_qubit_natives(quam_qubit) -> SingleQubitNatives:
    """Build ``SingleQubitNatives`` for one QuAM qubit from its ``.macros``.

    Only macros exposing a ``.pulse: str`` reference (``PulseMacro`` for
    ``x``/``sx``, ``MeasureMacro`` for ``measure``) can be converted --
    macros with no pulse reference (``VirtualZMacro`` for ``rz``,
    ``DelayMacro``, ``IdMacro``, ``ResetMacro``) contribute no calibrated
    Native entry, since qibolab's ``SingleQubitNatives`` has no field for
    them either (RZ/virtual-Z is a compiler-level frame operation with no
    stored pulse, matching Qibolab's own convention -- confirmed: qibolab's
    ``SingleQubitNatives`` model fields are ``RX, RX90, RX12, MZ, CP``, none
    of which is "RZ").
    """
    from qibolab._core.native import Native

    macros = getattr(quam_qubit, "macros", None) or {}
    fields: dict = {}
    for macro_name, native_field in MACRO_NAME_TO_SINGLE_QUBIT_NATIVE.items():
        macro = macros.get(macro_name)
        pulse_name = getattr(macro, "pulse", None)
        if pulse_name is None:
            continue

        is_measure = native_field == "MZ"
        channel = quam_qubit.resonator if is_measure else quam_qubit.xy
        if channel is None:
            continue
        if pulse_name not in channel.operations:
            raise MissingQuamAttributeError(
                f"Qubit {quam_qubit.id!r} macro {macro_name!r} references pulse "
                f"{pulse_name!r}, which is not in "
                f"{'resonator' if is_measure else 'xy'}.operations "
                f"({sorted(channel.operations)})."
            )
        quam_pulse = channel.operations[pulse_name]
        pulse = quam_envelope_to_qibolab_pulse(quam_pulse, max_voltage_for_channel(channel))

        if is_measure:
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
        pulse = quam_envelope_to_qibolab_pulse(flux_pulse, max_voltage_for_channel(moving_qubit.z))
        ch_id = channel_id(moving_qubit.id, "flux")
        fields[native_field] = Native([(ch_id, pulse)])

    return TwoQubitNatives(**fields)


def _build_native_gates(machine: "QuamRoot") -> NativeGates:
    """Build qibolab ``NativeGates`` (single- and two-qubit) from QuAM macros."""
    single_qubit = {
        name: _single_qubit_natives(machine.qubits[name]) for name in machine.active_qubit_names
    }
    two_qubit = {
        pair_name: _two_qubit_natives(machine.qubit_pairs[pair_name])
        for pair_name in getattr(machine, "active_qubit_pair_names", [])
    }
    return NativeGates(single_qubit=single_qubit, two_qubit=two_qubit)
