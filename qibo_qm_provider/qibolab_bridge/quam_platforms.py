"""qibolab ``Platform`` subclasses per QuAM machine type, and the registry
that picks one for a given machine.

The QuAM -> qibolab conversion is generic by default: native gates come
from the qubit's ``.macros`` (``x``, ``sx``, ``measure``), which makes no
assumption about how the underlying QuAM names its pulses. That is
:class:`QuamPlatform`.

A machine type whose pulse naming convention is known can add a fallback
for when a macro is absent, by subclassing :class:`QuamPlatform` and
overriding :meth:`QuamPlatform.fetch_single_qubit_pulse`.
:class:`FluxTunableQuamPlatform` does this for quam_builder's
``FluxTunableQuam`` (``xy`` ``x180``/``x90``, resonator ``readout``).

:data:`QUAM_PLATFORM_CLASSES` maps QuAM machine classes to platform
classes; :func:`platform_class_for` resolves a machine through its class
hierarchy, so subclasses of a registered machine type inherit its platform
class. ``quam_to_qibolab_platform`` uses it, so callers get the matching
platform without choosing one. To support a new machine type, subclass
:class:`QuamPlatform` and add an entry to :data:`QUAM_PLATFORM_CLASSES`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type

from qibolab import Platform
from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import FluxTunableQuam

from ..exceptions import MissingQuamAttributeError
from .naming import SINGLE_QUBIT_MACRO_NAMES

if TYPE_CHECKING:
    from quam.components.pulses import Pulse as QuamPulse
    from quam.core import QuamRoot

__all__ = [
    "FluxTunableQuamPlatform",
    "QUAM_PLATFORM_CLASSES",
    "QuamPlatform",
    "platform_class_for",
    "resolve_single_qubit_pulse",
]

PulseFetcher = Callable[[object, str], Optional[Tuple[str, "QuamPulse"]]]
"""``(quam_qubit, native_field) -> (operation_name, pulse) | None``."""

# QuAM channel attribute each single-qubit native plays on.
_NATIVE_CHANNEL_ATTR = {"RX": "xy", "RX90": "xy", "MZ": "resonator"}


class QuamPlatform(Platform):
    """A qibolab ``Platform`` built from a QuAM machine.

    Generic: single-qubit natives come from the qubit's macros only, since
    nothing is assumed about the machine's pulse names.
    """

    @staticmethod
    def fetch_single_qubit_pulse(quam_qubit, native: str) -> Optional[Tuple[str, "QuamPulse"]]:
        """Fallback pulse for ``native`` (``"RX"``, ``"RX90"`` or ``"MZ"``)
        when the qubit has no corresponding macro.

        Returns ``(operation_name, pulse)``, or ``None`` for no fallback. The
        pulse must live on the native's channel (``xy`` for ``RX``/``RX90``,
        ``resonator`` for ``MZ``), otherwise it is ignored. Override in a
        subclass to encode a machine type's naming convention.
        """
        return None


# quam_builder transmon operation names (see quam_builder's default
# transmon pulses).
_FLUX_TUNABLE_PULSE_NAMES = {"RX": "x180", "RX90": "x90", "MZ": "readout"}


class FluxTunableQuamPlatform(QuamPlatform):
    """Platform for quam_builder's ``FluxTunableQuam``.

    Macros still take priority; without them, natives fall back to
    quam_builder's operation names: ``x180`` (``RX``) and ``x90`` (``RX90``)
    on ``xy``, ``readout`` (``MZ``) on the resonator.
    """

    @staticmethod
    def fetch_single_qubit_pulse(quam_qubit, native: str) -> Optional[Tuple[str, "QuamPulse"]]:
        pulse_name = _FLUX_TUNABLE_PULSE_NAMES.get(native)
        if pulse_name is None:
            return None
        try:
            return pulse_name, quam_qubit.get_pulse(pulse_name)
        except ValueError:  # not found, or not unique across the qubit's channels
            return None


QUAM_PLATFORM_CLASSES: Dict[type, Type[QuamPlatform]] = {
    FluxTunableQuam: FluxTunableQuamPlatform,
}
"""QuAM machine class -> platform class. Extend to support new machine types."""


def platform_class_for(machine: "QuamRoot") -> Type[QuamPlatform]:
    """The platform class registered for ``machine``'s type, or for its
    closest registered base class; :class:`QuamPlatform` if none is."""
    for cls in type(machine).__mro__:
        if cls in QUAM_PLATFORM_CLASSES:
            return QUAM_PLATFORM_CLASSES[cls]
    return QuamPlatform


def resolve_single_qubit_pulse(
    quam_qubit, native: str, fetch: Optional[PulseFetcher] = None
) -> Optional[Tuple[str, object, "QuamPulse"]]:
    """Resolve the QuAM pulse behind single-qubit ``native`` as
    ``(operation_name, channel, pulse)``.

    Shared by native-gate export and acquisition wiring (``MZ``'s threshold/
    ``iq_angle``), so both always agree on the pulse.

    1. The native's macro (``x``/``sx``/``measure``), when it has a
       ``.pulse`` reference.
    2. Otherwise ``fetch(quam_qubit, native)`` (a platform class's
       :meth:`~QuamPlatform.fetch_single_qubit_pulse`), kept only if the
       pulse lives on the native's channel.

    Returns ``None`` if the qubit lacks the channel or neither source yields
    a pulse.

    Raises:
        MissingQuamAttributeError: If the macro references a pulse that is
            not in the channel's ``operations``.
    """
    channel_attr = _NATIVE_CHANNEL_ATTR[native]
    channel = getattr(quam_qubit, channel_attr, None)
    if channel is None:
        return None

    macro_name = SINGLE_QUBIT_MACRO_NAMES[native]
    macro = (getattr(quam_qubit, "macros", None) or {}).get(macro_name)
    pulse_name = getattr(macro, "pulse", None)
    if pulse_name is not None:
        if pulse_name not in channel.operations:
            raise MissingQuamAttributeError(
                f"Qubit {quam_qubit.id!r} macro {macro_name!r} references pulse "
                f"{pulse_name!r}, which is not in {channel_attr}.operations "
                f"({sorted(channel.operations)})."
            )
        return pulse_name, channel, channel.operations[pulse_name]

    fetched = fetch(quam_qubit, native) if fetch is not None else None
    if fetched is None:
        return None
    pulse_name, pulse = fetched
    if getattr(pulse, "channel", None) is not channel:
        return None
    return pulse_name, channel, pulse
