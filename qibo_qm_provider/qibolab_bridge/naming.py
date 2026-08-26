"""Single source of truth for the QuAM <-> qibolab naming grammar.

Two independent conventions used to be duplicated across
``native_import.py`` (formerly ``pulse_sequence_import.py``),
``_quam_platform_conversion.py``, and ``_quam_wiring.py``: the macro-name
<-> native-gate-field mapping (``"x"``
<-> ``"RX"``, etc.) and the channel-id grammar (``{qubit}/drive``,
``coupler_{pair}/flux``). Both directions of each are derived from one
table here, so the two conversion directions (QuAM -> qibolab, qibolab ->
QuAM) and the instrument-wiring module cannot drift apart.

Not about platform *names* (``qibo-qm-iqcc-arbel``) -- that grammar is
:mod:`qibo_qm_provider.qibolab_bridge.platform_naming`, a different concern
entirely (folder resolution vs. per-machine channel/native addressing).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Tuple

if TYPE_CHECKING:
    from quam.core import QuamRoot

__all__ = ["channel_id", "resolve_channel", "iter_channels"]

# Single-/two-qubit macro name <-> qibolab native-gate field. Both
# directions (`import_qibolab_natives_as_macros`, `_quam_platform_conversion`)
# derive from these two dicts instead of maintaining independent copies.
SINGLE_QUBIT_MACRO_NAMES = {"RX": "x", "RX90": "sx", "MZ": "measure"}
TWO_QUBIT_MACRO_NAMES = {"CZ": "cz"}
MACRO_NAME_TO_SINGLE_QUBIT_NATIVE = {v: k for k, v in SINGLE_QUBIT_MACRO_NAMES.items()}
MACRO_NAME_TO_TWO_QUBIT_NATIVE = {v: k for k, v in TWO_QUBIT_MACRO_NAMES.items()}

# Channel-role suffix <-> QuAM attribute name, for a qubit and for a pair's
# coupler respectively. Shared by the channel-id builder/resolver below and
# by _quam_wiring.build_qm_wiring (which is what actually registers these
# channel ids into a qibolab Platform).
_CHANNEL_SUFFIX_TO_ATTR = {"drive": "xy", "probe": "resonator", "acquisition": "resonator", "flux": "z"}
_PAIR_CHANNEL_SUFFIX_TO_ATTR = {"flux": "coupler"}
_COUPLER_PREFIX = "coupler_"


def channel_id(owner_name: str, role: str, *, is_pair: bool = False) -> str:
    """Build a qibolab ``ChannelId`` for one QuAM qubit/pair-coupler role.

    Mirrors ``Qubit.default``'s ``{qubit_name}/{channel_type}`` convention
    for a qubit (``channel_id("q0", "drive") == "q0/drive"``), and this
    package's own ``coupler_{pair}/flux`` convention for a pair's coupler
    (``channel_id("q0-q1", "flux", is_pair=True) == "coupler_q0-q1/flux"``).
    """
    if is_pair:
        return f"{_COUPLER_PREFIX}{owner_name}/{role}"
    return f"{owner_name}/{role}"


def resolve_channel(machine: "QuamRoot", channel_id_: str):
    """Resolve a qibolab ``ChannelId`` (as built by :func:`channel_id`) back
    to its QuAM channel object on ``machine``.

    Raises:
        KeyError: If the owning qubit/pair name isn't in
            ``machine.qubits``/``machine.qubit_pairs``.
        ValueError: If ``role`` isn't a recognized channel suffix for the
            owner kind (qubit vs. pair-coupler).
    """
    owner_name, _, role = channel_id_.rpartition("/")
    if owner_name.startswith(_COUPLER_PREFIX):
        pair_name = owner_name[len(_COUPLER_PREFIX) :]
        attr = _PAIR_CHANNEL_SUFFIX_TO_ATTR.get(role)
        if attr is None:
            raise ValueError(f"{channel_id_!r}: {role!r} is not a recognized pair-coupler channel role.")
        return getattr(machine.qubit_pairs[pair_name], attr)

    attr = _CHANNEL_SUFFIX_TO_ATTR.get(role)
    if attr is None:
        raise ValueError(f"{channel_id_!r}: {role!r} is not a recognized qubit channel role.")
    return getattr(machine.qubits[owner_name], attr)


def iter_channels(machine: "QuamRoot") -> Iterator[Tuple[str, object]]:
    """Yield ``(channel_id, quam_channel)`` for every wired channel on
    ``machine``'s active qubits and pairs -- the qibolab-addressable
    channel set, keyed the same way :func:`channel_id` builds ids and
    :mod:`qibo_qm_provider.qibolab_bridge._quam_wiring` registers them into
    ``Platform.instruments``.

    Skips any role whose QuAM attribute is ``None`` (e.g. a pair with no
    coupler), exactly like :func:`resolve_channel`'s callers already assume.
    """
    for name in getattr(machine, "active_qubit_names", []):
        qubit = machine.qubits[name]
        for role, attr in _CHANNEL_SUFFIX_TO_ATTR.items():
            channel = getattr(qubit, attr, None)
            if channel is not None:
                yield channel_id(name, role), channel

    for pair_name in getattr(machine, "active_qubit_pair_names", []):
        pair = machine.qubit_pairs[pair_name]
        for role, attr in _PAIR_CHANNEL_SUFFIX_TO_ATTR.items():
            channel = getattr(pair, attr, None)
            if channel is not None:
                yield channel_id(pair_name, role, is_pair=True), channel
