"""Parse ``qibo-qm-provider`` platform folder/entrypoint names.

A Qibolab platform "name" is resolved by ``qibolab.create_platform`` as a
literal folder name on ``$QIBOLAB_PLATFORMS`` -- there is no registry, and
``create()`` (the function each folder's ``platform.py`` defines) is called
with zero arguments. So distinguishing "load a local QuAM state" from
"fetch a named machine from IQCC" has to be encoded in the name itself,
not in a call argument.

Grammar, parsed by fixed-prefix stripping (never positional hyphen-split,
since an IQCC backend name may itself contain hyphens)::

    qibo-qm-local                  -> LocalSource()
    qibo-qm-iqcc-<backend_name>    -> IqccSource(backend_name=<backend_name>)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from ..exceptions import InvalidPlatformName

__all__ = ["LocalSource", "IqccSource", "parse_platform_name", "PREFIX", "LOCAL_NAME", "IQCC_PREFIX"]

PREFIX = "qibo-qm-"
LOCAL_NAME = "local"
IQCC_PREFIX = "iqcc-"


@dataclass(frozen=True)
class LocalSource:
    """Marker for a ``qibo-qm-local`` platform name."""


@dataclass(frozen=True)
class IqccSource:
    """A ``qibo-qm-iqcc-<backend_name>`` platform name."""

    backend_name: str


def parse_platform_name(name: str) -> Union[LocalSource, IqccSource]:
    """Parse a ``qibo-qm-provider`` platform name into its source kind.

    Args:
        name: A platform name/folder name, e.g. ``"qibo-qm-local"`` or
            ``"qibo-qm-iqcc-arbel"``.

    Returns:
        ``LocalSource()`` or ``IqccSource(backend_name=...)``.

    Raises:
        InvalidPlatformName: If ``name`` doesn't match the
            ``qibo-qm-{local,iqcc-<backend_name>}`` grammar.
    """
    if not name.startswith(PREFIX):
        raise InvalidPlatformName(
            f"{name!r} is not a qibo-qm-provider platform name "
            f"(expected {PREFIX!r}{LOCAL_NAME} or {PREFIX}{IQCC_PREFIX}<iqcc-backend-name>)."
        )
    rest = name.removeprefix(PREFIX)
    if rest == LOCAL_NAME:
        return LocalSource()
    if rest.startswith(IQCC_PREFIX):
        backend_name = rest.removeprefix(IQCC_PREFIX)
        if not backend_name:
            raise InvalidPlatformName(f"{name!r} is missing an IQCC backend name.")
        return IqccSource(backend_name=backend_name)
    raise InvalidPlatformName(
        f"{name!r}: unrecognized suffix {rest!r} "
        f"(expected {LOCAL_NAME!r} or {IQCC_PREFIX!r} + a backend name)."
    )
