"""The single named call site for "what config does QM hardware actually
receive" -- ``machine.generate_config()``, QuAM's own, always-correct
generator, never a qibolab-authored ``Configuration`` object.

Both :class:`~.quam_controller.QuamQmController` and
:class:`~.iqcc_controller.IQCCQmController` call this instead of building
their own config (see ``quam_controller``'s module docstring for why) --
naming the call once here, rather than inlining ``machine.generate_config()``
in both places, is what makes that "config authority" property visible and
grep-able as a single fact about this package, not an implementation detail
duplicated across two controllers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from quam.core import QuamRoot

__all__ = ["qua_config"]


def qua_config(machine: "QuamRoot") -> Dict[str, Any]:
    """The QM config to send to hardware for ``machine``, right now."""
    return machine.generate_config()
