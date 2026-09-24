"""Single-qubit QuAM macros for Qibo native gates that ``qiskit_qm_provider``'s
``add_basic_macros`` does not install itself.

``GPI2`` matters because it is one of the two single-qubit targets Qibo's
``Unroller``/``NativeGates`` decomposes onto (the other being ``U3``): with a
``gpi2`` macro installed, ``QiboQMBackend.natives`` gains ``GPI2`` and the
default transpiler can lower arbitrary single-qubit gates onto the machine.

``Z`` is needed alongside it because Qibo's decomposition tables assume ``Z``
is always native (``NativeGates.default()`` includes it) and emit it
regardless of the requested native set -- e.g. ``gpi2_dec``:
``H -> [Z, GPI2(pi/2)]``, ``X -> [GPI2(pi/2), GPI2(pi/2), Z]``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from quam.components.macro import QubitMacro
from quam.core import quam_dataclass

__all__ = ["GPI2Macro", "ZMacro"]


@quam_dataclass
class GPI2Macro(QubitMacro):
    r"""Qibo's ``GPI2`` gate, built from the qubit's existing ``sx`` and ``rz``
    macros.

    Qibo defines

    .. math::
        \mathrm{GPI2}(\phi) = \frac{1}{\sqrt2}\begin{pmatrix}
        1 & -i e^{-i\phi} \\ -i e^{i\phi} & 1 \end{pmatrix},

    i.e. a :math:`\pi/2` rotation about the equatorial axis
    :math:`(\cos\phi, \sin\phi, 0)`. Conjugating an :math:`X`-rotation by a
    :math:`Z`-rotation moves its axis by that angle, so

    .. math::
        \mathrm{GPI2}(\phi) = R_Z(\phi)\, R_X(\pi/2)\, R_Z(-\phi)
        \simeq R_Z(\phi)\, \sqrt{X}\, R_Z(-\phi),

    equal up to the global phase :math:`e^{-i\pi/4}` between ``SX`` and
    :math:`R_X(\pi/2)`. In time order: ``rz(-phi)``, ``sx``, ``rz(phi)``. With
    the virtual-Z ``rz`` macro that ``add_basic_macros`` installs, both ``rz``
    calls are frame updates, so the gate costs a single ``x90`` pulse.

    ``apply`` delegates to the sibling macros' own ``apply`` rather than
    playing pulses directly, so any recalibration of ``sx``/``rz`` carries over
    to ``gpi2`` automatically.

    Args:
        sx_macro: Name of the qubit macro implementing ``SX``.
        rz_macro: Name of the qubit macro implementing ``RZ(angle)``.
    """

    sx_macro: str = "sx"
    rz_macro: str = "rz"

    def apply(self, phi, **kwargs) -> None:
        macros = self.qubit.macros
        rz, sx = macros[self.rz_macro], macros[self.sx_macro]
        rz.apply(-phi)
        sx.apply()
        rz.apply(phi)

    @property
    def inferred_duration(self) -> Optional[float]:
        macros = self.qubit.macros
        durations = [macros[name].duration for name in (self.rz_macro, self.sx_macro, self.rz_macro)]
        if any(duration is None for duration in durations):
            return None
        return sum(durations)


@quam_dataclass
class ZMacro(QubitMacro):
    r"""Qibo's ``Z`` gate as ``RZ(pi)`` via the qubit's existing ``rz`` macro.

    :math:`Z = e^{i\pi/2} R_Z(\pi)`, so this is exact up to a global phase;
    with the virtual-Z ``rz`` macro it is a single frame update.

    Args:
        rz_macro: Name of the qubit macro implementing ``RZ(angle)``.
    """

    rz_macro: str = "rz"

    def apply(self, **kwargs) -> None:
        self.qubit.macros[self.rz_macro].apply(np.pi)

    @property
    def inferred_duration(self) -> Optional[float]:
        return self.qubit.macros[self.rz_macro].duration
