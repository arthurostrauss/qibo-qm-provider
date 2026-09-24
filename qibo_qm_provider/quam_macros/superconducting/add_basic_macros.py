"""Thin wrapper around ``qiskit_qm_provider.quam_macros.superconducting.
add_basic_macros``.

Per the "import and install it" decision recorded in the approved plan, this
is not vendored or reimplemented -- it is called directly, with a small
unwrapping step so callers can pass either a bare ``quam.core.QuamRoot`` or a
``qibo_qm_provider.backend.QiboQMBackend`` instance (in addition to the
``qiskit_qm_provider.backend.qm_backend.QMBackend`` the wrapped implementation
already accepts), plus the Qibo-specific ``gpi2`` macro on top.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from qiskit_qm_provider.backend.qm_backend import QMBackend
from qiskit_qm_provider.quam_macros.superconducting.add_basic_macros import (
    add_basic_macros as _add_basic_macros,
)
from quam.core import QuamRoot

from .single_qubit_macros import GPI2Macro, ZMacro

if TYPE_CHECKING:
    from qibo_qm_provider.backend.qibo_qm_backend import QiboQMBackend

__all__ = ["add_basic_macros"]


def add_basic_macros(
    backend,
    reset_type: Literal["active", "thermalize"] = "thermalize",
    **reset_macro_kwargs,
) -> None:
    """Populate a QuAM machine with standard superconducting gate-level macros.

    See ``qiskit_qm_provider.quam_macros.superconducting.add_basic_macros``
    for the full behavior (idempotent, only seeds qubits whose ``macros``
    dict is empty), which installs ``x, sx, sy, sydg, rz, measure, reset,
    delay, id`` on qubits and ``cz`` on pairs. On top of that, this wrapper:

    - accepts a ``QiboQMBackend`` (unwrapped to its ``qiskit_backend``);
    - installs a :class:`~qibo_qm_provider.quam_macros.superconducting.
      single_qubit_macros.GPI2Macro` under ``gpi2`` on every active qubit
      that has ``sx`` and ``rz`` macros but no ``gpi2`` yet (so it also
      tops up machines seeded before this macro existed). ``GPI2`` is one of
      the two single-qubit targets of Qibo's ``Unroller`` (with ``U3``), so
      this is what makes ``GPI2`` show up in ``QiboQMBackend.natives``;
    - likewise installs a :class:`~qibo_qm_provider.quam_macros.
      superconducting.single_qubit_macros.ZMacro` under ``z`` on every
      active qubit that has ``rz`` but no ``z`` -- Qibo's decomposition
      tables emit ``Z`` unconditionally, assuming it is always native.

    Gate-name translation between Qibo and Qiskit conventions happens once,
    at circuit-conversion time, not here.

    Args:
        backend: A ``quam.core.QuamRoot``, a
            ``qiskit_qm_provider.backend.qm_backend.QMBackend``, or a
            ``qibo_qm_provider.backend.QiboQMBackend`` instance.
        reset_type: Reset macro variant, ``"active"`` or ``"thermalize"``.
        **reset_macro_kwargs: Extra keyword arguments forwarded to
            ``ResetMacro``.
    """
    # Local import to avoid a circular import at module load time
    # (qibo_qm_provider.backend imports quam_macros indirectly via tests/usage,
    # not the reverse, but this keeps the dependency direction explicit).
    from qibo_qm_provider.backend.qibo_qm_backend import QiboQMBackend

    target = backend.qiskit_backend if isinstance(backend, QiboQMBackend) else backend
    if not isinstance(target, (QuamRoot, QMBackend)):
        raise ValueError("Backend should be a QuamRoot, QMBackend or QiboQMBackend instance")
    machine = target.machine if isinstance(target, QMBackend) else target
    # Seed the machine itself so the backend's target is refreshed only once,
    # after the gpi2/z macros below are in place too.
    _add_basic_macros(machine, reset_type=reset_type, **reset_macro_kwargs)

    for qubit in machine.active_qubits:
        if "gpi2" not in qubit.macros and {"sx", "rz"} <= set(qubit.macros):
            qubit.macros["gpi2"] = GPI2Macro()
        if "z" not in qubit.macros and "rz" in qubit.macros:
            qubit.macros["z"] = ZMacro()

    if isinstance(target, QMBackend):
        target.update_target()
