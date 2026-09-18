"""Thin wrapper around ``qiskit_qm_provider.quam_macros.superconducting.
add_basic_macros``.

Per the "import and install it" decision recorded in the approved plan, this
is not vendored or reimplemented -- it is re-exported directly, with a small
unwrapping step so callers can pass either a bare ``quam.core.QuamRoot`` or a
``qibo_qm_provider.backend.QiboQMBackend`` instance (in addition to the
``qiskit_qm_provider.backend.qm_backend.QMBackend`` the wrapped implementation
already accepts).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from qiskit_qm_provider.quam_macros.superconducting.add_basic_macros import (
    add_basic_macros as _add_basic_macros,
)

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
    dict is empty). This wrapper only adds ``QiboQMBackend`` unwrapping;
    macro names/keys installed are unchanged (``x, sx, sy, sydg, rz, measure,
    reset, delay, id`` on qubits, ``cz`` on pairs) -- gate-name translation
    between Qibo and Qiskit conventions happens once, at circuit-conversion
    time, not here.

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
    # _add_basic_macros itself validates that target is a QuamRoot or QMBackend.
    _add_basic_macros(target, reset_type=reset_type, **reset_macro_kwargs)
