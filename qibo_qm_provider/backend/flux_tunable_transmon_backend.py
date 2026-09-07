"""``FluxTunableTransmonBackend``: wraps ``qiskit_qm_provider.backend.
flux_tunable_transmon_backend.FluxTunableTransmonBackend`` instead of the
plain ``QMBackend``.

Because Qibo circuits carry no Qiskit-Pulse-style channel abstraction, there
is no ``channel_mapping``/``DriveChannel``/``ControlChannel`` wiring to build
here -- the wrapped backend already builds it from ``qubit.xy/z/resonator``
and ``qubit_pair.coupler``, and ``execute_circuit``/circuit conversion/
measurement translation are inherited unchanged from ``QiboQMBackend``.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

from qiskit_qm_provider.backend.flux_tunable_transmon_backend import (
    FluxTunableTransmonBackend as QiskitFluxTunableTransmonBackend,
)
from quam.core import QuamRoot

from .qibo_qm_backend import QiboQMBackend

try:
    from qm import QuantumMachinesManager
except ImportError:  # pragma: no cover
    QuantumMachinesManager = None  # type: ignore[assignment,misc]

__all__ = ["FluxTunableTransmonBackend"]


class FluxTunableTransmonBackend(QiboQMBackend):
    """Qibo backend for flux-tunable transmon QuAM machines."""

    def __init__(
        self,
        machine: QuamRoot,
        qmm: Optional["QuantumMachinesManager"] = None,
        name: Optional[str] = None,
    ):
        # Bypass QiboQMBackend.__init__ (which builds a plain QMBackend) and
        # construct the topology-aware wrapped backend directly.
        from qibo.backends import NumpyBackend

        NumpyBackend.__init__(self)
        self.name = name or "qibo-qm-provider"
        self._qiskit_backend = QiskitFluxTunableTransmonBackend(machine, qmm=qmm, name=name)

    def get_qubit_channels(self, qubit: Union[int, str]) -> Dict[str, object]:
        """Read-only view of a qubit's physical QuAM channels.

        Qibo circuits have no channel-level abstraction of their own, so this
        is provided purely for introspection/debugging -- it does not
        participate in ``execute_circuit``.
        """
        q = self._qiskit_backend.get_qubit(qubit)
        return {"xy": q.xy, "z": q.z, "resonator": q.resonator}

    def get_pair_coupler(self, qubits: Tuple[Union[int, str], Union[int, str]]):
        """Read-only view of a qubit pair's coupler channel, or ``None``."""
        pair = self._qiskit_backend.get_qubit_pair(qubits)
        return getattr(pair, "coupler", None)
