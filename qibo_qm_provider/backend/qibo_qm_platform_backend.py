"""``QiboQMPlatformBackend``: a Qibo backend wrapping a qibolab ``Platform``
built directly from QuAM.

This is a second, separate execution path alongside ``QiboQMBackend``: it
subclasses ``qibolab._core.backends.QibolabBackend`` and, apart from a
default gate-decomposition step ``execute_circuit`` adds (see its
docstring), inherits execution unchanged (qibolab's own
``Compiler.compile`` + ``platform.execute`` -- no OpenQASM round-trip,
unlike ``QiboQMBackend``). ``execute_circuits`` is inherited as-is, with no
equivalent decomposition step. See ``qibolab_platform_from_quam_plan.md`` and
``qibo_backend_vs_qibolab_platform.md`` (repo root) for the full design
rationale; in particular, this exists to reuse Qibocal's existing
calibration protocols (written purely against ``platform.execute``/
``platform.natives``), not to replace ``QiboQMBackend``'s QUA-native macro
path (kept for operations like real-time-randomized RB where a
``Platform``/``PulseSequence`` IR would lose the thing worth having).

No ``register_gate``/``update_target`` equivalent exists here in this pass
(deliberately -- see ``qibolab_platform_from_quam_plan.md`` §4.3): those
methods on ``QiboQMBackend`` are irreducibly shaped around holding a live
``qiskit_qm_provider.QMBackend`` (``component.macros[name] = macro``,
``update_target()``), and a plain ``qibolab.Platform`` has no ``.macros``/
``.get_qubit`` at all. Instead, this class keeps a reference to the QuAM
object it was built from (``self.machine``, when known) and exposes
``refresh()``: mutate ``self.machine`` directly, then call ``refresh()`` to
fully re-derive ``self.platform`` from it. A gate-level, incrementally
resyncing equivalent to ``register_gate`` is a documented stretch goal, not
built now.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from qibo.models import Circuit
from qibo.result import MeasurementOutcomes
from qibo.transpiler.unroller import NativeGates
from qibolab import Platform
from qibolab._core.backends import QibolabBackend
from qibolab._core.sequence import PulseSequence
from quam.core import QuamRoot

from ..qibolab_bridge.platform_from_quam import (
    _create_iqcc_with_machine,
    _create_local_with_machine,
    quam_to_qibolab_platform,
)
from ..qibolab_bridge.qua_macros import ParameterTarget, sequence_to_qua_macro
from .default_transpile import default_transpile

__all__ = ["QiboQMPlatformBackend"]


class QiboQMPlatformBackend(QibolabBackend):
    """A ``QibolabBackend`` whose ``Platform`` is built from QuAM.

    Construction accepts anything ``QibolabBackend`` itself accepts (a
    registered platform name string, or an already-built ``Platform``
    object) -- this is what lets both
    ``qibo.set_backend("qibo_qm_provider", platform="qibo-qm-iqcc-arbel")``
    (via the ``qibo_qm_provider.MetaBackend`` entrypoint) and plain
    ``qibo.set_backend("qibolab", platform="qibo-qm-iqcc-arbel")`` (a bare
    ``QibolabBackend``, no dependency on this class at all) resolve to the
    same underlying ``Platform``, built by the same converter.
    """

    def __init__(self, platform: Union[str, Platform], machine: Optional[QuamRoot] = None):
        """Wrap ``platform`` (a name string or an already-built ``Platform``).

        Args:
            platform: Forwarded to ``QibolabBackend.__init__`` unchanged --
                a registered platform name (resolved via
                ``qibolab.create_platform``) or a ``Platform`` object.
            machine: The source QuAM object, if known. Only ever set by
                ``from_iqcc``/``from_local``/``from_machine`` (the only
                paths that hold a live ``QuamRoot`` before conversion) --
                left ``None`` when constructed via a bare name/object
                through ``create_platform``, since that call chain has no
                way to carry the source machine back out (a hard
                constraint of qibolab's own zero-kwarg ``create()``
                convention, not a gap specific to this class).
        """
        super().__init__(platform)
        self.machine = machine

    def execute_circuit(
        self,
        circuit: Circuit,
        initial_state: Optional[Circuit] = None,
        nshots: int = 1000,
        transpile: bool = True,
    ) -> MeasurementOutcomes:
        """Execute ``circuit`` on ``self.platform``.

        Identical to ``QibolabBackend.execute_circuit`` (inherited
        unchanged, see class docstring) except for ``transpile``: qibolab's
        own ``Compiler.compile`` never checks a gate against the platform's
        native gates and decomposes it if not (confirmed against issue #4)
        -- it expects a circuit that "respects the platform's ... native
        gates" already. This override adds that check as a default,
        opt-out-able step, via :func:`~qibo_qm_provider.backend.
        default_transpile.default_transpile`, using :attr:`natives`
        (``QibolabBackend.natives``, inherited unchanged) as both what
        counts as already-native and what a non-native gate may be
        decomposed into. ``circuit.wire_names`` is preserved untouched --
        ``self.compiler.compile`` reads it directly for physical qubit
        placement, and this step never re-places qubits, only rewrites
        gates.

        Args:
            circuit: A Qibo circuit.
            initial_state: A Qibo circuit to prepend, or ``None``.
            nshots: Number of shots to sample.
            transpile: When ``True`` (default), decompose any gate not
                already native to ``self.platform`` before compiling. A
                decomposition emits a ``UserWarning`` naming the gates
                involved. When ``False``, skip that step only -- there is
                no pre-check that the circuit is already native; a
                non-native gate still fails later inside qibolab's
                compiler / execute path. Note ``execute_circuits`` never
                applies this step.
        """
        if isinstance(initial_state, Circuit):
            circuit = initial_state + circuit
            initial_state = None
        if transpile:
            circuit = default_transpile(
                circuit,
                already_native=self.natives,
                decomposition_targets=[
                    name for name in self.natives if name in NativeGates.__members__ and name != "NONE"
                ],
            )
        return super().execute_circuit(circuit, initial_state=initial_state, nshots=nshots)

    @classmethod
    def from_iqcc(
        cls,
        backend_name: str,
        *,
        state_path: Optional[str] = None,
        quam_class: Optional[str] = None,
        api_token: Optional[str] = None,
    ) -> "QiboQMPlatformBackend":
        """Fetch ``backend_name``'s latest state from IQCC and build a backend.

        No ``$QIBOLAB_PLATFORMS`` folder registration required -- this is
        the native-Python entrypoint, going through the same
        ``_create_iqcc_with_machine`` a registered
        ``qibo-qm-iqcc-<backend_name>`` folder's ``platform.py`` would call.
        """
        platform, machine = _create_iqcc_with_machine(
            f"qibo-qm-iqcc-{backend_name}",
            state_path=state_path,
            quam_class=quam_class,
            api_token=api_token,
        )
        return cls(platform=platform, machine=machine)

    @classmethod
    def from_local(
        cls,
        state_path: str,
        *,
        quam_class: Optional[str] = None,
    ) -> "QiboQMPlatformBackend":
        """Load an already-existing local QuAM state and build a backend.

        No ``$QIBOLAB_PLATFORMS`` folder registration required -- the
        native-Python entrypoint counterpart to ``from_iqcc``.
        """
        platform, machine = _create_local_with_machine(state_path=state_path, quam_class=quam_class)
        return cls(platform=platform, machine=machine)

    @classmethod
    def from_machine(cls, machine: QuamRoot, name: str = "qibo-qm-local") -> "QiboQMPlatformBackend":
        """Wrap an already-loaded/live QuAM object directly.

        For a machine already obtained some other way (built in-memory, or
        fetched via ``qiskit_qm_provider.IQCCProvider`` directly) -- no
        fetch/load step, just conversion.
        """
        return cls(platform=quam_to_qibolab_platform(machine, name=name), machine=machine)

    def refresh(self) -> None:
        """Fully re-derive ``self.platform`` from ``self.machine``.

        Call after mutating ``self.machine`` directly (e.g.
        ``backend.machine.qubits["q0"].xy.operations["x180"] = ...``) to
        pick up the change -- this is a full re-conversion, not an
        incremental resync (see class docstring for why).

        Note this discards any live ``QmController.manager`` connection on
        ``self.platform.instruments["qm"]`` -- ``quam_to_qibolab_platform``
        builds a fresh, disconnected ``QmController`` every time, since it
        has no way to carry a live connection across a full re-conversion.
        Call ``self.platform.connect()`` again after ``refresh()`` if the
        platform was previously connected.

        Raises:
            RuntimeError: If this backend wasn't built via
                ``from_iqcc``/``from_local``/``from_machine`` (``self.machine``
                is ``None``) -- there is no QuAM object to re-derive from.
        """
        if self.machine is None:
            raise RuntimeError(
                "refresh() requires a backend built via from_iqcc/from_local/"
                "from_machine (self.machine is None); backends built from a "
                "bare platform name/object have no QuAM object to re-derive from."
            )
        self.platform = quam_to_qibolab_platform(self.machine, name=self.platform.name)

    def _require_machine(self, caller_name: str) -> QuamRoot:
        if self.machine is None:
            raise RuntimeError(
                f"{caller_name} requires a backend built via from_iqcc/from_local/"
                "from_machine (self.machine is None); backends built from a bare "
                "platform name/object have no QuAM object to emit QUA against."
            )
        return self.machine

    def sequence_to_qua_macro(
        self,
        sequence: PulseSequence,
        *,
        parameters: Optional[dict[str, ParameterTarget]] = None,
        relaxation_time: Optional[float] = None,
        register_missing: bool = True,
    ) -> Callable:
        """Convert a qibolab ``PulseSequence`` into a reusable, embeddable
        QUA macro, playing on ``self.machine``'s QuAM channels.

        The qibolab-native analogue of
        ``qiskit_qm_provider.QMBackend.schedule_to_qua_macro`` -- see
        ``qibo_qm_provider.qibolab_bridge.qua_macros`` for the full
        contract (instruction mapping, real-time ``parameters``, the
        returned callable's ``.acquisitions``). Must be called inside an
        existing ``with qm.qua.program():`` block.

        Raises:
            RuntimeError: If this backend has no ``self.machine`` (see
                ``refresh()``'s docstring for why that can happen).
        """
        machine = self._require_machine("sequence_to_qua_macro")
        return sequence_to_qua_macro(
            machine,
            sequence,
            parameters=parameters,
            relaxation_time=relaxation_time,
            register_missing=register_missing,
        )

    def circuit_to_qua_macro(
        self,
        circuit: Circuit,
        *,
        parameters: Optional[dict[str, ParameterTarget]] = None,
        relaxation_time: Optional[float] = None,
        register_missing: bool = True,
    ) -> Callable:
        """Compile ``circuit`` to a pulse sequence via the already-inherited
        ``self.compiler.compile(circuit, self.platform)`` (the same step
        ``execute_circuit`` uses internally), then convert it into a
        reusable QUA macro via :meth:`sequence_to_qua_macro`.

        Unlike ``execute_circuit``, this path does **not** run the default
        gate-decomposition step -- pass an already-native circuit (or
        decompose yourself) before calling.

        The qibolab-native analogue of
        ``qiskit_qm_provider.QMBackend.quantum_circuit_to_qua`` -- unlike
        that method, ``circuit`` must have concrete (already-bound) gate
        parameters: qibolab's compiler rules do eager numeric arithmetic on
        them and cannot accept a symbolic value (verified; see
        ``qibolab_platform_from_quam_plan.md``'s "Real-time parameterization"
        section). Real-time parameterization at the pulse-field level is
        still available via ``parameters``, keyed by the *compiled* pulses'
        ids -- inspect ``measurement_map``/the returned macro to find them,
        or build the sequence directly via ``self.compiler.compile`` and
        call :meth:`sequence_to_qua_macro` yourself when the pulse ids are
        needed before compiling.

        Returns:
            The macro callable, with a ``.measurement_map`` attribute
            (``{gates.M: PulseSequence}``, as returned by
            ``self.compiler.compile``) attached so measurement outcomes stay
            traceable to their circuit-level ``gates.M``, in addition to the
            ``.acquisitions`` attribute :func:`sequence_to_qua_macro`
            already attaches.

        Raises:
            RuntimeError: If this backend has no ``self.machine``.
        """
        machine = self._require_machine("circuit_to_qua_macro")
        sequence, measurement_map = self.compiler.compile(circuit, self.platform)
        macro = sequence_to_qua_macro(
            machine,
            sequence,
            parameters=parameters,
            relaxation_time=relaxation_time,
            register_missing=register_missing,
        )
        macro.measurement_map = measurement_map
        return macro
