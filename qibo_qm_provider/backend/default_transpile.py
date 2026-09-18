"""Minimal, opt-out-able default transpile step for ``execute_circuit``.

Confirmed against #3/#4: nothing between building a Qibo circuit and handing
it to either backend's compiler ever checks a gate against what the target
machine can actually run and decomposes it if not -- bug #3 (a plain ``H``
reaching the OpenQASM3 exporter as a name with no registered macro) is the
concrete case. :func:`default_transpile` adds that step: every gate not
already native to the target is rewritten, via
``qibo.transpiler.unroller.translate_gate``, into gates from the machine's
own :class:`~qibo.transpiler.unroller.NativeGates` vocabulary.

This is deliberately *not* a single ``Unroller(...)`` call over the whole
circuit. Qibo's own decomposition tables raise a bare ``KeyError`` for
several gates this package treats as first-class machine natives (``GPI``,
``PRX``, ``U1q``, ``MS`` -- verified directly against qibo 0.3.3), because
``NativeGates`` is a small, fixed nine-member enum that has no member for
them at all. Running ``Unroller`` unconditionally over every gate would
crash on exactly the ion-trap-style circuits this package exists to
support, or -- for a gate that already *is* one of the nine members, e.g. a
circuit already written in ``GPI2``/``CZ`` -- silently rebuild it through
Qibo's generic decomposition instead of leaving it alone. So a gate already
in ``already_native`` (the target's actual, per-machine operation set, wider
than the ``NativeGates`` enum) is always left untouched, and only a
genuinely non-native gate is handed to ``translate_gate``.
"""

from __future__ import annotations

import warnings
from typing import Iterable, List

from qibo.models import Circuit as QiboCircuit
from qibo.transpiler.unroller import DecompositionError, NativeGates, translate_gate

from qibo_qm_provider.exceptions import UnsupportedGateError

from .gate_map import DEFERRED_GATES

__all__ = ["default_transpile"]

#: Gate names the generic decomposition machinery must never see.
#: ``DEFERRED_GATES`` already has no representation on any lowering path in
#: this package -- decomposing them here would replace the specific,
#: actionable error each path already raises with a generic transpile
#: failure (or, for a couple of them, a bare upstream ``KeyError``). ``M`` is
#: Qibo's measurement gate, handled separately throughout this package,
#: never through gate decomposition.
_SKIP_DECOMPOSITION = frozenset(DEFERRED_GATES) | {"M"}


def default_transpile(
    circuit: QiboCircuit,
    *,
    already_native: Iterable[str],
    decomposition_targets: Iterable[str],
    warn_on_change: bool = True,
) -> QiboCircuit:
    """Decompose every gate the target machine cannot run directly.

    ``circuit.wire_names`` (already-pinned physical layout) is preserved
    unchanged -- this only rewrites gates, it never re-places qubits.

    Args:
        circuit: The Qibo circuit to transpile.
        already_native: Qibo gate class names the target can execute without
            decomposition (e.g. every QuAM macro currently installed, or a
            qibolab ``Platform``'s own native-gate list) -- left untouched,
            in whatever form the caller already wrote them.
        decomposition_targets: The subset of ``qibo.transpiler.unroller.
            NativeGates``'s nine members the target can execute, used as the
            decomposition's output vocabulary. Entries outside that enum
            (e.g. ``"GPI"``, ``"Align"``, present in
            ``QibolabBackend.natives`` but not in ``NativeGates.
            __members__``) are ignored for this purpose -- a gate needing
            one of those as its *output* has no representation here, and
            must already be listed in ``already_native`` instead.
        warn_on_change: Emit a ``UserWarning`` naming the gates that were
            decomposed, so a caller profiling gate counts/durations is not
            silently surprised by gates they did not write. Defaults to
            ``True``; pass ``transpile=False`` to the calling
            ``execute_circuit`` (which skips calling this function
            entirely) to require an already-native circuit instead.

    Returns:
        A new circuit with every non-native gate replaced by its
        decomposition, and ``already_native``/``M``/``DEFERRED_GATES`` gates
        carried over unchanged.

    Raises:
        UnsupportedGateError: If a gate is neither already native nor
            decomposable into ``decomposition_targets`` by Qibo's
            transpiler.
    """
    skip = frozenset(already_native) | _SKIP_DECOMPOSITION
    targets = [name for name in decomposition_targets if name in NativeGates.__members__ and name != "NONE"]
    native_flag = NativeGates[targets] if targets else None

    new_circuit = QiboCircuit(**circuit.init_kwargs)
    decomposed_gate_names: List[str] = []
    for gate in circuit.queue:
        name = type(gate).__name__
        if name in skip:
            new_circuit.add(gate)
            continue
        if native_flag is None:
            raise UnsupportedGateError(
                f"Qibo gate {name} on qubits {gate.qubits} is not native to this "
                f"machine, and no qibo.transpiler.unroller.NativeGates member is "
                f"available to decompose it into (the machine reports none as "
                f"native). Decompose it yourself first, or pass transpile=False."
            )
        try:
            decomposition = translate_gate(gate, native_flag)
        except (KeyError, DecompositionError, TypeError) as exc:
            # KeyError: no decomposition rule at all for this gate class (e.g.
            # PRX, U1q, MS, GPI -- qibo's tables simply have no entry).
            # DecompositionError: qibo's own single-qubit decomposer refuses
            # when neither U3 nor GPI2 is in `targets` (raised via
            # qibo.config.raise_error, not returned).
            # TypeError: qibo's two-qubit decomposer falls through with an
            # implicit `None` (no explicit "else" branch) when none of
            # CZ/iSWAP/CNOT is in `targets`, and the caller then tries to
            # iterate over it.
            raise UnsupportedGateError(
                f"Qibo gate {name} on qubits {gate.qubits} is not native to this "
                f"machine, and Qibo's transpiler has no decomposition rule for it "
                f"into this machine's native gate set ({sorted(targets)}). "
                f"Decompose it yourself first, or pass transpile=False and supply "
                f"an already-native circuit."
            ) from exc
        for decomposed_gate in decomposition if isinstance(decomposition, list) else [decomposition]:
            new_circuit.add(decomposed_gate)
        decomposed_gate_names.append(name)

    if warn_on_change and decomposed_gate_names:
        warnings.warn(
            f"execute_circuit's default transpile step decomposed "
            f"{sorted(set(decomposed_gate_names))} into this machine's native "
            f"gates because they were not already native. Pass transpile=False "
            f"to require an already-native circuit instead.",
            stacklevel=3,
        )
    return new_circuit
