"""Import calibrated Qibolab native pulses as QuAM macros.

This is a standalone importer, deliberately independent of the OQ3/`qm_qasm`
lowering path (`qibo_qm_provider.backend`): it only ever produces ordinary
QuAM macros/pulses, installed the same way ``add_basic_macros`` does, that the
wrapped ``QMBackend``'s existing Target/registry mechanism then picks up like
any hand-written macro (via ``update_target()``). It is not a second
execution path.

Scope for this slice: single-qubit ``RX``/``RX90``/``MZ`` and two-qubit ``CZ``
natives, for pulses whose envelope is ``Rectangular``, ``Gaussian``, or
``Drag`` (the qibolab envelope kinds with a direct QuAM ``Pulse`` subclass
equivalent). Other envelope kinds raise explicitly rather than being silently
dropped or approximated.

Qibolab's ``Pulse.amplitude`` is a dimensionless value normalized to roughly
``[-1, 1]``, while QuAM ``Pulse`` amplitudes are in volts -- this importer
does not attempt unit reconciliation; it passes the qibolab amplitude through
as the QuAM pulse's amplitude field, so calibration data imported this way
must already have been produced in volts (e.g. hand-populated
``platform.natives`` for QM hardware), not assumed compatible across
arbitrary Qibolab platforms without review.

Qubit/pair identity mapping: Qibolab ``QubitId``/``QubitPairId`` values are
matched to QuAM qubit/pair names via ``str(qubit_id)``/``f"{c}-{t}"`` by
default -- there is no guaranteed correspondence between the two label
spaces in general, so callers with a different naming convention should
adapt this function rather than relying on the default.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, Optional, Sequence

from qibolab._core.pulses.pulse import Pulse as QibolabPulse
from qibolab._core.pulses.pulse import Readout as QibolabReadout
from quam.components.pulses import DragCosinePulse, GaussianPulse, Pulse as QuamPulse, SquarePulse
from quam.core import QuamRoot
from quam.core.macro import QuamMacro

if TYPE_CHECKING:
    import qibolab

__all__ = ["import_qibolab_natives_as_macros"]

# RX/RX90 map onto a plain drive Pulse; MZ (a Readout) only has its `.probe`
# leg imported (best-effort: no integration-weight/kernel setup is attempted,
# so an imported "measure" macro is not acquisition-complete on its own).
# CZ's VirtualZ phase-compensation legs are skipped -- only the coupler's
# flux Pulse leg is imported (see _install_native_as_macro's Pulse-only
# filter below); a phase-accurate CZ macro is a follow-up, not this slice.
_SINGLE_QUBIT_MACRO_NAMES = {"RX": "x", "RX90": "sx", "MZ": "measure"}
_TWO_QUBIT_MACRO_NAMES = {"CZ": "cz"}

_CHANNEL_SUFFIX_TO_ATTR = {
    "drive": "xy",
    "probe": "resonator",
    "acquisition": "resonator",
    "flux": "z",
}
_PAIR_CHANNEL_SUFFIX_TO_ATTR = {"flux": "coupler"}


def _quam_pulse_from_qibolab_pulse(pulse: "qibolab.Pulse", name: str, anharmonicity: float = 0.0) -> QuamPulse:
    """Build a QuAM ``Pulse`` matching one qibolab ``Pulse``'s envelope/duration/amplitude."""
    kind = pulse.envelope.kind
    if kind == "rectangular":
        return SquarePulse(length=pulse.duration, amplitude=pulse.amplitude, id=name)
    if kind == "gaussian":
        return GaussianPulse(
            length=pulse.duration,
            amplitude=pulse.amplitude,
            sigma=pulse.envelope.rel_sigma * pulse.duration,
            id=name,
        )
    if kind == "drag":
        # qibolab's `beta` and QuAM's `alpha` are both DRAG-scaling
        # coefficients but are not verified to share the same sign/unit
        # convention -- passed through as-is (best-effort, flagged here
        # rather than silently assumed correct). `anharmonicity` is read off
        # the target QuAM qubit when available (0.0, physically wrong for a
        # real transmon, otherwise -- see caller).
        return DragCosinePulse(
            length=pulse.duration,
            amplitude=pulse.amplitude,
            alpha=pulse.envelope.beta,
            anharmonicity=anharmonicity,
            axis_angle=0.0,
            id=name,
        )
    raise NotImplementedError(
        f"Envelope kind {kind!r} (pulse {name!r}) has no QuAM Pulse equivalent supported by "
        "this importer. Supported kinds: rectangular, gaussian, drag."
    )


def _install_native_as_macro(native, macro_name: str, target_component, name_prefix: str) -> Optional[QuamMacro]:
    """Register each importable pulse in a qibolab ``Native`` on its QuAM
    channel, and install a ``PulseMacro`` on
    ``target_component.macros[macro_name]`` for the (single) channel that
    carries it.

    Only plain qibolab ``Pulse`` instructions are imported -- a ``Readout``
    (as used by ``MZ``) contributes its ``.probe`` leg only (best-effort, no
    integration-weight/kernel setup), and non-pulse instructions such as
    ``VirtualZ`` (CZ's phase-compensation legs) are skipped with a warning
    rather than raising, since they have no single-channel QuAM ``Pulse``
    equivalent. Channels with no corresponding QuAM attribute on
    ``target_component`` (e.g. a qubit-pair's coupler when absent) are
    likewise skipped. Returns ``None`` (installs nothing) if no importable
    leg was found.
    """
    from quam.components.macro import PulseMacro

    is_pair = hasattr(target_component, "qubit_control")
    suffix_to_attr = _PAIR_CHANNEL_SUFFIX_TO_ATTR if is_pair else _CHANNEL_SUFFIX_TO_ATTR
    anharmonicity = getattr(target_component, "anharmonicity", 0.0) or 0.0

    installed_pulse_name = None
    for channel_id, instruction in native:
        if isinstance(instruction, QibolabReadout):
            pulse = instruction.probe
        elif isinstance(instruction, QibolabPulse):
            pulse = instruction
        else:
            warnings.warn(
                f"Skipping non-pulse instruction {type(instruction).__name__!r} on "
                f"channel {channel_id!r} while importing native {macro_name!r} -- "
                "no single-channel QuAM Pulse equivalent (e.g. a VirtualZ "
                "phase-compensation leg)."
            )
            continue

        suffix = channel_id.rsplit("/", 1)[-1]
        attr = suffix_to_attr.get(suffix)
        if attr is None or not hasattr(target_component, attr) or getattr(target_component, attr) is None:
            continue
        channel = getattr(target_component, attr)
        pulse_name = f"{name_prefix}_{macro_name}"
        channel.operations[pulse_name] = _quam_pulse_from_qibolab_pulse(pulse, pulse_name, anharmonicity)
        installed_pulse_name = pulse_name

    if installed_pulse_name is None:
        return None
    macro = PulseMacro(pulse=installed_pulse_name)
    target_component.macros[macro_name] = macro
    return macro


def import_qibolab_natives_as_macros(
    platform: "qibolab.Platform",
    machine: QuamRoot,
    qubits: Optional[Sequence[int]] = None,
) -> Dict[str, QuamMacro]:
    """Read calibrated Qibolab single-/two-qubit native pulses and materialize
    them as QuAM macros on ``machine``'s qubits/pairs.

    Does not touch a ``QiboQMBackend`` directly -- returns what it installed
    so a caller can call ``backend.qiskit_backend.update_target()``
    afterward, the same integration seam as ``add_basic_macros``.

    Args:
        platform: A qibolab ``Platform`` with calibrated ``natives``.
        machine: The QuAM root to install macros on.
        qubits: Qibolab qubit ids to import (default: all of
            ``platform.qubits``).

    Returns:
        Mapping from ``"{qubit_or_pair}:{macro_name}"`` to the installed
        ``QuamMacro``.
    """
    installed: Dict[str, QuamMacro] = {}
    qubit_ids = list(qubits) if qubits is not None else list(platform.qubits)

    for qubit_id in qubit_ids:
        quam_qubit_name = str(qubit_id)
        if quam_qubit_name not in machine.qubits:
            continue
        quam_qubit = machine.qubits[quam_qubit_name]
        single_natives = platform.natives.single_qubit.get(qubit_id)
        if single_natives is None:
            continue
        for native_field, macro_name in _SINGLE_QUBIT_MACRO_NAMES.items():
            native = getattr(single_natives, native_field)
            if native is None:
                continue
            macro = _install_native_as_macro(native, macro_name, quam_qubit, f"qibolab_{quam_qubit_name}")
            if macro is not None:
                installed[f"{quam_qubit_name}:{macro_name}"] = macro

    for pair_id, two_natives in platform.natives.two_qubit.items():
        control, target = pair_id
        if qubits is not None and (control not in qubits or target not in qubits):
            continue
        quam_pair_name = f"{control}-{target}"
        if quam_pair_name not in machine.qubit_pairs:
            continue
        quam_pair = machine.qubit_pairs[quam_pair_name]
        for native_field, macro_name in _TWO_QUBIT_MACRO_NAMES.items():
            native = getattr(two_natives, native_field)
            if native is None:
                continue
            macro = _install_native_as_macro(native, macro_name, quam_pair, f"qibolab_{quam_pair_name}")
            if macro is not None:
                installed[f"{quam_pair_name}:{macro_name}"] = macro

    return installed
