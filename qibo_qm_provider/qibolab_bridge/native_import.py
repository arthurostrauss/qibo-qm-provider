"""Import calibrated Qibolab native pulses as QuAM macros.

This is a standalone importer, deliberately independent of the OQ3/`qm_qasm`
lowering path (`qibo_qm_provider.backend`): it only ever produces ordinary
QuAM macros/pulses, installed the same way ``add_basic_macros`` does, that the
wrapped ``QMBackend``'s existing Target/registry mechanism then picks up like
any hand-written macro (via ``update_target()``). It is not a second
execution path.

Scope for this slice: single-qubit ``RX``/``RX90``/``MZ`` and two-qubit ``CZ``
natives -- envelope conversion itself (all 7 qibolab envelope kinds) is
handled by :mod:`qibo_qm_provider.qibolab_bridge.quam_pulses`, shared with
the QuAM -> qibolab direction and with ``qua_macros.py``.

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
from quam.core import QuamRoot
from quam.core.macro import QuamMacro

from .naming import (
    SINGLE_QUBIT_MACRO_NAMES,
    TWO_QUBIT_MACRO_NAMES,
)
from .quam_pulses import max_voltage_for_channel, quam_pulse_from_qibolab_pulse, quam_readout_pulse_from_qibolab_readout

if TYPE_CHECKING:
    import qibolab

__all__ = ["import_qibolab_natives_as_macros"]

_CHANNEL_SUFFIX_TO_ATTR = {"drive": "xy", "probe": "resonator", "acquisition": "resonator", "flux": "z"}
_PAIR_CHANNEL_SUFFIX_TO_ATTR = {"flux": "coupler"}


def _install_native_as_macro(
    native,
    macro_name: str,
    target_component,
    name_prefix: str,
    acquisition_configs: Optional[Dict[str, object]] = None,
) -> Optional[QuamMacro]:
    """Register each importable pulse in a qibolab ``Native`` on its QuAM
    channel, and install a ``PulseMacro`` on
    ``target_component.macros[macro_name]`` for the (single) channel that
    carries it.

    Only plain qibolab ``Pulse`` instructions are imported -- a ``Readout``
    (as used by ``MZ``) contributes its ``.probe`` leg only, and non-pulse
    instructions such as ``VirtualZ`` (CZ's phase-compensation legs) are
    skipped with a warning rather than raising, since they have no
    single-channel QuAM ``Pulse`` equivalent. Channels with no corresponding
    QuAM attribute on ``target_component`` (e.g. a qubit-pair's coupler when
    absent) are likewise skipped. Returns ``None`` (installs nothing) if no
    importable leg was found.

    ``acquisition_configs`` (``Platform.parameters.configs``, keyed by
    qibolab ``ChannelId``) supplies ``threshold``/``iq_angle`` for a
    ``Readout`` leg's acquisition channel -- the qibolab ``Readout`` object
    itself carries neither (see ``quam_pulses.
    quam_readout_pulse_from_qibolab_readout``) -- so the imported ``measure``
    macro is acquisition-complete for ``AcquisitionType.DISCRIMINATION``
    whenever the source platform itself has that calibration.
    """
    from quam.components.macro import PulseMacro

    is_pair = hasattr(target_component, "qubit_control")
    suffix_to_attr = _PAIR_CHANNEL_SUFFIX_TO_ATTR if is_pair else _CHANNEL_SUFFIX_TO_ATTR

    installed_pulse_name = None
    for channel_id, instruction in native:
        if not isinstance(instruction, (QibolabReadout, QibolabPulse)):
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
        max_voltage = max_voltage_for_channel(channel)

        if isinstance(instruction, QibolabReadout):
            acq_channel_id = f"{channel_id.rsplit('/', 1)[0]}/acquisition"
            acq_config = (acquisition_configs or {}).get(acq_channel_id)
            channel.operations[pulse_name] = quam_readout_pulse_from_qibolab_readout(
                instruction,
                pulse_name,
                max_voltage,
                threshold=getattr(acq_config, "threshold", None),
                integration_weights_angle=getattr(acq_config, "iq_angle", None),
            )
        else:
            channel.operations[pulse_name] = quam_pulse_from_qibolab_pulse(instruction, pulse_name, max_voltage)
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

    An imported ``MZ`` macro's readout pulse gets its ``threshold``/
    ``iq_angle`` from ``platform.parameters.configs``'s acquisition-channel
    config (see :func:`_install_native_as_macro`) -- so it is
    acquisition-complete for ``AcquisitionType.DISCRIMINATION`` only if
    ``platform`` itself was calibrated for shot discrimination.

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
    acquisition_configs = platform.parameters.configs

    for qubit_id in qubit_ids:
        quam_qubit_name = str(qubit_id)
        if quam_qubit_name not in machine.qubits:
            continue
        quam_qubit = machine.qubits[quam_qubit_name]
        single_natives = platform.natives.single_qubit.get(qubit_id)
        if single_natives is None:
            continue
        for native_field, macro_name in SINGLE_QUBIT_MACRO_NAMES.items():
            native = getattr(single_natives, native_field)
            if native is None:
                continue
            macro = _install_native_as_macro(
                native, macro_name, quam_qubit, f"qibolab_{quam_qubit_name}", acquisition_configs
            )
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
        for native_field, macro_name in TWO_QUBIT_MACRO_NAMES.items():
            native = getattr(two_natives, native_field)
            if native is None:
                continue
            macro = _install_native_as_macro(
                native, macro_name, quam_pair, f"qibolab_{quam_pair_name}", acquisition_configs
            )
            if macro is not None:
                installed[f"{quam_pair_name}:{macro_name}"] = macro

    return installed
