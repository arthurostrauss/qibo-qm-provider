"""Backend representation that mirrors a Qibo device for QM stack integrations."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QiboDeviceSpec:
    """Minimal Qibo device-class representation."""

    name: str
    qubits: int
    native_gates: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


class QmBackendRepresentation:
    """Serializable bridge object to map a Qibo-like device into QM concepts."""

    def __init__(self, device: QiboDeviceSpec):
        self.device = device

    def to_qm_payload(self) -> dict[str, object]:
        return {
            "device": {
                "name": self.device.name,
                "qubits": self.device.qubits,
                "native_gates": list(self.device.native_gates),
                "metadata": dict(self.device.metadata),
            },
            "qua": {
                "description": "Placeholder QUA-oriented device descriptor",
            },
        }
