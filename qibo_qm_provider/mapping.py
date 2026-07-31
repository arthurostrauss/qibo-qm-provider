"""Direct mapping helpers between Qibocal nodes and Qualibrate nodes."""

from __future__ import annotations


class QibocalQualibrateMapper:
    """Stores explicit links between Qibocal node names and Qualibrate nodes."""

    def __init__(self) -> None:
        self._mapping: dict[str, str] = {}

    def register(self, qibocal_node: str, qualibrate_node: str) -> None:
        if not qibocal_node or not qualibrate_node:
            raise ValueError("Both qibocal_node and qualibrate_node are required.")
        self._mapping[qibocal_node] = qualibrate_node

    def resolve(self, qibocal_node: str) -> str:
        try:
            return self._mapping[qibocal_node]
        except KeyError as exc:
            raise KeyError(f"No Qualibrate mapping defined for Qibocal node '{qibocal_node}'.") from exc

    def as_dict(self) -> dict[str, str]:
        return dict(self._mapping)
