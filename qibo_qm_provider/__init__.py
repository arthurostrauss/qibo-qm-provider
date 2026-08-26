"""An interface for compiling and executing Qibo workflows on Quantum
Machines' Quantum Orchestration Platform.

``QiboQMBackend`` and ``FluxTunableTransmonBackend`` are translation shims
around ``qiskit_qm_provider.backend.qm_backend.QMBackend``: macro
installation, Target/operation-mapping population, calibration-override
precedence, QUA program construction, QM connection handling, job
submission, and result fetching are all inherited from that wrapped
instance, not reimplemented here.

``QiboQMPlatformBackend`` is a separate execution path: a Qibo backend
wrapping a genuine ``qibolab.Platform`` built directly from QuAM (see
``qibolab_bridge.platform_from_quam``), so that Qibocal's existing
calibration protocols work against QM hardware without an OpenQASM
round-trip. This module's top-level ``MetaBackend`` is what makes
``qibo.set_backend("qibo_qm_provider", platform=...)`` resolve to that
class -- the standard Qibo backend-discovery mechanism (any importable
package exposing a ``MetaBackend.load(**kwargs) -> Backend`` staticmethod
becomes a loadable backend by name), the same mechanism ``qibolab`` itself
uses for ``qibo.set_backend("qibolab", ...)``.
"""

from typing import Union

from qibolab import Platform

from .backend import (
    FluxTunableTransmonBackend,
    QiboParameterTable,
    QiboQMBackend,
    QiboQMPlatformBackend,
    build_qiskit_circuit_directly,
    circuit_has_symbols,
    qibo_circuit_to_qiskit,
    translate_measurements,
)
from .qibolab_bridge import import_qibolab_natives_as_macros
from .quam_macros.superconducting import add_basic_macros

__all__ = [
    "QiboQMBackend",
    "QiboQMPlatformBackend",
    "FluxTunableTransmonBackend",
    "qibo_circuit_to_qiskit",
    "build_qiskit_circuit_directly",
    "circuit_has_symbols",
    "QiboParameterTable",
    "translate_measurements",
    "add_basic_macros",
    "import_qibolab_natives_as_macros",
    "MetaBackend",
]


class MetaBackend:
    """Meta-backend class which takes care of loading ``qibo_qm_provider``
    as a Qibo backend, e.g. via ``qibo.set_backend("qibo_qm_provider",
    platform=...)``.

    Deliberately does **not** route through ``qibolab``'s own
    ``MetaBackend``/``QibolabBackend`` (which is hardcoded to always
    construct a plain ``QibolabBackend``) -- registering under this
    package's own name instead is what lets ``platform=...`` resolve to
    ``QiboQMPlatformBackend`` here, while plain
    ``qibo.set_backend("qibolab", platform=...)`` remains fully valid and
    unaffected, returning a bare ``QibolabBackend`` on the exact same
    platform (see ``qibo_backend_vs_qibolab_platform.md`` §4.2, repo root).
    """

    @staticmethod
    def load(platform: Union[str, Platform], **kwargs) -> QiboQMPlatformBackend:
        """Load the backend.

        Args:
            platform: A registered platform name (resolved via
                ``qibolab.create_platform``) or an already-built ``Platform``
                object.
            kwargs: Additional keyword arguments forwarded to
                ``QiboQMPlatformBackend``.
        """
        return QiboQMPlatformBackend(platform=platform, **kwargs)

    def list_available(self) -> dict:
        """Lists available qibo-qm-provider platforms.

        Delegates to qibolab's own ``available_platforms()`` (this package
        doesn't maintain a separate registry) -- a ``qibo-qm-provider``
        platform folder is a regular ``$QIBOLAB_PLATFORMS`` entry.
        """
        from qibolab._core.platform.load import available_platforms

        return {name: True for name in available_platforms()}
