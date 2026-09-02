"""Build a qibolab ``Platform`` directly from a QuAM object.

See ``qibolab_platform_from_quam_plan.md`` (repo root) for the full design
rationale. Summary: a Qibolab platform "name" is resolved by
``qibolab.create_platform`` as a literal folder name on
``$QIBOLAB_PLATFORMS``, with the folder's ``platform.py::create()`` called
with zero arguments -- so ``create_iqcc``/``create_local`` below are written
to work both as the implementation such a ``create()`` stub calls (passing
its own ``folder``, sourcing defaults from a ``quam_source.json`` sidecar)
and as a direct, zero-folder Python entrypoint (explicit keyword arguments,
no filesystem registration needed at all).

``quam_to_qibolab_platform`` builds qubits/couplers/native_gates (see
``quam_platform_conversion``) and, when the machine's wiring is supported
(OPX1000 MW-FEM + LF-FEM only -- see ``quam_wiring``'s module docstring),
``instruments``/``parameters.configs`` too, so ``Platform.channels`` is
populated and Qibocal protocols can address it. Unsupported wiring (Octave/
IQ-mixer channels, non-FEM ports, non-1-GSa/s ports) falls back to an
instruments-less ``Platform`` with a warning, preserving this function's
original inspection-only behavior for machines this package cannot wire --
it never raises where it previously succeeded.
"""

from __future__ import annotations

import json
import warnings
from importlib import import_module
from pathlib import Path
from typing import Optional, Tuple

from qibolab import Platform
from qibolab._core.instruments.qm import QmController
from qibolab._core.instruments.qm.components import QmConfigs
from qibolab._core.parameters import ConfigKinds, NativeGates, Parameters, Settings
from quam.core import QuamRoot

from ..exceptions import MissingQuamAttributeError, UnsupportedWiringError
from .iqcc_controller import IQCCQmController
from .platform_naming import IqccSource, parse_platform_name
from .quam_platform_conversion import _build_couplers, _build_native_gates, _build_qubits
from .quam_wiring import build_qm_wiring

__all__ = ["create_iqcc", "create_local", "quam_to_qibolab_platform", "DEFAULT_QUAM_CLASS"]

DEFAULT_QUAM_CLASS = (
    "quam_builder.architecture.superconducting.qpu.flux_tunable_quam.FluxTunableQuam"
)

_SOURCE_CONFIG_FILENAME = "quam_source.json"

# QM's standard cluster port, used when machine.network carries no "port"
# entry (confirmed on real IQCC state: network only has host/cluster_name/
# qmm_class/qmm_settings) -- QmController.connect() does
# self.address.split(":") unconditionally, so a port is always required even
# though QuAM's own network config doesn't carry one.
DEFAULT_QM_PORT = 9510

# qibolab's Parameters.configs field validates against a discriminated union
# of *registered* Config "kind" tags (qibolab._core.parameters.ConfigKinds) --
# only a fixed built-in set ("dc", "oscillator", "iq", "acquisition", ...) is
# registered by default. QM's own kinds ("opx-output", "mw-fem-oscillator",
# "qm-acquisition") are not among them, so constructing `Parameters(configs=...)`
# with QM Config objects raises a pydantic ValidationError ("union_tag_invalid")
# unless QmConfigs has been registered first. ConfigKinds is process-global
# state (its own docstring warns about this), so this is done once, at import
# time, rather than per-call -- idempotent because ConfigKinds.extend is only
# ever called from here in this package.
ConfigKinds.extend([QmConfigs])


def _import_class(dotted: str) -> type:
    """Import a class from its dotted path, e.g. ``"pkg.mod.ClassName"``."""
    module_name, _, cls_name = dotted.rpartition(".")
    return getattr(import_module(module_name), cls_name)


def _read_source_config(folder: Optional[Path]) -> dict:
    if folder is None:
        return {}
    path = Path(folder) / _SOURCE_CONFIG_FILENAME
    return json.loads(path.read_text()) if path.exists() else {}


def _resolve_state_path(state_path: str, folder: Optional[Path]) -> str:
    """Resolve ``state_path`` against ``folder`` when relative.

    An absolute ``state_path`` is returned unchanged. A relative one
    resolves against the platform's own folder, so per-machine state (e.g.
    a cached IQCC fetch) lives inside that machine's own
    ``$QIBOLAB_PLATFORMS`` folder by default, never colliding with a
    different machine's folder or with a separately-maintained local state.
    """
    p = Path(state_path)
    if p.is_absolute() or folder is None:
        return str(p)
    return str(Path(folder) / p)


def _resolve_controller_class(machine: QuamRoot) -> type:
    """Whether ``machine.connect()`` resolves to IQCC's cloud manager.

    Reuses ``quam_builder``'s own private resolution helper
    (``Quam._get_qmm_class``) rather than re-deriving its
    ``use_custom_qmm``/``qmm_class`` branching here, so this can never drift
    from what ``connect()`` itself would do. Deliberately fails soft (falls
    back to the plain ``QmController``, today's unconditional behavior) on
    any resolution problem -- a bad ``qmm_class``/missing
    ``iqcc_cloud_client`` should surface at ``connect()`` time, exactly as it
    did before this function existed, not block ``Platform`` construction.
    """
    try:
        get_qmm_class = getattr(machine, "_get_qmm_class", None)
        qmm_class = get_qmm_class() if get_qmm_class is not None else None
    except Exception:  # noqa: BLE001 - see docstring: fail soft, let connect() raise later.
        return QmController
    if qmm_class is None:
        return QmController
    try:
        from iqcc_cloud_client.qmm_cloud import CloudQuantumMachinesManager
    except ImportError:
        return QmController
    return IQCCQmController if issubclass(qmm_class, CloudQuantumMachinesManager) else QmController


def _build_qm_controller(machine: QuamRoot, *, port: Optional[int] = None):
    """Build a qibolab ``QmController`` (or ``IQCCQmController``) + matching
    ``configs`` from a QuAM object's wiring.

    The actual channel/config conversion lives in
    ``quam_wiring.build_qm_wiring``; this function only assembles the
    controller itself (``address``, ``cluster_name``, ``fems``, and --
    exclusively for ``IQCCQmController`` -- ``machine``) from
    ``machine.network``. Which controller class gets built is decided by
    :func:`_resolve_controller_class`, from that same ``network`` config --
    the identical, config-driven distinction ``machine.connect()`` itself
    makes, so a locally-wired machine and an IQCC-fetched one need no
    separate call path here.

    Args:
        machine: The QuAM root to wire.
        port: Cluster port override. Falls back to ``machine.network["port"]``
            if present, else :data:`DEFAULT_QM_PORT` (with a warning, since a
            wrong guess here only surfaces as a failure at ``connect()``
            time, far from this call).

    Raises:
        MissingQuamAttributeError: If ``machine.network`` has no ``"host"``.
        UnsupportedWiringError: See ``quam_wiring.build_qm_wiring``.
    """
    network = getattr(machine, "network", None) or {}
    host = network.get("host")
    if not host:
        raise MissingQuamAttributeError(
            "machine.network is missing a 'host' entry -- required to build a "
            "QmController.address."
        )
    resolved_port = port if port is not None else network.get("port")
    if resolved_port is None:
        resolved_port = DEFAULT_QM_PORT
        warnings.warn(
            f"machine.network has no 'port' entry -- defaulting the QmController "
            f"address's port to {DEFAULT_QM_PORT} (QM's standard cluster port). Pass "
            "an explicit `port` to quam_to_qibolab_platform if this cluster uses a "
            "different one.",
            stacklevel=3,
        )

    channels, configs, fems = build_qm_wiring(machine)
    controller_cls = _resolve_controller_class(machine)
    extra = {"machine": machine} if controller_cls is IQCCQmController else {}
    controller = controller_cls(
        address=f"{host}:{resolved_port}",
        cluster_name=network.get("cluster_name"),
        channels=channels,
        fems=fems,
        **extra,
    )
    return controller, configs


def quam_to_qibolab_platform(machine: QuamRoot, name: str, *, port: Optional[int] = None) -> Platform:
    """Build a qibolab ``Platform`` from a QuAM object.

    Populates ``qubits``/``couplers`` (topology) and ``parameters.
    native_gates`` (from QuAM's ``.macros``) in one pass -- see
    ``quam_platform_conversion`` for the per-piece logic.
    ``parameters.settings.relaxation_time`` is set from
    ``machine.thermalization_time`` when available (best-effort; falls back
    to qibolab's own default otherwise).

    ``instruments``/``parameters.configs`` are populated from the machine's
    OPX1000 MW-FEM/LF-FEM wiring (see ``quam_wiring``) when supported. If
    the machine's wiring is not supported (Octave/IQ-mixer channels,
    non-FEM ports, non-1-GSa/s ports) or ``machine.network`` has no
    ``"host"`` (e.g. a machine never connected to a real cluster, like this
    package's own synthetic test fixture), this falls back to an
    instruments-less ``Platform`` with a warning naming the cause, rather
    than raising -- topology/native-gate inspection and
    ``QiboQMPlatformBackend`` construction still work either way; only
    ``.execute()``/``.connect()`` need working instruments.

    Args:
        machine: The QuAM root to convert.
        name: The platform's name (informational only in this pass --
            ``Platform.name`` doesn't affect resolution, since qibolab
            resolves platforms by folder name before ``create()`` ever
            runs).
        port: Forwarded to :func:`_build_qm_controller` as a cluster-port
            override; see its docstring.

    Returns:
        A qibolab ``Platform``, with working ``instruments`` when the
        machine's wiring is supported and a fully populated topology and
        native-gate set either way.
    """
    qubits = _build_qubits(machine)
    couplers = _build_couplers(machine)
    native_gates: NativeGates = _build_native_gates(machine)

    settings = Settings()
    try:
        settings = Settings(relaxation_time=machine.thermalization_time)
    except Exception:  # noqa: BLE001 - best-effort; not every QuAM machine exposes this.
        pass

    instruments: dict = {}
    configs: dict = {}
    try:
        controller, configs = _build_qm_controller(machine, port=port)
        instruments = {"qm": controller}
    except (UnsupportedWiringError, MissingQuamAttributeError) as exc:
        warnings.warn(
            f"Could not build QmController instrument wiring for platform {name!r}: "
            f"{exc} -- falling back to an instruments-less Platform (topology/"
            "native-gate inspection only; .execute()/.connect() will not work).",
            stacklevel=2,
        )

    parameters = Parameters(settings=settings, configs=configs, native_gates=native_gates)
    return Platform(name=name, parameters=parameters, instruments=instruments, qubits=qubits, couplers=couplers)


def _create_iqcc_with_machine(
    name: str,
    folder: Optional[Path] = None,
    *,
    state_path: Optional[str] = None,
    quam_class: Optional[str] = None,
    api_token: Optional[str] = None,
) -> Tuple[Platform, QuamRoot]:
    """Fetch the named machine's latest state from IQCC and convert it.

    ``folder``'s ``quam_source.json`` supplies defaults (used by a
    ``platform.py`` stub inside a registered ``$QIBOLAB_PLATFORMS`` folder);
    explicit keyword arguments override them (used by
    ``QiboQMPlatformBackend.from_iqcc`` and other direct-Python callers).
    Both call paths go through this one implementation.

    Raises:
        InvalidPlatformName: If ``name`` doesn't match the
            ``qibo-qm-iqcc-<backend_name>`` grammar.
        ValueError: If the IQCC backend can't be reached or doesn't exist
            (wraps the ``ValueError``/``ConnectionError`` that
            ``iqcc_cloud_client``'s HTTP error handling actually raises for
            an unknown/unreachable backend name).
    """
    source = parse_platform_name(name)
    assert isinstance(source, IqccSource)

    config = _read_source_config(folder)
    resolved_state_path = _resolve_state_path(
        state_path or config.get("state_path", "quam_state"), folder
    )
    quam_cls = _import_class(quam_class or config.get("quam_class", DEFAULT_QUAM_CLASS))
    Path(resolved_state_path).mkdir(parents=True, exist_ok=True)

    # Lazy-imported: qiskit_qm_provider/iqcc_cloud_client are not declared
    # dependencies of this module's own import-time surface, matching this
    # codebase's existing convention for iqcc_cloud_client-adjacent code.
    from qiskit_qm_provider.providers.iqcc_cloud_provider import get_machine_from_iqcc

    try:
        machine, _ = get_machine_from_iqcc(
            source.backend_name,
            api_token=api_token,
            quam_state_folder_path=resolved_state_path,
            quam_cls=quam_cls,
        )
    except (ValueError, ConnectionError) as exc:
        raise ValueError(
            f"IQCC backend {source.backend_name!r} (from platform name {name!r}) "
            "could not be reached or does not exist."
        ) from exc
    return quam_to_qibolab_platform(machine, name=name), machine


def _create_local_with_machine(
    folder: Optional[Path] = None,
    *,
    state_path: Optional[str] = None,
    quam_class: Optional[str] = None,
) -> Tuple[Platform, QuamRoot]:
    """Load an already-existing local QuAM state and convert it.

    Same override relationship as ``_create_iqcc_with_machine``: ``folder``'s
    ``quam_source.json`` supplies defaults, explicit keyword arguments win
    when given.

    Raises:
        ValueError: If no ``state_path`` is available from either source --
            there is no safe default for a pre-existing local state.
    """
    config = _read_source_config(folder)
    raw_state_path = state_path or config.get("state_path")
    if raw_state_path is None:
        raise ValueError(
            "create_local requires a state_path, either via a quam_source.json "
            "sidecar or the state_path keyword argument -- there is no safe "
            "default for a pre-existing local state."
        )
    resolved_state_path = _resolve_state_path(raw_state_path, folder)
    quam_cls = _import_class(quam_class or config.get("quam_class", DEFAULT_QUAM_CLASS))
    machine = quam_cls.load(resolved_state_path)
    name = folder.name if folder is not None else "qibo-qm-local"
    return quam_to_qibolab_platform(machine, name=name), machine


def create_iqcc(
    name: str,
    folder: Optional[Path] = None,
    *,
    state_path: Optional[str] = None,
    quam_class: Optional[str] = None,
    api_token: Optional[str] = None,
) -> Platform:
    """Fetch the named machine's latest state from IQCC and build a `Platform`.

    See :func:`_create_iqcc_with_machine` for the full behavior/error
    contract; this is the public, machine-object-discarding wrapper used by
    a ``$QIBOLAB_PLATFORMS/qibo-qm-iqcc-<name>/platform.py`` stub.
    """
    return _create_iqcc_with_machine(
        name, folder, state_path=state_path, quam_class=quam_class, api_token=api_token
    )[0]


def create_local(
    folder: Optional[Path] = None,
    *,
    state_path: Optional[str] = None,
    quam_class: Optional[str] = None,
) -> Platform:
    """Load an already-existing local QuAM state and build a `Platform`.

    See :func:`_create_local_with_machine` for the full behavior/error
    contract; this is the public, machine-object-discarding wrapper used by
    a ``$QIBOLAB_PLATFORMS/qibo-qm-local/platform.py`` stub.
    """
    return _create_local_with_machine(folder, state_path=state_path, quam_class=quam_class)[0]
