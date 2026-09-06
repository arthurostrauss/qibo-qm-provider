"""Scaffold the on-disk ``$QIBOLAB_PLATFORMS`` folder pair that
``qibolab.create_platform``/name-based resolution needs.

See ``platform_registration_guidelines.md`` §9 and
``qibolab_platform_from_quam_plan.md`` §1-3 (repo root) for the design this
implements. ``create_iqcc``/``create_local`` (what the written
``platform.py`` stub calls) already exist and are tested -- the only thing
missing, and the only thing this module adds, is writing the
``platform.py`` + ``quam_source.json`` pair to disk.

Folder names are always computed through ``platform_naming`` (never
re-derived here), so a scaffolded folder is guaranteed to parse the same
way ``create_iqcc``/``create_local`` parse it at load time. Per
``qibolab_platform_from_quam_plan.md`` §1, ``qibo-qm-local`` is
deliberately singular in this version -- there is no
``qibo-qm-local-<label>`` variant (confirmed rejected by
``test_platform_naming.py::test_rejects_invalid_names``), so
:func:`scaffold_local_platform` scaffolds exactly one folder, matching the
grammar as actually implemented rather than as a hypothetical future
extension.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from .platform_from_quam import DEFAULT_QUAM_CLASS, create_iqcc
from .platform_naming import IQCC_PREFIX, LOCAL_NAME, PREFIX

__all__ = ["scaffold_iqcc_platform", "scaffold_local_platform", "DEFAULT_PLATFORMS_DIR"]

# Per-user directory, per platform_registration_guidelines.md §2: survives
# repo re-clones/reinstalls and is identical regardless of which checkout or
# venv is active, unlike a path inside this repo or inside the installed
# package itself (not writable without elevated permissions, fragile across
# upgrades).
DEFAULT_PLATFORMS_DIR = Path.home() / ".qibolab" / "platforms"

_SOURCE_CONFIG_FILENAME = "quam_source.json"

# Byte-identical across every folder of a given kind -- all per-machine
# variation lives in quam_source.json, not in this stub. See
# qibolab_platform_from_quam_plan.md §2.
_IQCC_PLATFORM_PY = '''from pathlib import Path

from qibo_qm_provider.qibolab_bridge.platform_from_quam import create_iqcc

FOLDER = Path(__file__).parent


def create():
    return create_iqcc(FOLDER.name, FOLDER)
'''

_LOCAL_PLATFORM_PY = '''from pathlib import Path

from qibo_qm_provider.qibolab_bridge.platform_from_quam import create_local

FOLDER = Path(__file__).parent


def create():
    return create_local(FOLDER)
'''


def _write_folder(dest_dir: Union[str, Path], name: str, platform_py: str, source_config: dict) -> Path:
    folder = Path(dest_dir).expanduser() / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "platform.py").write_text(platform_py)
    (folder / _SOURCE_CONFIG_FILENAME).write_text(json.dumps(source_config, indent=2) + "\n")
    return folder


def scaffold_iqcc_platform(
    backend_name: str,
    dest_dir: Union[str, Path],
    *,
    quam_class: Optional[str] = None,
    validate: bool = True,
    api_token: Optional[str] = None,
) -> Path:
    """Write the ``qibo-qm-iqcc-<backend_name>`` folder pair to ``dest_dir``.

    Args:
        backend_name: The IQCC backend name, e.g. ``"arbel"`` -- not the
            ``qibo-qm-iqcc-`` prefixed folder name, which is computed here.
        dest_dir: Directory to scaffold under (e.g. ``~/.qibolab/platforms``,
            :data:`DEFAULT_PLATFORMS_DIR`) -- must be a directory
            ``$QIBOLAB_PLATFORMS`` will point at, not this package's own
            installed directory (see ``platform_registration_guidelines.md``
            §2).
        quam_class: Recorded in ``quam_source.json``; defaults to
            :data:`~.platform_from_quam.DEFAULT_QUAM_CLASS` if omitted.
        validate: If ``True`` (default), immediately calls :func:`create_iqcc`
            once after writing, so a typo'd backend name or unreachable IQCC
            backend is caught and reported now, at scaffold time, rather than
            at every future ``create_platform`` call (per
            ``qibolab_platform_from_quam_plan.md`` §3). Also reports whether
            the resulting ``Platform.instruments`` came back populated --
            not every wired machine's instruments build successfully (see
            ``platform_registration_guidelines.md`` §7), and a silently
            inspection-only platform is a confusing thing to discover for
            the first time mid-experiment.
        api_token: Forwarded to the validation call only -- never written to
            ``quam_source.json``, which is meant to be a committable,
            non-secret sidecar (``platform_registration_guidelines.md`` §8).

    Returns:
        The scaffolded folder's path.

    Raises:
        ValueError: If ``validate`` is ``True`` and the IQCC backend can't be
            reached or doesn't exist (see :func:`create_iqcc`).
    """
    name = f"{PREFIX}{IQCC_PREFIX}{backend_name}"
    folder = _write_folder(
        dest_dir,
        name,
        _IQCC_PLATFORM_PY,
        {"state_path": "quam_state", "quam_class": quam_class or DEFAULT_QUAM_CLASS},
    )
    if validate:
        platform = create_iqcc(name, folder, quam_class=quam_class, api_token=api_token)
        status = (
            "populated"
            if platform.instruments
            else "empty (instruments-less Platform -- unsupported wiring, or no machine.network host)"
        )
        print(f"{name}: scaffolded at {folder}, instruments {status}.")
    return folder


def scaffold_local_platform(
    state_path: Union[str, Path],
    dest_dir: Union[str, Path],
    *,
    quam_class: Optional[str] = None,
) -> Path:
    """Write the (singular, v1) ``qibo-qm-local`` folder to ``dest_dir``,
    referencing an already-existing local QuAM state.

    Never copies ``state_path``'s contents into the scaffolded folder --
    only records its resolved absolute path in ``quam_source.json`` -- so
    the platform folder never becomes a second, driftable source of truth
    for calibration data that remains the user's own responsibility to
    maintain (``platform_registration_guidelines.md`` §5). Not validated by
    attempting a load here (unlike the IQCC case, there is no live
    connection to test at scaffold time); a typo'd ``state_path`` surfaces
    the first time ``create_platform``/``create_local`` actually loads it.

    Args:
        state_path: Path to the already-existing local QuAM state directory.
        dest_dir: Directory to scaffold under.
        quam_class: Recorded in ``quam_source.json``; defaults to
            :data:`~.platform_from_quam.DEFAULT_QUAM_CLASS` if omitted.

    Returns:
        The scaffolded folder's path.
    """
    name = f"{PREFIX}{LOCAL_NAME}"
    return _write_folder(
        dest_dir,
        name,
        _LOCAL_PLATFORM_PY,
        {
            "state_path": str(Path(state_path).expanduser().resolve()),
            "quam_class": quam_class or DEFAULT_QUAM_CLASS,
        },
    )


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m qibo_qm_provider.qibolab_bridge.scaffold",
        description="Scaffold qibo-qm-provider platform folders on $QIBOLAB_PLATFORMS.",
    )
    subparsers = parser.add_subparsers(dest="kind", required=True)

    iqcc_parser = subparsers.add_parser("iqcc", help="Scaffold one folder per IQCC backend name.")
    iqcc_parser.add_argument("backend_names", nargs="+", help="e.g. arbel gilboa qolab")
    iqcc_parser.add_argument("--dest", default=str(DEFAULT_PLATFORMS_DIR), help="Destination directory.")
    iqcc_parser.add_argument("--quam-class", default=None, dest="quam_class")
    iqcc_parser.add_argument("--api-token", default=None, dest="api_token")
    iqcc_parser.add_argument(
        "--no-validate", action="store_false", dest="validate", help="Skip the live IQCC validation fetch."
    )

    local_parser = subparsers.add_parser("local", help="Scaffold the single qibo-qm-local folder.")
    local_parser.add_argument("state_path", help="Path to an already-existing local QuAM state directory.")
    local_parser.add_argument("--dest", default=str(DEFAULT_PLATFORMS_DIR), help="Destination directory.")
    local_parser.add_argument("--quam-class", default=None, dest="quam_class")

    return parser


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    if args.kind == "iqcc":
        for backend_name in args.backend_names:
            scaffold_iqcc_platform(
                backend_name,
                args.dest,
                quam_class=args.quam_class,
                validate=args.validate,
                api_token=args.api_token,
            )
    else:
        folder = scaffold_local_platform(args.state_path, args.dest, quam_class=args.quam_class)
        print(f"qibo-qm-local: scaffolded at {folder}.")


if __name__ == "__main__":
    main()
