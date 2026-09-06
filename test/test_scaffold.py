"""Tests for qibo_qm_provider.qibolab_bridge.scaffold.

Mirrors test_platform_from_quam.py's mocking convention for the IQCC path
(monkeypatching qiskit_qm_provider.providers.iqcc_cloud_provider.
get_machine_from_iqcc rather than any real network call).
"""

import json

import pytest

from qibo_qm_provider.qibolab_bridge.platform_naming import IqccSource, LocalSource, parse_platform_name
from qibo_qm_provider.qibolab_bridge.scaffold import scaffold_iqcc_platform, scaffold_local_platform


def test_scaffold_local_platform_writes_folder_pair(tmp_path):
    state_dir = tmp_path / "my_rig"
    state_dir.mkdir()
    dest = tmp_path / "platforms"

    folder = scaffold_local_platform(str(state_dir), dest)

    assert folder == dest / "qibo-qm-local"
    assert parse_platform_name(folder.name) == LocalSource()

    platform_py = (folder / "platform.py").read_text()
    assert "create_local(FOLDER)" in platform_py

    source_config = json.loads((folder / "quam_source.json").read_text())
    assert source_config["state_path"] == str(state_dir.resolve())
    assert "quam_builder" in source_config["quam_class"]


def test_scaffold_local_platform_records_explicit_quam_class(tmp_path):
    folder = scaffold_local_platform(str(tmp_path / "rig"), tmp_path / "platforms", quam_class="pkg.mod.Cls")
    source_config = json.loads((folder / "quam_source.json").read_text())
    assert source_config["quam_class"] == "pkg.mod.Cls"


def test_scaffold_local_platform_never_copies_state(tmp_path):
    """The scaffolded folder references state_path -- it must not contain a
    copy of any file from the referenced directory (platform_registration_
    guidelines.md §5: a copy would silently drift from the original)."""
    state_dir = tmp_path / "my_rig"
    state_dir.mkdir()
    (state_dir / "state.json").write_text("{}")

    folder = scaffold_local_platform(str(state_dir), tmp_path / "platforms")

    assert not (folder / "state.json").exists()
    assert set(p.name for p in folder.iterdir()) == {"platform.py", "quam_source.json"}


def test_scaffold_iqcc_platform_writes_folder_pair_without_validation(tmp_path):
    dest = tmp_path / "platforms"

    folder = scaffold_iqcc_platform("arbel", dest, validate=False)

    assert folder == dest / "qibo-qm-iqcc-arbel"
    assert parse_platform_name(folder.name) == IqccSource(backend_name="arbel")

    platform_py = (folder / "platform.py").read_text()
    assert "create_iqcc(FOLDER.name, FOLDER)" in platform_py

    source_config = json.loads((folder / "quam_source.json").read_text())
    assert source_config["state_path"] == "quam_state"


def test_scaffold_iqcc_platform_validates_by_fetching(dummy_machine, monkeypatch, tmp_path):
    def fake_get_machine_from_iqcc(backend_name, api_token=None, quam_state_folder_path=None, quam_cls=None):
        assert backend_name == "arbel"
        return dummy_machine, None

    monkeypatch.setattr(
        "qiskit_qm_provider.providers.iqcc_cloud_provider.get_machine_from_iqcc",
        fake_get_machine_from_iqcc,
    )

    folder = scaffold_iqcc_platform("arbel", tmp_path / "platforms")

    # quam_state/ is created inside the scaffolded folder itself (the
    # default, per-machine state_path -- never a shared global path).
    assert (folder / "quam_state").is_dir()


def test_scaffold_iqcc_platform_validation_failure_propagates(monkeypatch, tmp_path):
    def fake_get_machine_from_iqcc(*args, **kwargs):
        raise ValueError("Error code 404 Not Found")

    monkeypatch.setattr(
        "qiskit_qm_provider.providers.iqcc_cloud_provider.get_machine_from_iqcc",
        fake_get_machine_from_iqcc,
    )

    with pytest.raises(ValueError, match="typo-name"):
        scaffold_iqcc_platform("typo-name", tmp_path / "platforms")

    # The folder is still written before validation runs -- a failed
    # validation reports the problem, it doesn't silently discard the
    # scaffolded files (re-running after fixing the typo just re-validates
    # the same folder).
    assert (tmp_path / "platforms" / "qibo-qm-iqcc-typo-name" / "platform.py").exists()
