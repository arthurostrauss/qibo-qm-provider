"""Tests for qibo_qm_provider.qibolab_bridge.platform_naming."""

import pytest

from qibo_qm_provider.exceptions import InvalidPlatformName
from qibo_qm_provider.qibolab_bridge.platform_naming import (
    IqccSource,
    LocalSource,
    parse_platform_name,
)


def test_parses_local():
    assert parse_platform_name("qibo-qm-local") == LocalSource()


def test_parses_iqcc_backend_name():
    assert parse_platform_name("qibo-qm-iqcc-arbel") == IqccSource(backend_name="arbel")


def test_parses_hyphenated_iqcc_backend_name():
    """Locks in fixed-prefix stripping, not a positional hyphen split --
    an IQCC backend name may itself contain hyphens."""
    assert parse_platform_name("qibo-qm-iqcc-my-hyphenated-name") == IqccSource(
        backend_name="my-hyphenated-name"
    )


@pytest.mark.parametrize(
    "name",
    [
        "qibolab",
        "qibo-qm",
        "local",
        "arbel",
        "qibo-qm-iqcc-",
        "qibo-qm-weird",
        "qibo-qm-local-extra",
    ],
)
def test_rejects_invalid_names(name):
    with pytest.raises(InvalidPlatformName):
        parse_platform_name(name)
