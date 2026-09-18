"""Tests for qibo_qm_provider.qibolab_bridge.qua_acquisition.

``collect``/``split`` are pure numpy shape math (no QUA/QuAM dependency),
so they're tested directly here with synthetic arrays -- the riskiest piece
flagged in the Option-A convergence plan, since a wrong axis order/shuffle
compiles and runs fine, it just silently returns wrong-shaped or
wrong-correspondence data. ``IntegratedAcquisition.declare/measure/download``
(which do need a real QUA program scope) are exercised end to end via
``test_qua_macros.py``'s acquisition tests instead.
"""

import numpy as np
import pytest

from qibo_qm_provider.qibolab_bridge.qua_acquisition import (
    IntegratedAcquisition,
    ShotsAcquisition,
    collect,
    fetch_results,
    split,
)


class _FakeResult:
    def __init__(self, data):
        self._data = data

    def fetch_all(self):
        return self._data


class _FakeResultHandles:
    """Maps a saved-stream name to pre-seeded fake data -- mirrors
    ``test_iqcc_qm_controller.py``'s ``_FakeCloudResultHandles``, but keyed
    by name rather than returning the same data for every handle, since
    ``fetch()``/``fetch_results`` correctness here specifically depends on
    ``_I``/``_Q`` resolving to the right arrays.
    """

    def __init__(self, data_by_name):
        self._data_by_name = data_by_name

    def get(self, name):
        return _FakeResult(self._data_by_name[name])


def test_collect_and_split_recover_per_readout_arrays_for_multiplexed_2d_sweep():
    """Two multiplexed readout occurrences (npulses=2) under a 2D sweep
    (outer len 3, inner len 4) must, after collect()+split(), reproduce
    qibolab's own documented per-readout shape
    (``ExecutionParameters.results_shape``: ``bins(sweepers) + (2,)`` for
    ``INTEGRATION``, i.e. ``(3, 4, 2)`` here) -- and, more importantly,
    must not silently shuffle which value belongs to which readout
    occurrence or which sweep point.
    """
    n_a, n_b, npulses = 3, 4, 2
    # Distinct values everywhere so a wrong axis order/shuffle is caught,
    # not just a wrong shape.
    i = np.arange(n_a * n_b * npulses).reshape(n_a, n_b, npulses).astype(float)
    q = i + 1000.0

    signal = collect(i, q, npulses)
    assert signal.shape == (n_a, n_b, 2, npulses)

    per_readout = split(signal, npulses)
    assert len(per_readout) == npulses
    for k, arr in enumerate(per_readout):
        assert arr.shape == (n_a, n_b, 2)
        for a in range(n_a):
            for b in range(n_b):
                assert arr[a, b, 0] == i[a, b, k]
                assert arr[a, b, 1] == q[a, b, k]


def test_collect_and_split_single_readout_no_extra_axis():
    """``npulses == 1``: no extra "which readout" axis -- IQ is simply the
    last axis, and ``split()`` is a no-op wrapper (per its own docstring)."""
    n_a = 5
    i = np.arange(n_a).astype(float)
    q = i + 100.0

    signal = collect(i, q, npulses=1)
    assert signal.shape == (n_a, 2)
    assert np.array_equal(signal[:, 0], i)
    assert np.array_equal(signal[:, 1], q)

    per_readout = split(signal, npulses=1)
    assert len(per_readout) == 1
    assert per_readout[0] is signal


def test_integrated_acquisition_fetch_and_fetch_results_split_by_key():
    """``IntegratedAcquisition.fetch``/``fetch_results`` end to end for a
    multiplexed group (npulses=2, no sweep dims): synthetic per-shot I/Q
    data, keyed by this group's ``name`` (as ``download()`` would have
    saved it under), must come back split into one array per PulseId key,
    in ``keys`` order -- and single-key groups must collapse to a bare
    array (qibolab's own ``fetch_results`` back-compat behavior, ported
    unchanged).
    """
    group = IntegratedAcquisition(operation="readout", element="qA1|resonator")
    key_a, key_b = "acq-a", "acq-b"
    group.keys = [key_a, key_b]

    nshots = 5
    i = np.arange(nshots * 2).reshape(nshots, 2).astype(float)
    q = i + 100.0
    handles = _FakeResultHandles({f"{group.name}_I": i, f"{group.name}_Q": q})

    per_key = group.fetch(handles)
    assert len(per_key) == 2
    for k, arr in enumerate(per_key):
        assert arr.shape == (nshots, 2)
        assert np.array_equal(arr[:, 0], i[:, k])
        assert np.array_equal(arr[:, 1], q[:, k])

    results = fetch_results(handles, [group])
    assert set(results) == {key_a, key_b}
    assert np.array_equal(results[key_a], per_key[0])
    assert np.array_equal(results[key_b], per_key[1])


def test_shots_acquisition_requires_threshold():
    """A group built for DISCRIMINATION with no threshold (e.g. an
    un-calibrated readout pulse) must fail loudly at construction, not
    silently compare against ``None`` deep inside QUA."""
    with pytest.raises(ValueError, match="threshold"):
        ShotsAcquisition(operation="readout", element="qA1|resonator", threshold=None)


def test_shots_acquisition_arithmetic_matches_qibolab():
    """Standalone numeric cross-check (flagged in the Option-A convergence
    plan as required before trusting DISCRIMINATION on hardware): this
    class's classification formula must agree with qibolab's own
    ``ShotsAcquisition.measure()`` (``i*cos(angle) - q*sin(angle) >
    threshold``) for the same threshold/angle and a spread of I/Q values --
    checked directly in numpy, independent of any QUA program scope, since
    the formula itself (not the QUA plumbing around it) is what must match.
    """
    from qibolab._core.instruments.qm.program.acquisition import ShotsAcquisition as QibolabShotsAcquisition

    threshold, angle = 0.001, 1.2
    ours = ShotsAcquisition(operation="readout", element="qA1|resonator", threshold=threshold, angle=angle)
    theirs = QibolabShotsAcquisition(operation="readout", element="qA1|resonator", average=False, threshold=threshold, angle=angle)

    assert ours.cos == pytest.approx(theirs.cos)
    assert ours.sin == pytest.approx(theirs.sin)

    rng = np.random.default_rng(0)
    i_vals = rng.uniform(-1, 1, size=50)
    q_vals = rng.uniform(-1, 1, size=50)
    ours_shots = (i_vals * ours.cos - q_vals * ours.sin > threshold).astype(int)
    theirs_shots = (i_vals * theirs.cos - q_vals * theirs.sin > threshold).astype(int)
    assert np.array_equal(ours_shots, theirs_shots)


def test_shots_acquisition_fetch_and_fetch_results_split_by_key():
    """``ShotsAcquisition.fetch``/``fetch_results`` end to end for a
    multiplexed group (npulses=2): synthetic per-shot classified-state data,
    keyed by ``{name}_shots`` (as ``download()`` would have saved it under),
    must come back split into one array per PulseId key, in ``keys`` order.
    """
    group = ShotsAcquisition(operation="readout", element="qA1|resonator", threshold=0.001, angle=1.2)
    key_a, key_b = "acq-a", "acq-b"
    group.keys = [key_a, key_b]

    nshots = 5
    shots = np.arange(nshots * 2).reshape(nshots, 2) % 2
    handles = _FakeResultHandles({f"{group.name}_shots": shots})

    per_key = group.fetch(handles)
    assert len(per_key) == 2
    for k, arr in enumerate(per_key):
        assert np.array_equal(arr, shots[:, k])

    results = fetch_results(handles, [group])
    assert set(results) == {key_a, key_b}
    assert np.array_equal(results[key_a], per_key[0])
    assert np.array_equal(results[key_b], per_key[1])


def test_fetch_results_collapses_single_key_group():
    """A group with exactly one readout occurrence must return a bare
    array for its key, not a length-1 list -- matches qibolab's own
    ``fetch_results``' back-compat collapsing."""
    group = IntegratedAcquisition(operation="readout", element="qA1|resonator")
    group.keys = ["only-key"]

    i = np.array([0.1, 0.2, 0.3])
    q = np.array([1.1, 1.2, 1.3])
    handles = _FakeResultHandles({f"{group.name}_I": i, f"{group.name}_Q": q})

    results = fetch_results(handles, [group])
    assert set(results) == {"only-key"}
    assert results["only-key"].shape == (3, 2)
