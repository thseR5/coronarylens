"""Regression tests for the judge-review fixes: status chips, screening thresholds, centred intervals."""

import numpy as np

from ml import conformal
from ml.thresholds import fit_thresholds
from ml.uncertainty import centered_intervals


def test_status_never_contradicts_threshold_decision():
    q = {"0": 0.81, "1": 0.34}  # CAD-like class-conditional quantiles (sets disagree with a 0.44 threshold)
    for p in np.linspace(0.01, 0.99, 197):
        positive = p >= 0.443
        s = conformal.reconciled_status(float(p), q, positive)
        assert not (s == "likely" and not positive)
        assert not (s == "unlikely" and positive)


def test_screening_threshold_never_above_balanced():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.35 * y + rng.normal(0.4, 0.2, 300), 0, 1)
    thr = fit_thresholds({"CAD": y}, {"CAD": p}, ["CAD"])["CAD"]
    assert thr["screening"] <= thr["f1"]


def test_centred_interval_contains_point_and_stays_in_unit_range():
    rng = np.random.default_rng(1)
    Pb = rng.dirichlet(np.ones(8), size=(50, 20))  # (B, n, 8)
    point = {"CAD": rng.uniform(0, 1, 20)}
    cen, raw = centered_intervals(Pb, point)
    lo, hi = cen["CAD"][:, 0], cen["CAD"][:, 1]
    assert np.all(lo <= point["CAD"] + 1e-12) and np.all(point["CAD"] <= hi + 1e-12)
    assert np.all((lo >= 0) & (hi <= 1))
    assert raw["CAD"].shape == (20, 2)
