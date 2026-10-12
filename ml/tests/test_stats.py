"""Nadeau-Bengio corrected resampled t-test."""

import numpy as np
from scipy import stats

from ml.sprint.stats import corrected, paired


def test_corrected_variance_formula():
    rng = np.random.default_rng(0)
    d = rng.normal(0.01, 0.02, 50)
    c = corrected(d, n_train=200, n_test=50)
    s2 = d.var(ddof=1)
    assert np.isclose(c["se_corrected"] ** 2, (1 / 50 + 50 / 200) * s2)
    assert c["se_corrected"] > c["se_naive"]
    assert np.isclose(c["p"], 2 * stats.t.sf(abs(c["t"]), 49))


def test_paired_verdicts():
    rows_better = [{"a": {"m": 1.0 - 0.1}, "b": {"m": 1.0}} for _ in range(10)]
    rows_better[0]["a"]["m"] = 0.95  # non-zero variance
    r = paired(rows_better, "a", "b", "m", lower_is_better=True, n_train=100, n_test=25)
    assert r["verdict"] == "significant improvement" and r["folds_improved"] == 10
    rng = np.random.default_rng(1)
    noisy = [{"a": {"m": float(x)}, "b": {"m": 0.0}} for x in rng.normal(0.001, 0.05, 30)]
    r2 = paired(noisy, "a", "b", "m", lower_is_better=False, n_train=100, n_test=25)
    assert r2["p"] > 0.05 and "not" in r2["verdict"]
