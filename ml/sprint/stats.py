"""Nadeau-Bengio corrected resampled t-test (Machine Learning 52, 2003).

Repeated k-fold CV folds share training data, so the J fold-level differences are positively correlated and the plain
SE (s / sqrt(J)) is too small. The corrected variance of the mean difference is (1/J + n_test/n_train) * s^2;
t = mean / sqrt(corrected variance) with J - 1 degrees of freedom. The same correction gives honest CIs for
CV means (Bouckaert & Frank 2004 used the same factor for repeated CV).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def corrected(values, n_train: float, n_test: float, alpha: float = 0.05) -> dict:
    """values: J fold-level quantities (differences for a paired test, or a metric for a CI)."""
    d = np.asarray(values, float)
    d = d[~np.isnan(d)]
    J = d.size
    mean, s2 = float(d.mean()), float(d.var(ddof=1))
    var_c = (1.0 / J + n_test / n_train) * s2
    se_c = float(np.sqrt(var_c))
    t = mean / se_c if se_c > 0 else float("inf") * np.sign(mean)
    p = float(2 * stats.t.sf(abs(t), J - 1)) if np.isfinite(t) else 0.0
    q = float(stats.t.ppf(1 - alpha / 2, J - 1))
    return {"mean": mean, "sd": float(np.sqrt(s2)), "se_naive": float(np.sqrt(s2 / J)), "se_corrected": se_c,
            "t": float(t), "df": J - 1, "p": p, "ci95": [mean - q * se_c, mean + q * se_c], "J": J,
            "n_train": float(n_train), "n_test": float(n_test)}


def paired(fold_rows: list[dict], key_a: str, key_b: str, metric: str, lower_is_better: bool,
           n_train: float, n_test: float, alpha: float = 0.05) -> dict:
    """Corrected paired test of a - b on `metric` over aligned folds; adds folds-improved and a plain verdict."""
    d = [r[key_a][metric] - r[key_b][metric] for r in fold_rows]
    res = corrected(d, n_train, n_test, alpha)
    arr = np.asarray(d, float)
    res["folds_improved"] = int(np.sum(arr < 0) if lower_is_better else np.sum(arr > 0))
    res["n_folds"] = int(arr.size)
    res["better_is"] = "lower" if lower_is_better else "higher"
    good = res["mean"] < 0 if lower_is_better else res["mean"] > 0
    if res["p"] < alpha:
        res["verdict"] = "significant improvement" if good else "significant worsening"
    else:
        res["verdict"] = "small, not statistically significant" if good else "no improvement (not significant)"
    return res
