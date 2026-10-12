"""Operating thresholds chosen on out-of-fold predictions (never on test data)."""

from __future__ import annotations

import numpy as np


def max_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(p):
        pred = p >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1 + 1e-12:
            best_f1, best_t = f1, float(t)
    return best_t


def min_sensitivity_threshold(y: np.ndarray, p: np.ndarray, sens: float = 0.90) -> float:
    """Largest threshold whose sensitivity is still >= sens."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pos = np.sort(p[y == 1])[::-1]
    if pos.size == 0:
        return 0.5
    k = int(np.ceil(sens * pos.size))  # need at least k positives at or above threshold
    return float(pos[k - 1])


def fit_thresholds(Y: dict[str, np.ndarray], Pm: dict[str, np.ndarray], targets: list[str],
                   sens: float = 0.90) -> dict[str, dict[str, float]]:
    return {t: {"f1": max_f1_threshold(Y[t], Pm[t]), "screening": min_sensitivity_threshold(Y[t], Pm[t], sens)}
            for t in targets}
