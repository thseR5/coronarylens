"""Mondrian (class-conditional) conformal prediction for each binary target, 90% target coverage.

Non-conformity score for label c: s = 1 - p_c (p_1 = p, p_0 = 1 - p). Per class c the quantile q_c is the
ceil((n_c + 1)(1 - alpha))-th smallest score among calibration points of class c. Calibration scores are the
out-of-fold predictions of a K-fold refit on the training split (cross-conformal style; every training
point is used once as calibration). Set = {c : 1 - p_c <= q_c}.
Status: {1} likely, {0,1} uncertain, {0} unlikely; an empty set is reported as "uncertain" (conservative).
"""

from __future__ import annotations

import numpy as np


def fit_quantiles(y: np.ndarray, p: np.ndarray, alpha: float = 0.10) -> dict[str, float]:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    q = {}
    for c in (0, 1):
        s = np.sort(1 - (p[y == c] if c == 1 else 1 - p[y == c]))
        n = s.size
        k = int(np.ceil((n + 1) * (1 - alpha)))
        q[str(c)] = 1.0 if k > n else float(s[k - 1])
    return q


def prediction_set(p: np.ndarray, q: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    in1 = (1 - p) <= q["1"] + 1e-12
    in0 = p <= q["0"] + 1e-12
    return in0, in1


def status(p: float, q: dict[str, float]) -> str:
    in0, in1 = prediction_set(np.array([p]), q)
    if in1[0] and not in0[0]:
        return "likely"
    if in0[0] and not in1[0]:
        return "unlikely"
    return "uncertain"


def reconciled_status(p: float, q: dict[str, float], positive: bool) -> str:
    """Status shown in the UI. "likely"/"unlikely" only when the 90% conformal set is a single outcome AND it
    agrees with the operating-threshold decision; otherwise "uncertain". (Mondrian sets are class-conditional and
    are not aligned with the max-F1 threshold, so the two can disagree; the UI must never show a contradiction.)"""
    s = status(p, q)
    if s == "likely" and positive:
        return "likely"
    if s == "unlikely" and not positive:
        return "unlikely"
    return "uncertain"


def displayed_status_rates(p, q, thr: float) -> dict:
    p = np.asarray(p, dtype=float)
    st = [reconciled_status(float(x), q, bool(x >= thr)) for x in p]
    return {k: float(np.mean([s == k for s in st])) for k in ("likely", "uncertain", "unlikely")}


def evaluate(y, p, q) -> dict:
    y = np.asarray(y, dtype=int)
    in0, in1 = prediction_set(p, q)
    covered = np.where(y == 1, in1, in0)
    size = in0.astype(int) + in1.astype(int)
    return {"coverage": float(covered.mean()),
            "coverage_class1": float(covered[y == 1].mean()) if (y == 1).any() else None,
            "coverage_class0": float(covered[y == 0].mean()) if (y == 0).any() else None,
            "mean_set_size": float(size.mean()), "frac_singleton": float((size == 1).mean()),
            "frac_uncertain": float((size == 2).mean()), "frac_empty": float((size == 0).mean()), "n": len(y)}
