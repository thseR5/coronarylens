"""Marginalization of the joint 8-pattern distribution into the served outputs.

Pattern code = sum(bit_i * 2^(k-1-i)) with bit order from targets.yaml (LAD, LCX, RCA).
P(vessel) = sum of patterns with that bit; P(CAD) = 1 - P(000) >= max vessel by construction.
"""

from __future__ import annotations

import numpy as np

K_VESSELS = 3
N_PATTERNS = 2**K_VESSELS
VESSELS = ["LAD", "LCX", "RCA"]
TARGETS = ["CAD", "LAD", "LCX", "RCA"]
EXPLAIN_OUTPUTS = ["CAD", "LAD", "LCX", "RCA", "P_GE2"]

# bit matrix (8, 3): BITS[code, i] = 1 if vessel i stenotic in pattern `code`
BITS = np.array([[(c >> (K_VESSELS - 1 - i)) & 1 for i in range(K_VESSELS)] for c in range(N_PATTERNS)])
N_VESSELS = BITS.sum(1)
PATTERNS = [format(c, f"0{K_VESSELS}b") for c in range(N_PATTERNS)]


def marginals(P: np.ndarray) -> dict[str, np.ndarray]:
    """(n, 8) pattern probabilities -> dict of (n,) marginals."""
    P = np.asarray(P, dtype=float)
    out = {v: P @ BITS[:, i] for i, v in enumerate(VESSELS)}
    out["CAD"] = 1.0 - P[:, 0]
    out["P_GE2"] = P[:, N_VESSELS >= 2].sum(1)
    out["P_3VD"] = P[:, N_VESSELS == 3].sum(1)
    for n in range(K_VESSELS + 1):
        out[f"N{n}"] = P[:, N_VESSELS == n].sum(1)
    return out


def explain_matrix(P: np.ndarray) -> np.ndarray:
    """(n, 8) -> (n, 5) in EXPLAIN_OUTPUTS order."""
    m = marginals(P)
    return np.column_stack([m[k] for k in EXPLAIN_OUTPUTS])


def patterns_from_independent(p_vessels: np.ndarray) -> np.ndarray:
    """(n, 3) independent vessel probabilities -> (n, 8) product distribution (conditional independence)."""
    pv = np.clip(np.asarray(p_vessels, dtype=float), 0, 1)
    P = np.ones((pv.shape[0], N_PATTERNS))
    for c in range(N_PATTERNS):
        for i in range(K_VESSELS):
            P[:, c] *= pv[:, i] if BITS[c, i] else 1 - pv[:, i]
    return P


def align_proba(proba: np.ndarray, classes: np.ndarray, n_classes: int = N_PATTERNS) -> np.ndarray:
    """Place predict_proba columns into a fixed (n, n_classes) layout (classes missing in a fold -> 0)."""
    out = np.zeros((proba.shape[0], n_classes))
    out[:, np.asarray(classes, dtype=int)] = proba
    return out
