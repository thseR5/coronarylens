"""Post-hoc calibration of the 8-class joint output, fitted on inner out-of-fold predictions.

- none:        identity
- temperature: softmax(log p / T), one parameter
- vector:      softmax(w * log p + b), diagonal w and bias b (2 x 8 parameters, light L2 towards identity)
Calibration is applied before marginalization, so coherence (P(CAD) >= max vessel) is preserved.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_softmax

EPS = 1e-6


def _logp(P: np.ndarray) -> np.ndarray:
    P = np.clip(np.asarray(P, dtype=float), EPS, 1.0)
    return np.log(P / P.sum(1, keepdims=True))


def _nll(logits: np.ndarray, y: np.ndarray) -> float:
    return float(-log_softmax(logits, axis=1)[np.arange(len(y)), y].mean())


class Calibrator:
    def __init__(self, method: str = "temperature", l2: float = 1e-2):
        if method not in ("none", "temperature", "vector"):
            raise ValueError(method)
        self.method = method
        self.l2 = l2

    def fit(self, P: np.ndarray, y: np.ndarray) -> Calibrator:
        L, y = _logp(P), np.asarray(y, dtype=int)
        k = L.shape[1]
        self.T_, self.w_, self.b_ = 1.0, np.ones(k), np.zeros(k)
        if self.method == "temperature":
            r = minimize_scalar(lambda lt: _nll(L / np.exp(lt), y), bounds=(-3, 3), method="bounded",
                                options={"xatol": 1e-8})
            self.T_ = float(np.exp(r.x))
        elif self.method == "vector":
            def obj(theta):
                w, b = theta[:k], theta[k:]
                return _nll(L * w + b, y) + self.l2 * (np.sum((w - 1) ** 2) + np.sum(b**2))

            r = minimize(obj, np.r_[np.ones(k), np.zeros(k)], method="L-BFGS-B")
            self.w_, self.b_ = r.x[:k], r.x[k:]
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        if self.method == "none":
            P = np.asarray(P, dtype=float)
            return P / P.sum(1, keepdims=True)
        L = _logp(P)
        Z = L / self.T_ if self.method == "temperature" else L * self.w_ + self.b_
        return np.exp(log_softmax(Z, axis=1))

    def params(self) -> dict:
        return {"method": self.method, "T": self.T_, "w": list(map(float, self.w_)), "b": list(map(float, self.b_))}
