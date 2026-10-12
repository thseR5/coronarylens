"""Vectorized re-implementation of libsvm's probability prediction for a fitted sklearn RBF SVC.

libsvm evaluates the kernel row by row (~220 ms for the ~5,000 coalition rows of one explanation). Here the RBF
kernel against the support vectors is one matrix product, and the one-vs-one decision values, Platt sigmoids and
the pairwise-coupling iteration (Wu, Lin & Weng 2004; libsvm `multiclass_probability`, including its per-row
stopping rule) run on all rows at once. Output matches sklearn's predict_proba (tested to 1e-6).
"""

from __future__ import annotations

import numpy as np


class FastSVC:
    def __init__(self, svc):
        if svc.kernel != "rbf":
            raise ValueError("FastSVC supports the RBF kernel only")
        self.classes_ = svc.classes_
        self.k = len(svc.classes_)
        self.sv = np.asarray(svc.support_vectors_, float)
        self.sv_sq = (self.sv**2).sum(1)
        self.gamma = float(svc._gamma)
        self.dual = np.asarray(svc.dual_coef_, float)
        self.intercept = np.asarray(svc.intercept_, float)
        self.probA = np.asarray(svc.probA_, float)
        self.probB = np.asarray(svc.probB_, float)
        starts = np.r_[0, np.cumsum(svc.n_support_)]
        self.sv_slices = [slice(starts[c], starts[c + 1]) for c in range(self.k)]
        self.pairs = [(i, j) for i in range(self.k) for j in range(i + 1, self.k)]

    def decision_ovo(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        d2 = (X**2).sum(1)[:, None] + self.sv_sq[None, :] - 2.0 * X @ self.sv.T
        K = np.exp(-self.gamma * np.maximum(d2, 0.0))
        out = np.empty((len(X), len(self.pairs)))
        for p, (i, j) in enumerate(self.pairs):
            si, sj = self.sv_slices[i], self.sv_slices[j]
            out[:, p] = K[:, si] @ self.dual[j - 1, si] + K[:, sj] @ self.dual[i, sj] + self.intercept[p]
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dec = self.decision_ovo(X)
        n, k = len(dec), self.k
        f = dec * self.probA + self.probB  # libsvm sigmoid_predict
        pos = f >= 0
        r_pair = np.where(pos, np.exp(-np.abs(f)) / (1 + np.exp(-np.abs(f))), 1 / (1 + np.exp(-np.abs(f))))
        r_pair = np.clip(r_pair, 1e-7, 1 - 1e-7)
        r = np.zeros((n, k, k))
        for p, (i, j) in enumerate(self.pairs):
            r[:, i, j] = r_pair[:, p]
            r[:, j, i] = 1 - r_pair[:, p]
        # libsvm multiclass_probability, vectorized over rows (converged rows are frozen, as in libsvm's break)
        Q = -r.transpose(0, 2, 1) * r  # Q[t][j] = -r[j][t] * r[t][j]
        diag = (r.transpose(0, 2, 1) ** 2).sum(2) - (np.diagonal(r, axis1=1, axis2=2) ** 2)  # sum_{j!=t} r[j][t]^2
        idx = np.arange(k)
        Q[:, idx, idx] = diag
        p = np.full((n, k), 1.0 / k)
        active = np.ones(n, dtype=bool)
        eps = 0.005 / k
        for _ in range(max(100, k)):
            Qp = np.einsum("ntj,nj->nt", Q, p)
            pQp = (p * Qp).sum(1)
            err = np.abs(Qp - pQp[:, None]).max(1)
            active &= ~(err < eps)
            if not active.any():
                break
            a = np.flatnonzero(active)
            pa, Qpa, pQpa, Qa = p[a], Qp[a], pQp[a], Q[a]
            for t in range(k):
                qtt = Qa[:, t, t]
                diff = (-Qpa[:, t] + pQpa) / qtt
                pa[:, t] += diff
                pQpa = (pQpa + diff * (diff * qtt + 2 * Qpa[:, t])) / (1 + diff) ** 2
                Qpa = (Qpa + diff[:, None] * Qa[:, t, :]) / (1 + diff)[:, None]
                pa /= (1 + diff)[:, None]
            p[a] = pa
        return p
