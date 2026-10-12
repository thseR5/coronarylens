"""Applicability domain: Mahalanobis distance (Ledoit-Wolf shrunk covariance) on standardized features.

Threshold = 99th percentile of *out-of-fold* training scores (each training patient scored by a model fitted
without it), so the in-domain rate on new data from the same population is close to 99%.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class DomainModel:
    def __init__(self, quantile: float = 0.99, seed: int = 42):
        self.quantile = quantile
        self.seed = seed

    @staticmethod
    def _fit_core(X):
        sc = StandardScaler().fit(X)
        Z = sc.transform(X)
        lw = LedoitWolf().fit(Z)
        return sc, lw

    @staticmethod
    def _score(sc, lw, X):
        Z = sc.transform(np.asarray(X, dtype=float))
        return np.sqrt(np.maximum(lw.mahalanobis(Z), 0)) / np.sqrt(Z.shape[1])

    def fit(self, X) -> DomainModel:
        X = np.asarray(X, dtype=float)
        oof = np.zeros(len(X))
        for tr, va in KFold(5, shuffle=True, random_state=self.seed).split(X):
            sc, lw = self._fit_core(X[tr])
            oof[va] = self._score(sc, lw, X[va])
        self.sc_, self.lw_ = self._fit_core(X)
        self.threshold_ = float(np.quantile(oof, self.quantile))
        self.oof_scores_ = oof
        return self

    def score(self, X) -> np.ndarray:
        return self._score(self.sc_, self.lw_, X)
