"""Candidate base learners for the predictive-performance sprint.

Every learner maps (X, pattern codes 0..7) -> 8-class probabilities aligned to all 8 vessel patterns, so it can
plug into the joint (coherent) model: all served outputs stay marginals of one distribution over patterns.

Factories take (seed, **params) and return an object with fit(X, y) / predict_proba(X) -> (n, 8).
Optional dependencies (catboost, interpret, tabpfn) are imported lazily; an ImportError marks the candidate as
unavailable (logged in docs/EXPERIMENTS.md), never silently skipped.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

N_PATTERNS = 8


def align(proba: np.ndarray, classes, n: int = N_PATTERNS) -> np.ndarray:
    out = np.zeros((proba.shape[0], n))
    out[:, np.asarray(classes, dtype=int)] = proba
    s = out.sum(1, keepdims=True)
    return out / np.where(s > 0, s, 1.0)


class Aligned:
    """Wrap any sklearn-style classifier so predict_proba always has 8 aligned columns."""

    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.est.fit(np.asarray(X, float), np.asarray(y, int))
        return self

    def predict_proba(self, X):
        return align(self.est.predict_proba(np.asarray(X, float)), self.est.classes_)


# ----------------------------------------------------------------------------- sklearn
def lr_en(seed, C=0.3, l1_ratio=0.5):
    lr = LogisticRegression(C=C, l1_ratio=l1_ratio, solver="saga", max_iter=5000, tol=1e-4, random_state=seed)
    return Aligned(Pipeline([("scale", StandardScaler()), ("clf", lr)]))


def lr_l2(seed, C=0.1):
    """Multinomial ridge LR = MAP estimate under a zero-mean Gaussian prior (variance ~ C) on standardized
    coefficients: a 'Bayesian' logistic regression with a skeptical prior."""
    lr = LogisticRegression(C=C, max_iter=5000, random_state=seed)
    return Aligned(Pipeline([("scale", StandardScaler()), ("clf", lr)]))


def rf(seed, n_estimators=500, min_samples_leaf=3, max_features="sqrt"):
    return Aligned(RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                          max_features=max_features, random_state=seed, n_jobs=1))


def et(seed, n_estimators=500, min_samples_leaf=3, max_features="sqrt"):
    return Aligned(ExtraTreesClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                        max_features=max_features, random_state=seed, n_jobs=1))


# ----------------------------------------------------------------------------- boosting
def lgbm(seed, n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10, reg_lambda=1.0):
    from lightgbm import LGBMClassifier

    return Aligned(LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                                  min_child_samples=min_child_samples, reg_lambda=reg_lambda, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=1,
                                  verbose=-1, deterministic=True, force_row_wise=True))


def catboost(seed, iterations=400, depth=4, learning_rate=0.03, l2_leaf_reg=5.0):
    from catboost import CatBoostClassifier

    return Aligned(CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=learning_rate,
                                      l2_leaf_reg=l2_leaf_reg, loss_function="MultiClass", random_seed=seed,
                                      thread_count=1, verbose=False, allow_writing_files=False))


# ----------------------------------------------------------------------------- glass-box / foundation
def ebm(seed, max_bins=32, interactions=0, outer_bags=8, learning_rate=0.01):
    from interpret.glassbox import ExplainableBoostingClassifier

    return Aligned(ExplainableBoostingClassifier(max_bins=max_bins, interactions=interactions, outer_bags=outer_bags,
                                                 learning_rate=learning_rate, random_state=seed, n_jobs=1))


def tabpfn(seed, n_estimators=4):
    from tabpfn import TabPFNClassifier

    return Aligned(TabPFNClassifier(n_estimators=n_estimators, random_state=seed, device="cpu"))


LEARNERS: dict[str, Callable] = {
    "lr_en": lr_en, "lr_l2": lr_l2, "rf": rf, "et": et,
    "lgbm": lgbm, "catboost": catboost, "ebm": ebm, "tabpfn": tabpfn,
}


def make(name: str, seed: int, **params):
    if name not in LEARNERS:
        from ml.sprint.learners2 import LEARNERS2

        return LEARNERS2[name](seed, **params)
    return LEARNERS[name](seed, **params)
