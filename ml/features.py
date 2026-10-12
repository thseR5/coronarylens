"""Engineered features as a stateless sklearn transformer (used inside the pipeline, i.e. inside CV folds).

Kept in the served pipeline only if nested CV supports it (see ml/train.py and docs/notes/ml.md).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ml import config
from ml.ingest import model_columns

ENGINEERED = ["egfr", "tg_hdl", "ldl_hdl", "nlr", "risk_factor_count", "ecg_abnormality_count"]


def egfr_ckd_epi_2021(cr: np.ndarray, age: np.ndarray, male: np.ndarray) -> np.ndarray:
    """Race-free CKD-EPI 2021 creatinine equation (mL/min/1.73 m^2); cr in mg/dL."""
    male = male >= 0.5
    kappa = np.where(male, 0.9, 0.7)
    alpha = np.where(male, -0.302, -0.241)
    r = cr / kappa
    return (
        142.0
        * np.minimum(r, 1.0) ** alpha
        * np.maximum(r, 1.0) ** -1.200
        * 0.9938**age
        * np.where(male, 1.0, 1.012)
    )


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Append engineered columns to the encoded model matrix (numpy, columns = model_columns())."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X, y=None):
        cols = self.columns or model_columns(config.features_cfg())
        self.idx_ = {c: i for i, c in enumerate(cols)}
        self.n_features_in_ = len(cols)
        return self

    def _c(self, X, name):
        return X[:, self.idx_[name]]

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        c = self._c
        hdl = np.maximum(c(X, "hdl"), 1e-6)
        lymph = np.maximum(c(X, "lymph"), 1e-6)
        rf = sum(c(X, k) for k in ["dm", "htn", "current_smoker", "fh", "dlp", "obesity"])
        bbb_cols = [k for k in self.idx_ if k.startswith("bbb_")]
        ecg = sum(c(X, k) for k in ["q_wave", "st_elevation", "st_depression", "t_inversion", "lvh",
                                    "poor_r_progression"])
        ecg = ecg + np.clip(sum(c(X, k) for k in bbb_cols), 0, 1)
        eng = np.column_stack(
            [
                egfr_ckd_epi_2021(c(X, "cr"), c(X, "age"), c(X, "sex_male")),
                c(X, "tg") / hdl,
                c(X, "ldl") / hdl,
                c(X, "neut") / lymph,
                rf,
                ecg,
            ]
        )
        return np.hstack([X, eng])

    def get_feature_names_out(self, input_features=None):
        cols = self.columns or model_columns(config.features_cfg())
        return np.array(list(cols) + ENGINEERED)
