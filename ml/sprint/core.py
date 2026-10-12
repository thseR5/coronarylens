"""Sprint machinery: per-fold learner caches and cheap, cross-fitted post-processing "procedures".

A *procedure* turns base-learner outputs into one calibrated 8-pattern distribution:
    learners -> combine (average | inner-learned simplex weights) -> optional vessel-count blend -> calibration.
All post-processing parameters (weights, blend lambda, calibrator) are fitted on inner out-of-fold (OOF)
predictions of the training part only, and applied to the held-out part. The held-out part is never seen.

For *selection* inside nested CV, procedures are scored on the inner OOF predictions with cross-fitting
(post-processing fitted on K-1 OOF folds, scored on the remaining one), so a procedure is never graded on the
data its own weights/calibrator were fitted to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ml import config
from ml.calibrate import Calibrator
from ml.evaluate import auc, average_precision, brier, multiclass_log_loss
from ml.features import FeatureEngineer
from ml.ingest import load_dataset, model_columns
from ml.joint import N_VESSELS, TARGETS, marginals
from ml.sprint.learners import make

N_PAT = 8
EPS = 1e-6


# ----------------------------------------------------------------------------- data
def training_split():
    """Training part of the locked split only (the test indices are never loaded here)."""
    fcfg, tcfg = config.features_cfg(), config.targets_cfg()
    ds = load_dataset(fcfg, tcfg)
    split = json.loads((config.ARTIFACTS / "split.json").read_text("utf-8"))
    tr = np.asarray(split["train"])
    X = ds.X.to_numpy(dtype=float)[tr]
    T = ds.T.iloc[tr].reset_index(drop=True)
    return X, T, model_columns(fcfg)


def labels(T):
    y = {"pattern": T["pattern_code"].to_numpy(int), "CAD": T["cad"].to_numpy(int)}
    for v in ("LAD", "LCX", "RCA"):
        y[v] = T[v].to_numpy(int)
    return y


def score(Y, P) -> dict:
    """Fold-level metrics of an (n, 8) pattern distribution."""
    m = marginals(P)
    out = {f"auc_{t}": auc(Y[t], m[t]) for t in TARGETS}
    out["brier_mean"] = float(np.mean([brier(Y[t], m[t]) for t in TARGETS]))
    out["pattern_log_loss"] = multiclass_log_loss(Y["pattern"], P)
    for t in TARGETS:
        out[f"brier_{t}"] = brier(Y[t], m[t])
        out[f"pr_auc_{t}"] = average_precision(Y[t], m[t])
    n_v = N_VESSELS[Y["pattern"]]
    out["auc_P_GE2"] = auc((n_v >= 2).astype(int), m["P_GE2"])
    out["auc_P_3VD"] = auc((n_v == 3).astype(int), m["P_3VD"])
    return out


# ----------------------------------------------------------------------------- feature variants
def transform_features(variant: str, Xtr, Xte, columns):
    """Feature variants, fitted on the training part only."""
    if variant == "base":
        return Xtr, Xte
    if variant == "fe":
        fe = FeatureEngineer(columns).fit(Xtr)
        return fe.transform(Xtr), fe.transform(Xte)
    raise KeyError(variant)


# ----------------------------------------------------------------------------- learner cache
@dataclass
class LearnerOut:
    oof: np.ndarray  # (n_train, 8) inner OOF predictions on the training part
    test: np.ndarray  # (n_test, 8) predictions of the learner refit on the whole training part


def learner_outputs(spec: dict, Xtr, ytr, Xte, seed: int, inner_folds: int) -> LearnerOut:
    """spec = {"learner": name, "params": {...}}; inner K-fold OOF + full refit."""
    oof = np.zeros((len(Xtr), N_PAT))
    skf = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    for a, b in skf.split(Xtr, ytr):
        oof[b] = make(spec["learner"], seed, **spec.get("params", {})).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])
    test = make(spec["learner"], seed, **spec.get("params", {})).fit(Xtr, ytr).predict_proba(Xte)
    return LearnerOut(oof, test)


def count_model_outputs(Xtr, ytr, Xte, seed: int, inner_folds: int, C: float = 0.1) -> LearnerOut:
    """Auxiliary model for the number of diseased vessels (0-3), multinomial ridge LR; returned as (n, 4)."""
    n_tr = N_VESSELS[ytr]

    def fit_pred(a, Xp):
        m = LogisticRegression(C=C, max_iter=5000, random_state=seed)
        sc = StandardScaler().fit(Xtr[a])
        m.fit(sc.transform(Xtr[a]), n_tr[a])
        out = np.zeros((len(Xp), 4))
        out[:, m.classes_] = m.predict_proba(sc.transform(Xp))
        return out

    oof = np.zeros((len(Xtr), 4))
    skf = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    for a, b in skf.split(Xtr, ytr):
        oof[b] = fit_pred(a, Xtr[b])
    return LearnerOut(oof, fit_pred(np.arange(len(Xtr)), Xte))


# ----------------------------------------------------------------------------- post-processing pieces
def _norm(P):
    P = np.clip(P, EPS, None)
    return P / P.sum(1, keepdims=True)


def learn_weights(Ps: list[np.ndarray], y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    """Simplex weights minimizing log-loss of the weighted average (softmax parametrisation)."""
    if len(Ps) == 1:
        return np.ones(1)

    def obj(z):
        w = softmax(z)
        P = _norm(sum(wi * p for wi, p in zip(w, Ps, strict=True)))
        return -np.log(P[np.arange(len(y)), y]).mean() + l2 * np.sum(z**2)

    return softmax(minimize(obj, np.zeros(len(Ps)), method="L-BFGS-B").x)


def count_blend(P: np.ndarray, Pc: np.ndarray, lam: float) -> np.ndarray:
    """Replace the joint's vessel-count marginal with a lambda-mixture of it and the count model's;
    the conditional P(pattern | count) of the joint model is kept, so the result is still one coherent
    distribution over the 8 patterns."""
    if lam == 0:
        return P
    out = np.empty_like(P)
    for n in range(4):
        cols = N_VESSELS == n
        pn = P[:, cols].sum(1, keepdims=True)
        target = (1 - lam) * pn + lam * Pc[:, [n]]
        out[:, cols] = P[:, cols] / np.maximum(pn, EPS) * target
    return _norm(out)


class Dirichlet:
    """Dirichlet calibration (Kull et al., NeurIPS 2019): softmax(W log p + b) with off-diagonal and
    intercept L2 regularisation (ODIR). Full 8x8 matrix, so it is regularised strongly toward identity."""

    def __init__(self, lam_off: float = 1.0, lam_b: float = 0.1):
        self.lam_off, self.lam_b = lam_off, lam_b

    def fit(self, P, y):
        L, y = np.log(_norm(P)), np.asarray(y, int)
        n, k = L.shape
        off = ~np.eye(k, dtype=bool)
        Y = np.zeros((n, k))
        Y[np.arange(n), y] = 1

        def obj(theta):  # regularised NLL and its analytic gradient
            W, b = theta[: k * k].reshape(k, k), theta[k * k:]
            Z = L @ W.T + b
            lp = log_softmax(Z, axis=1)
            G = (np.exp(lp) - Y) / n
            f = -(Y * lp).sum() / n + (self.lam_off * np.sum(W[off] ** 2) + self.lam_b * np.sum(b**2)) / k
            gW = G.T @ L + 2 * self.lam_off * np.where(off, W, 0.0) / k
            gb = G.sum(0) + 2 * self.lam_b * b / k
            return f, np.r_[gW.ravel(), gb]

        r = minimize(obj, np.r_[np.eye(k).ravel(), np.zeros(k)], jac=True, method="L-BFGS-B")
        self.W_, self.b_ = r.x[: k * k].reshape(k, k), r.x[k * k:]
        return self

    def transform(self, P):
        return np.exp(log_softmax(np.log(_norm(P)) @ self.W_.T + self.b_, axis=1))


def make_calibrator(name: str):
    return Dirichlet() if name == "dirichlet" else Calibrator(name)


# ----------------------------------------------------------------------------- procedures
@dataclass(frozen=True)
class Procedure:
    learners: tuple[str, ...]  # keys into the learner cache
    combine: str = "avg"  # avg | learned
    calib: str = "vector"  # none | temperature | vector | dirichlet
    count_lambda: float = 0.0  # 0 = off; >0 blends in the vessel-count model
    features: str = "base"  # base | fe

    @property
    def name(self) -> str:
        s = "+".join(self.learners) + f"|{self.combine}|{self.calib}"
        if self.count_lambda:
            s += f"|count{self.count_lambda:g}"
        if self.features != "base":
            s += f"|{self.features}"
        return s


@dataclass
class Fitted:
    weights: np.ndarray
    calibrator: object
    lam: float
    extras: dict = field(default_factory=dict)


def fit_post(proc: Procedure, oofs: list[np.ndarray], count_oof: np.ndarray | None, y: np.ndarray) -> Fitted:
    w = learn_weights(oofs, y) if proc.combine == "learned" else np.full(len(oofs), 1 / len(oofs))
    P = _norm(sum(wi * p for wi, p in zip(w, oofs, strict=True)))
    if proc.count_lambda:
        P = count_blend(P, count_oof, proc.count_lambda)
    cal = make_calibrator(proc.calib).fit(P, y)
    return Fitted(w, cal, proc.count_lambda)


def apply_post(fit: Fitted, preds: list[np.ndarray], count_pred: np.ndarray | None) -> np.ndarray:
    P = _norm(sum(wi * p for wi, p in zip(fit.weights, preds, strict=True)))
    if fit.lam:
        P = count_blend(P, count_pred, fit.lam)
    return _norm(fit.calibrator.transform(P))


def crossfit_oof_score(proc: Procedure, oofs, count_oof, y, seed: int, k: int = 5) -> float:
    """Pattern log-loss of the procedure on the inner OOF predictions, post-processing cross-fitted."""
    P = np.zeros((len(y), N_PAT))
    skf = StratifiedKFold(k, shuffle=True, random_state=seed + 7)
    for a, b in skf.split(np.zeros(len(y)), y):
        f = fit_post(proc, [o[a] for o in oofs], None if count_oof is None else count_oof[a], y[a])
        P[b] = apply_post(f, [o[b] for o in oofs], None if count_oof is None else count_oof[b])
    return multiclass_log_loss(y, P)
