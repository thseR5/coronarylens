"""Second batch of sprint learners (all output 8 aligned pattern probabilities, so they plug into the joint model).

- bayes_lr   (d) multinomial logistic regression, MAP under a Gaussian prior whose MEAN encodes clinical direction:
                 for a feature with monotone = +1 in features.yaml (a known risk factor), the prior nudges its
                 coefficient upward for patterns with more diseased vessels (and -1 the other way); 0 = no nudge.
- sel_joint  (e) in-fold L1 feature selection (multinomial L1-LR on the pattern), then the joint elastic-net LR.
- stack      (f+g) per-target binary models (CAD, LAD, LCX, RCA), each with its OWN in-fold L1 feature selection
                 (per-target feature sets), produce inner out-of-fold logits that are appended to the features of a
                 joint multinomial LR. Earlier-stage predictions are always OOF, never fitted on the rows they score.
- svm        (k) RBF SVM with Platt probabilities (best classifier in Alizadehsani et al., Expert Systems 2020).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, logit, softmax
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml import config
from ml.ingest import model_columns
from ml.joint import BITS, N_VESSELS
from ml.sprint.learners import Aligned, align

N_PAT = 8


def _monotone_by_column() -> np.ndarray:
    """monotone sign per model column (one-hot columns inherit their feature's sign)."""
    fcfg = config.features_cfg()
    mono = {f["id"]: int(f.get("monotone", 0)) for f in fcfg["features"]}
    out = []
    for c in model_columns(fcfg):
        fid = next((f for f in mono if c == f or c.startswith(f + "_")), None)
        out.append(mono.get(fid, 0) if fid else 0)
    return np.array(out, dtype=float)


class BayesLR:
    """MAP multinomial LR: loss = NLL + (1 / (2 sigma^2)) * ||W - M||^2, standardized inputs.
    Prior mean M[k, j] = tau * monotone_j * (n_vessels(k) - 1.5) / 1.5."""

    def __init__(self, seed=0, sigma=0.5, tau=0.15):
        self.sigma, self.tau, self.seed = sigma, tau, seed

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, int)
        self.sc_ = StandardScaler().fit(X)
        Z = self.sc_.transform(X)
        n, d = Z.shape
        mono = _monotone_by_column()
        if mono.size != d:
            mono = np.zeros(d)
        M = self.tau * np.outer((N_VESSELS - 1.5) / 1.5, mono)  # (8, d)
        Y = np.zeros((n, N_PAT))
        Y[np.arange(n), y] = 1
        lam = 1.0 / (2 * self.sigma**2)

        def f(theta):
            W, b = theta[: N_PAT * d].reshape(N_PAT, d), theta[N_PAT * d:]
            L = Z @ W.T + b
            lp = log_softmax(L, axis=1)
            nll = -(Y * lp).sum() / n
            G = (softmax(L, axis=1) - Y) / n
            gW = G.T @ Z + 2 * lam * (W - M) / n
            gb = G.sum(0)
            return nll + lam * np.sum((W - M) ** 2) / n, np.r_[gW.ravel(), gb]

        r = minimize(f, np.r_[M.ravel(), np.zeros(N_PAT)], jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        self.W_, self.b_ = r.x[: N_PAT * d].reshape(N_PAT, d), r.x[N_PAT * d:]
        return self

    def predict_proba(self, X):
        return softmax(self.sc_.transform(np.asarray(X, float)) @ self.W_.T + self.b_, axis=1)


def bayes_lr(seed, sigma=0.5, tau=0.15):
    return BayesLR(seed, sigma, tau)


def sel_joint(seed, C_sel=0.1, C=0.3, l1_ratio=0.5):
    sel = SelectFromModel(LogisticRegression(C=C_sel, l1_ratio=1.0, solver="saga", max_iter=5000, random_state=seed))
    lr = LogisticRegression(C=C, l1_ratio=l1_ratio, solver="saga", max_iter=5000, tol=1e-4, random_state=seed)
    return Aligned(Pipeline([("scale", StandardScaler()), ("sel", sel), ("clf", lr)]))


def _binary_targets(y):
    bits = BITS[np.asarray(y, int)]
    return {"CAD": (np.asarray(y) > 0).astype(int), "LAD": bits[:, 0], "LCX": bits[:, 1], "RCA": bits[:, 2]}


def _per_target_model(seed, C_sel):
    sel = SelectFromModel(LogisticRegression(C=C_sel, l1_ratio=1.0, solver="saga", max_iter=5000, random_state=seed))
    clf = LogisticRegression(C=0.3, max_iter=5000, random_state=seed)
    return Pipeline([("scale", StandardScaler()), ("sel", sel), ("clf", clf)])


class Stack:
    def __init__(self, seed=0, C_sel=0.2, C=0.3, inner=5):
        self.seed, self.C_sel, self.C, self.inner = seed, C_sel, C, inner

    def _logits(self, models, X):
        p = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
        return logit(np.clip(p, 1e-4, 1 - 1e-4))

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, int)
        bt = _binary_targets(y)
        oof = np.zeros((len(X), 4))
        skf = StratifiedKFold(self.inner, shuffle=True, random_state=self.seed)
        for a, b in skf.split(X, y):
            ms = [_per_target_model(self.seed, self.C_sel).fit(X[a], bt[t][a]) for t in bt]
            oof[b] = self._logits(ms, X[b])
        self.base_ = [_per_target_model(self.seed, self.C_sel).fit(X, bt[t]) for t in bt]
        self.final_ = Pipeline([("scale", StandardScaler()),
                                ("clf", LogisticRegression(C=self.C, max_iter=5000, random_state=self.seed))])
        self.final_.fit(np.hstack([X, oof]), y)
        self.selected_ = {t: int(m.named_steps["sel"].get_support().sum()) for t, m in zip(bt, self.base_, strict=True)}
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        Z = np.hstack([X, self._logits(self.base_, X)])
        return align(self.final_.predict_proba(Z), self.final_.classes_)


def stack(seed, C_sel=0.2, C=0.3):
    return Stack(seed, C_sel, C)


def svm(seed, C=1.0, gamma="scale"):
    return Aligned(Pipeline([("scale", StandardScaler()),
                             ("clf", SVC(C=C, gamma=gamma, probability=True, random_state=seed))]))


LEARNERS2 = {"bayes_lr": bayes_lr, "sel_joint": sel_joint, "stack": stack, "svm": svm}
