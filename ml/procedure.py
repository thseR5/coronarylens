"""Served model = the validated selection PROCEDURE from the Part 2 sprint (docs/EXPERIMENTS.md).

ProcedureModel.fit(X, patterns) does on the full training split exactly what the nested CV did inside each outer fold:
  1. every candidate learner produces inner K-fold out-of-fold (OOF) 8-pattern predictions and is refit on all rows;
  2. every candidate procedure is scored on those OOF predictions with cross-fitted post-processing;
  3. the procedure with the lowest score is chosen; its combination weights and calibrator are fitted on the full OOF.
predict_patterns returns the chosen learners' combined (uncalibrated) 8-pattern distribution; the chosen calibrator
is exposed as `calibrator_` and served like any other calibrator, so every displayed probability is still a marginal
of one calibrated joint distribution (coherent by construction).
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ml.calibrate import Calibrator
from ml.sprint.core import Procedure, crossfit_oof_score, learn_weights
from ml.sprint.learners import make
from ml.sprint.screen import SPECS

N_PAT = 8


def _norm(P):
    P = np.clip(P, 1e-6, None)
    return P / P.sum(1, keepdims=True)


class ProcedureModel:
    kind = "joint"

    def __init__(self, candidates: list[Procedure], seed: int, inner_folds: int = 5):
        self.candidates = candidates
        self.seed = seed
        self.inner_folds = inner_folds

    # ------------------------------------------------------------------ fit
    def fit(self, X, patterns) -> ProcedureModel:
        X, y = np.asarray(X, float), np.asarray(patterns, int)
        names = sorted({n for p in self.candidates for n in p.learners})
        skf = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed)
        folds = list(skf.split(X, y))
        oof = {n: np.zeros((len(X), N_PAT)) for n in names}
        for n in names:
            for a, b in folds:
                oof[n][b] = make(SPECS[n]["learner"], self.seed, **SPECS[n].get("params", {})).fit(X[a], y[a]) \
                    .predict_proba(X[b])
        self.inner_scores_ = {p.name: crossfit_oof_score(p, [oof[n] for n in p.learners], None, y, self.seed)
                              for p in self.candidates}
        self.chosen_ = min(self.candidates, key=lambda p: self.inner_scores_[p.name])
        p = self.chosen_
        oofs = [oof[n] for n in p.learners]
        self.weights_ = learn_weights(oofs, y) if p.combine == "learned" else np.full(len(oofs), 1 / len(oofs))
        P_oof = _norm(sum(w * o for w, o in zip(self.weights_, oofs, strict=True)))
        self.calibrator_ = Calibrator(p.calib).fit(P_oof, y)
        # cross-fitted calibrated OOF probabilities (calibrator never scores rows it was fitted on): used for
        # thresholds, conformal quantiles and decision curves
        self.oof_P_ = np.zeros_like(P_oof)
        cf = StratifiedKFold(5, shuffle=True, random_state=self.seed + 7)
        for a, b in cf.split(P_oof, y):
            self.oof_P_[b] = Calibrator(p.calib).fit(P_oof[a], y[a]).transform(P_oof[b])
        self.learners_ = [make(SPECS[n]["learner"], self.seed, **SPECS[n].get("params", {})).fit(X, y)
                          for n in p.learners]
        return self

    def fit_targets(self, X, T):
        return self.fit(X, T["pattern_code"].to_numpy())

    # ------------------------------------------------------------------ predict
    def predict_patterns(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        preds = getattr(self, "_fast", None) or [m.predict_proba for m in self.learners_]
        return _norm(sum(w * f(X) for w, f in zip(self.weights_, preds, strict=True)))

    def compile(self, n_threads: int = 4) -> ProcedureModel:
        """Serving speed-ups with identical output: vectorized libsvm probabilities for RBF SVMs (ml/fastsvm.py,
        tested to 1e-12) and multi-threaded CatBoost prediction (training used one thread per worker)."""
        from ml.fastsvm import FastSVC
        from ml.sprint.learners import align

        fast = []
        for m in self.learners_:
            est = getattr(m, "est", None)
            name = type(est).__name__
            if name == "Pipeline" and type(est.steps[-1][1]).__name__ == "SVC":
                scale, svc = est[:-1], est.steps[-1][1]
                fs = FastSVC(svc)
                fast.append(lambda X, scale=scale, fs=fs, cls=svc.classes_: align(fs.predict_proba(scale.transform(X)), cls))
            elif name == "CatBoostClassifier":
                fast.append(lambda X, est=est: align(est.predict_proba(X, thread_count=n_threads), est.classes_))
            else:
                fast.append(m.predict_proba)
        self._fast = fast
        return self

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("_fast", None)  # compiled views are rebuilt at load time
        return d

    def describe(self) -> dict:
        return {"chosen": self.chosen_.name, "weights": [float(w) for w in self.weights_],
                "inner_scores": self.inner_scores_, "n_candidates": len(self.candidates)}


class ProcedureBootstrap:
    """90% intervals: the chosen procedure's learners refit on B bootstrap resamples (combination weights and the
    served calibrator kept fixed), so intervals reflect model-fitting variability of the served procedure."""

    def __init__(self, model: ProcedureModel, X, patterns, B: int, seed: int, param_overrides: dict | None = None):
        X, y = np.asarray(X, float), np.asarray(patterns, int)
        rng = np.random.default_rng(seed)
        self.weights = model.weights_
        self.models = []
        over = param_overrides or {}
        for b in range(B):
            ii = rng.integers(0, len(X), len(X))
            members = []
            for n in model.chosen_.learners:
                params = {**SPECS[n].get("params", {}), **over.get(SPECS[n]["learner"], {})}
                members.append(make(SPECS[n]["learner"], seed + 1000 + b, **params).fit(X[ii], y[ii]))
            self.models.append(members)

    def predict_patterns(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        return np.stack([_norm(sum(w * m.predict_proba(X) for w, m in zip(self.weights, ms, strict=True)))
                         for ms in self.models])

    def intervals(self, X, calibrator, lo: float = 5.0, hi: float = 95.0) -> dict[str, np.ndarray]:
        from ml.joint import marginals

        Pb = self.predict_patterns(X)
        B, n, _ = Pb.shape
        Pc = calibrator.transform(Pb.reshape(B * n, -1)).reshape(B, n, -1)
        out = {}
        for k, v in marginals(Pc.reshape(B * n, -1)).items():
            v = v.reshape(B, n)
            out[k] = np.stack([np.percentile(v, lo, axis=0), np.percentile(v, hi, axis=0)], axis=1)
        return out


def candidates_from_config(mcfg: dict) -> list[Procedure]:
    return [Procedure(tuple(d["learners"]), d.get("combine", "avg"), d.get("calib", "vector"),
                      float(d.get("count_lambda", 0.0)), d.get("features", "base"))
            for d in mcfg["served_procedure"]["candidates"]]
