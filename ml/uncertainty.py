"""Bootstrap ensemble for 90% intervals: B refits of the served joint variant on bootstrap resamples of the
training split (smaller forests to keep artifacts and latency small; see docs/notes/ml.md).

At inference every bootstrap member predicts the 8-pattern distribution, the served calibrator is applied,
outputs are marginalized, and the 5th/95th percentiles across members form the interval. All trees of all
members are flattened into one array set so a single vectorized traversal predicts every tree at once.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from ml.joint import N_PATTERNS, align_proba, marginals
from ml.models import JointEnsemble


def fit_bootstrap(member_specs, X, patterns, B: int, n_estimators: int, seed: int, fe: bool,
                  columns) -> list[JointEnsemble]:
    rng = np.random.default_rng(seed)
    out = []
    n = len(X)
    for b in range(B):
        ii = rng.integers(0, n, n)
        m = JointEnsemble(member_specs, seed + 1000 + b, n_estimators, fe, columns).fit(X[ii], patterns[ii])
        out.append(m)
    return out


class FlatForests:
    """All trees of many forests in one flat node table for a vectorized traversal."""

    def __init__(self, forests: list):
        feats, thr, left, right, vals, roots, owner = [], [], [], [], [], [], []
        off = 0
        for fi, f in enumerate(forests):
            classes = np.asarray(f.classes_, dtype=int)
            for est in f.estimators_:
                t = est.tree_
                v = t.value[:, 0, :]
                v = v / np.maximum(v.sum(1, keepdims=True), 1e-300)
                full = np.zeros((t.node_count, N_PATTERNS))
                full[:, classes] = v
                leaf = t.children_left == -1
                feats.append(np.where(leaf, 0, t.feature).astype(np.int32))
                thr.append(t.threshold.astype(np.float64))
                left.append(np.where(leaf, np.arange(t.node_count), t.children_left) + off)
                right.append(np.where(leaf, np.arange(t.node_count), t.children_right) + off)
                vals.append(full)
                roots.append(off)
                owner.append(fi)
                off += t.node_count
        self.feature = np.concatenate(feats)
        self.threshold = np.concatenate(thr)
        self.left = np.concatenate(left).astype(np.int64)
        self.right = np.concatenate(right).astype(np.int64)
        self.value = np.concatenate(vals)
        self.is_leaf = self.left == np.arange(off)
        self.roots = np.array(roots, dtype=np.int64)
        self.owner = np.array(owner)
        self.n_forests = len(forests)
        self.counts = np.bincount(self.owner, minlength=self.n_forests)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X (n, d) -> (n, n_forests, 8) per-forest averaged class probabilities."""
        X = np.asarray(X, dtype=np.float32).astype(np.float64)
        n = X.shape[0]
        node = np.tile(self.roots, (n, 1))
        rows = np.arange(n)[:, None]
        for _ in range(64):
            active = ~self.is_leaf[node]
            if not active.any():
                break
            go_left = X[rows, self.feature[node]] <= self.threshold[node]
            node = np.where(active, np.where(go_left, self.left[node], self.right[node]), node)
        v = self.value[node]  # (n, n_trees, 8)
        out = np.zeros((n, self.n_forests, N_PATTERNS))
        for fi in range(self.n_forests):
            out[:, fi] = v[:, self.owner == fi].mean(1)
        return out


class BootstrapEnsemble:
    def __init__(self, models: list[JointEnsemble]):
        self.models = models
        self.weights = [np.array([w for _, _, w in m.members_]) for m in models]
        forests, self.slots = [], []
        for m in models:
            slots = []
            for _, pipe, _ in m.members_:
                clf = pipe.steps[-1][1]
                if isinstance(clf, (RandomForestClassifier, ExtraTreesClassifier)):
                    slots.append(len(forests))
                    forests.append(clf)
                else:
                    slots.append(None)
            self.slots.append(slots)
        self.flat = FlatForests(forests) if forests else None

    def predict_patterns(self, X: np.ndarray) -> np.ndarray:
        """-> (B, n, 8)."""
        X = np.asarray(X, dtype=float)
        m0 = self.models[0]
        Zfe = m0.members_[0][1].named_steps["fe"].transform(X) if m0.fe else X
        forest_out = self.flat.predict(Zfe) if self.flat is not None else None
        res = []
        for m, slots, w in zip(self.models, self.slots, self.weights, strict=True):
            parts = []
            for (_, pipe, _), s in zip(m.members_, slots, strict=True):
                if s is not None:
                    parts.append(forest_out[:, s])
                else:
                    parts.append(align_proba(pipe.predict_proba(X), pipe.classes_))
            P = sum(wi * p for wi, p in zip(w, parts, strict=True)) / w.sum()
            res.append(P / P.sum(1, keepdims=True))
        return np.stack(res)

    def intervals(self, X, calibrator, lo: float = 5.0, hi: float = 95.0) -> dict[str, np.ndarray]:
        Pb = self.predict_patterns(X)
        B, n, _ = Pb.shape
        Pc = calibrator.transform(Pb.reshape(B * n, -1)).reshape(B, n, -1)
        out = {}
        for k, v in marginals(Pc.reshape(B * n, -1)).items():
            v = v.reshape(B, n)
            out[k] = np.stack([np.percentile(v, lo, axis=0), np.percentile(v, hi, axis=0)], axis=1)
        return out
