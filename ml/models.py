"""Model registry driven by config/models.yaml.

Every model exposes:
- fit_targets(X, T)     T has columns cad, LAD, LCX, RCA, pattern_code
- predict_patterns(X)   (n, 8) joint pattern distribution (product distribution for non-joint models)
- predict_targets(X)    dict CAD/LAD/LCX/RCA/P_GE2/P_3VD -> (n,)
Joint models also implement the PatternModel protocol: fit(X, patterns).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.fastforest import CompiledForest
from ml.features import FeatureEngineer
from ml.joint import N_PATTERNS, align_proba, marginals, patterns_from_independent

TARGET_COLS = {"CAD": "cad", "LAD": "LAD", "LCX": "LCX", "RCA": "RCA"}


@runtime_checkable
class PatternModel(Protocol):
    def fit(self, X: np.ndarray, patterns: np.ndarray) -> PatternModel: ...

    def predict_patterns(self, X: np.ndarray) -> np.ndarray: ...


# ----------------------------------------------------------------------------- members
def make_member(cls: str, params: dict, seed: int, n_estimators: int | None = None,
                fe: bool = False, columns: list[str] | None = None) -> Pipeline:
    params = dict(params)
    steps: list = [("fe", FeatureEngineer(columns))] if fe else []
    if cls == "ElasticNetLogReg":
        lr = LogisticRegression(C=params.get("C", 1.0), l1_ratio=params.get("l1_ratio", 0.5), solver="saga",
                                max_iter=params.get("max_iter", 5000), tol=1e-4, random_state=seed)
        steps += [("scale", StandardScaler()), ("clf", lr)]
    elif cls == "LogReg":  # plain L2 logistic regression: the baseline most teams would build
        lr = LogisticRegression(C=params.get("C", 1.0), max_iter=params.get("max_iter", 5000), random_state=seed)
        steps += [("scale", StandardScaler()), ("clf", lr)]
    elif cls in ("RandomForest", "ExtraTrees"):
        if n_estimators is not None:
            params["n_estimators"] = n_estimators
        Est = RandomForestClassifier if cls == "RandomForest" else ExtraTreesClassifier
        steps += [("clf", Est(**params, random_state=seed, n_jobs=1))]
    elif cls == "LightGBM":
        from lightgbm import LGBMClassifier

        if n_estimators is not None:
            params["n_estimators"] = n_estimators
        steps += [("clf", LGBMClassifier(**params, subsample_freq=1, random_state=seed, n_jobs=1, verbose=-1,
                                         deterministic=True, force_row_wise=True))]
    else:
        raise KeyError(f"unknown member class {cls}")
    return Pipeline(steps)


def _proba(pipe: Pipeline, X: np.ndarray, n_classes: int) -> np.ndarray:
    return align_proba(pipe.predict_proba(X), pipe.classes_, n_classes)


def _targets_from_patterns(P: np.ndarray) -> dict[str, np.ndarray]:
    return marginals(P)


# ----------------------------------------------------------------------------- joint
class JointEnsemble:
    """Weighted average of 8-class members (multinomial elastic-net LR, RF, ExtraTrees, LightGBM...)."""

    kind = "joint"

    def __init__(self, members: list[dict], seed: int, n_estimators: int | None = None, fe: bool = False,
                 columns: list[str] | None = None):
        self.member_specs = members
        self.seed = seed
        self.n_estimators = n_estimators
        self.fe = fe
        self.columns = columns
        self._compiled: list | None = None

    def fit(self, X: np.ndarray, patterns: np.ndarray) -> JointEnsemble:
        X = np.asarray(X, dtype=float)
        self.members_ = []
        for m in self.member_specs:
            pipe = make_member(m["cls"], m.get("params", {}), self.seed, self.n_estimators, self.fe, self.columns)
            pipe.fit(X, np.asarray(patterns, dtype=int))
            self.members_.append((m["name"], pipe, float(m.get("weight", 1.0))))
        self._compiled = None
        return self

    def fit_targets(self, X, T):
        return self.fit(X, T["pattern_code"].to_numpy())

    def compile(self, n_threads: int = 4) -> JointEnsemble:
        """Swap forest prediction to the low-overhead CompiledForest (identical output)."""
        comp = []
        for _, pipe, _ in self.members_:
            clf = pipe.steps[-1][1]
            if isinstance(clf, (RandomForestClassifier, ExtraTreesClassifier)):
                comp.append(CompiledForest(clf, N_PATTERNS, n_threads))
            else:
                comp.append(None)
        self._compiled = comp
        return self

    def member_patterns(self, X: np.ndarray) -> list[np.ndarray]:
        X = np.asarray(X, dtype=float)
        out = []
        for i, (_, pipe, _) in enumerate(self.members_):
            cf = self._compiled[i] if self._compiled else None
            if cf is not None:
                Z = pipe[:-1].transform(X) if len(pipe.steps) > 1 else X
                out.append(cf.predict_proba(Z))
            else:
                out.append(_proba(pipe, X, N_PATTERNS))
        return out

    def predict_patterns(self, X: np.ndarray) -> np.ndarray:
        ws = np.array([w for _, _, w in self.members_])
        P = sum(w * p for w, p in zip(ws, self.member_patterns(X), strict=True)) / ws.sum()
        return P / P.sum(1, keepdims=True)

    def predict_targets(self, X):
        return _targets_from_patterns(self.predict_patterns(X))

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_compiled"] = None  # never pickle compiled views
        return d


# ----------------------------------------------------------------------------- binary adapter
class BinaryModel:
    """Adapter: an averaged binary ensemble (same member classes as the joint model) for one target."""

    def __init__(self, members: list[dict], seed: int, n_estimators: int | None = None):
        self.members = members
        self.seed = seed
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.pipes_ = [make_member(m["cls"], m.get("params", {}), self.seed, self.n_estimators).fit(X, y)
                       for m in self.members]
        return self

    def predict_proba1(self, X) -> np.ndarray:
        return np.mean([_proba(p, X, 2)[:, 1] for p in self.pipes_], axis=0)


class IndependentEnsemble:
    """Benchmark: one binary ensemble per target (CAD, LAD, LCX, RCA), no coherence guarantee."""

    kind = "independent"

    def __init__(self, members: list[dict], seed: int, n_estimators: int | None = None):
        self.members = members
        self.seed = seed
        self.n_estimators = n_estimators

    def fit_targets(self, X, T):
        self.models_ = {t: BinaryModel(self.members, self.seed, self.n_estimators).fit(X, T[c].to_numpy())
                        for t, c in TARGET_COLS.items()}
        return self

    def predict_targets(self, X):
        p = {t: m.predict_proba1(X) for t, m in self.models_.items()}
        P = patterns_from_independent(np.column_stack([p["LAD"], p["LCX"], p["RCA"]]))
        m = marginals(P)
        m.update(p)  # CAD and vessels from their own models
        return m

    def predict_patterns(self, X):
        p = self.predict_targets(X)
        return patterns_from_independent(np.column_stack([p["LAD"], p["LCX"], p["RCA"]]))


class ClassifierChains:
    """Benchmark: ensemble of classifier chains over (CAD, LAD, LCX, RCA) in random orders.

    Link j sees X plus the *out-of-fold predicted probabilities* of earlier links during training and the
    predicted probabilities at inference; true labels are never used as chain inputs."""

    kind = "chain"

    def __init__(self, base: dict, seed: int, n_chains: int = 6, inner_folds: int = 3):
        self.base = base
        self.seed = seed
        self.n_chains = n_chains
        self.inner_folds = inner_folds

    def fit_targets(self, X, T):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        names = list(TARGET_COLS)
        self.chains_ = []
        for c in range(self.n_chains):
            order = [names[i] for i in rng.permutation(len(names))]
            aug, links = X, []
            for t in order:
                y = T[TARGET_COLS[t]].to_numpy()
                proto = make_member(self.base["cls"], self.base.get("params", {}), self.seed)
                cv = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed + c)
                oof = cross_val_predict(clone(proto), aug, y, cv=cv, method="predict_proba")[:, 1]
                links.append((t, clone(proto).fit(aug, y)))
                aug = np.column_stack([aug, oof])
            self.chains_.append(links)
        return self

    def predict_targets(self, X):
        X = np.asarray(X, dtype=float)
        acc = {t: np.zeros(X.shape[0]) for t in TARGET_COLS}
        for links in self.chains_:
            aug = X
            for t, m in links:
                p = _proba(m, aug, 2)[:, 1]
                acc[t] += p
                aug = np.column_stack([aug, p])
        p = {t: v / len(self.chains_) for t, v in acc.items()}
        m = marginals(patterns_from_independent(np.column_stack([p["LAD"], p["LCX"], p["RCA"]])))
        m.update(p)
        return m

    def predict_patterns(self, X):
        p = self.predict_targets(X)
        return patterns_from_independent(np.column_stack([p["LAD"], p["LCX"], p["RCA"]]))


# ----------------------------------------------------------------------------- registry
def member_lookup(mcfg: dict) -> dict[str, dict]:
    return {m["name"]: m for m in mcfg["pattern_models"]["joint_ensemble"]["members"]}


def build(name: str, mcfg: dict, *, n_estimators: int | None = None, fe: bool = False,
          columns: list[str] | None = None):
    """Instantiate a registered model by name (pattern_models or benchmarks in models.yaml)."""
    seed = int(mcfg["seed"])
    spec = mcfg["pattern_models"].get(name) or mcfg["benchmarks"].get(name)
    if spec is None:
        raise KeyError(f"model {name!r} not registered in models.yaml")
    kind = spec["kind"]
    if kind == "joint":
        return JointEnsemble(spec["members"], seed, n_estimators, fe, columns)
    lookup = member_lookup(mcfg)
    if kind == "independent":
        # members are names of joint_ensemble members, or inline member specs
        members = [lookup[m] if isinstance(m, str) else m for m in spec["members"]]
        return IndependentEnsemble(members, seed, n_estimators)
    if kind == "chain":
        return ClassifierChains(lookup["logreg"], seed, int(spec.get("n_chains", 6)),
                                int(mcfg["validation"]["inner_folds"]))
    raise KeyError(f"unknown kind {kind}")
