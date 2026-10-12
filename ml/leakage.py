"""Leakage guard: forbidden label columns never reach a model; proxy scan for near-label features."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ml import config


class LeakageError(RuntimeError):
    pass


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def forbidden_names(tcfg: dict | None = None) -> set[str]:
    tcfg = tcfg or config.targets_cfg()
    names = set(tcfg["forbidden_columns"])
    names |= {tcfg["cad"]["column"], tcfg["cad"]["id"]}
    for v in tcfg["vessels"]:
        names |= {v["id"], v["column"]}
    return {_norm(n) for n in names}


def assert_clean(X: pd.DataFrame | list[str], tcfg: dict | None = None) -> None:
    """Raise LeakageError if any column (exact, id or case/punctuation variant) is a label column."""
    cols = list(X.columns) if isinstance(X, pd.DataFrame) else list(X)
    bad = forbidden_names(tcfg)
    hits = [c for c in cols if _norm(c) in bad]
    if hits:
        raise LeakageError(f"forbidden label columns present in model input: {hits}")


def proxy_scan(
    X: pd.DataFrame, y: np.ndarray, threshold: float = 0.95, n_splits: int = 5, seed: int = 42
) -> dict[str, float]:
    """In-fold univariate AUC (direction-free) per column on training folds; return columns whose
    max over folds exceeds `threshold` (possible label proxies). Empty dict = clean."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    worst: dict[str, float] = {}
    y = np.asarray(y)
    for tr, _ in skf.split(X, y):
        for c in X.columns:
            x = X[c].to_numpy()[tr]
            if np.unique(x).size < 2:
                continue
            a = roc_auc_score(y[tr], x)
            worst[c] = max(worst.get(c, 0.0), max(a, 1 - a))
    return {c: round(a, 4) for c, a in worst.items() if a > threshold}
