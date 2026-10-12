"""Low-overhead, deterministic prediction for fitted sklearn forests.

sklearn's predict_proba re-validates input per tree; for one row that overhead dominates (about 0.1 s for
500 trees). Here leaf class distributions are pre-normalized and aligned to a fixed class layout, then each
tree's Cython `apply` is called directly. Chunks of trees run in threads and are summed in a fixed order,
so results are deterministic. Matches sklearn predict_proba to ~1e-15 (tested).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

_POOL: ThreadPoolExecutor | None = None


def _pool(n: int) -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=n)
    return _POOL


class CompiledForest:
    def __init__(self, forest, n_classes: int, n_threads: int = 4):
        classes = np.asarray(forest.classes_, dtype=int)
        self.trees = [e.tree_ for e in forest.estimators_]
        self.values = []
        for t in self.trees:
            v = t.value[:, 0, :]
            v = v / np.maximum(v.sum(1, keepdims=True), 1e-300)
            full = np.zeros((v.shape[0], n_classes))
            full[:, classes] = v
            self.values.append(full)
        self.n_classes = n_classes
        self.n_threads = n_threads

    def _chunk(self, Xf: np.ndarray, ix: np.ndarray) -> np.ndarray:
        s = np.zeros((Xf.shape[0], self.n_classes))
        for i in ix:
            s += self.values[i][self.trees[i].apply(Xf)]
        return s

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xf = np.ascontiguousarray(X, dtype=np.float32)
        n = len(self.trees)
        if Xf.shape[0] < 200 or self.n_threads <= 1:
            s = self._chunk(Xf, np.arange(n))
        else:
            parts = np.array_split(np.arange(n), self.n_threads)
            s = np.zeros((Xf.shape[0], self.n_classes))
            for r in _pool(self.n_threads).map(lambda ix: self._chunk(Xf, ix), parts):
                s += r  # fixed order -> deterministic
        return s / n
