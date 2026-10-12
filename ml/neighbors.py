"""k-NN similar patients over the standardized training split, returning their true angiography."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class NeighborIndex:
    def fit(self, X, row_index, patterns, cad) -> NeighborIndex:
        X = np.asarray(X, dtype=float)
        self.sc_ = StandardScaler().fit(X)
        self.nn_ = NearestNeighbors(n_neighbors=20).fit(self.sc_.transform(X))
        self.X_ = X
        self.row_index_ = np.asarray(row_index, dtype=int)
        self.patterns_ = np.asarray(patterns).astype(str)
        self.cad_ = np.asarray(cad, dtype=int)
        return self

    def query(self, x, k: int = 5) -> list[dict]:
        d, ix = self.nn_.kneighbors(self.sc_.transform(np.asarray(x, dtype=float).reshape(1, -1)),
                                    n_neighbors=min(k, len(self.X_)))
        return [{"pos": int(i), "index": int(self.row_index_[i]), "distance": float(di),
                 "pattern": str(self.patterns_[i]), "cad": bool(self.cad_[i])}
                for di, i in zip(d[0], ix[0], strict=True)]
