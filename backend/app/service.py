"""Model service: loads the served bundle once, caches responses by feature-vector hash."""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from ml import config
from ml.predictor import ServedModel


def artifacts_dir() -> Path:
    return Path(os.environ.get("CORONARYLENS_ARTIFACTS", config.ARTIFACTS))


class LRU:
    def __init__(self, maxsize: int = 512):
        self.maxsize = maxsize
        self.data: OrderedDict[str, Any] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
        return None

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            while len(self.data) > self.maxsize:
                self.data.popitem(last=False)


class Service:
    def __init__(self, art_dir: Path | None = None):
        self.model = ServedModel(art_dir or artifacts_dir())
        self.cache = LRU(512)

    def _key(self, kind: str, features: dict, *extra) -> str:
        """Hash of the encoded (imputed) feature vector + which ids were supplied (imputed list differs)."""
        x, imputed, _ = self.model.encode(features)
        h = hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes())
        h.update(repr((kind, sorted(imputed), extra)).encode())
        return h.hexdigest()

    def _cached(self, kind, features, fn, *extra):
        key = self._key(kind, features, *extra)  # also validates input (raises InputError)
        hit = self.cache.get(key)
        if hit is None:
            hit = fn()
            self.cache.put(key, hit)
        return hit

    def analyze(self, features, mode):
        return self._cached("analyze", features, lambda: self.model.analyze(features, mode), mode)

    def explain(self, features):
        return self._cached("explain", features, lambda: self.model.explain(features))

    def counterfactual(self, features, target, max_changes):
        return self._cached("cf", features, lambda: self.model.counterfactual(features, target, max_changes),
                            target, max_changes)

    def similar(self, features, k):
        return self._cached("similar", features, lambda: self.model.similar(features, k), k)
