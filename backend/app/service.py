"""Model service: loads the served bundle once, caches responses by feature-vector hash."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import threading
from collections import OrderedDict
from concurrent.futures import CancelledError, Future, ProcessPoolExecutor
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


# ----------------------------------------------------------------------------- heavy-work process
# explain / counterfactual run in a separate process with its own copy of the model, so their Python-heavy loops
# can never hold the GIL while /analyze is answering (the fast path that drives probabilities and 3D colours).
_W: ServedModel | None = None


def _worker_init(art_dir: str) -> None:
    global _W
    _W = ServedModel(Path(art_dir))


def _w_ping():
    return _W is not None


def _w_explain(features):
    return _W.explain(features)


def _w_counterfactual(features, target, max_changes):
    return _W.counterfactual(features, target, max_changes)


class Superseded(Exception):
    """A newer request from the same client replaced this one before it started."""


class HeavyPool:
    def __init__(self, art_dir: Path, workers: int = 1):
        self.ex = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_worker_init,
                                      initargs=(str(art_dir),))
        self.pending: dict[tuple[str, str], list[Future]] = {}
        self.lock = threading.Lock()
        self.warm = self.ex.submit(_w_ping)  # load the model in the worker now, not on the first user request

    def run(self, kind: str, client: str | None, fn, *args):
        fut = self.ex.submit(fn, *args)
        if client:
            with self.lock:
                for old in self.pending.get((kind, client), []):
                    old.cancel()  # only succeeds for jobs that have not started yet
                self.pending[(kind, client)] = [fut]
        try:
            return fut.result()
        except CancelledError as e:
            raise Superseded(kind) from e

    def shutdown(self):
        self.ex.shutdown(wait=False, cancel_futures=True)


class Service:
    def __init__(self, art_dir: Path | None = None, heavy_pool: bool | None = None):
        art = art_dir or artifacts_dir()
        self.model = ServedModel(art)
        self.cache = LRU(512)
        self.inflight: dict[str, tuple[str, Future]] = {}  # explain jobs started speculatively by /analyze
        self.inflight_lock = threading.Lock()
        if heavy_pool is None:
            heavy_pool = os.environ.get("CORONARYLENS_HEAVY_POOL", "1") != "0"
        self.pool = HeavyPool(art) if heavy_pool else None

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

    def analyze(self, features, mode, client: str | None = None):
        out = self._cached("analyze", features, lambda: self.model.analyze(features, mode), mode)
        if self.pool is not None and client:
            self._prefetch_explain(features, client)
        return out

    def _prefetch_explain(self, features, client: str) -> None:
        """Start this patient's explanation in the worker while the analyze response travels and paints; the
        client's /explain then finds it cached or joins the running job instead of starting from scratch."""
        key = self._key("explain", features)
        if self.cache.get(key) is not None:
            return
        with self.inflight_lock:
            if key in self.inflight:
                return
            for k, (c, f) in list(self.inflight.items()):  # this tab moved on: drop ITS queued prefetches only
                if c == client and (f.cancel() or f.done()):
                    self.inflight.pop(k, None)
            fut = self.pool.ex.submit(_w_explain, features)
            self.inflight[key] = (client, fut)

        def done(f, key=key):
            with self.inflight_lock:
                self.inflight.pop(key, None)
            if not f.cancelled() and f.exception() is None:
                self.cache.put(key, f.result())

        fut.add_done_callback(done)

    def explain(self, features, client: str | None = None):
        if self.pool is None:
            return self._cached("explain", features, lambda: self.model.explain(features))
        key = self._key("explain", features)
        with self.inflight_lock:
            fut = self.inflight.get(key, (None, None))[1]
        if fut is not None and not fut.cancelled():
            try:
                return fut.result()  # join the speculative job started by /analyze
            except CancelledError:
                pass
        return self._cached("explain", features, lambda: self.pool.run("explain", client, _w_explain, features))

    def counterfactual(self, features, target, max_changes, client: str | None = None):
        if self.pool is None:
            fn = lambda: self.model.counterfactual(features, target, max_changes)
        else:
            fn = lambda: self.pool.run("cf", client, _w_counterfactual, features, target, max_changes)
        return self._cached("cf", features, fn, target, max_changes)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()

    def cohort_map(self, features):
        return self._cached("cohort", features, lambda: self.model.cohort_map(features))

    def similar(self, features, k):
        return self._cached("similar", features, lambda: self.model.similar(features, k), k)
