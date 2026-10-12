"""/explain runs in a separate process: it must never slow /analyze, and a newer request from the same browser tab
supersedes queued ones."""

import threading
import time

import numpy as np


def _vary(base, i):
    f = dict(base)
    f["bp"] = 90 + (i % 150)
    f["age"] = 30 + (i * 7 % 60)
    return f


def test_analyze_not_blocked_by_explain(client, presets):
    base = presets[0]["features"]
    stop = threading.Event()

    def hammer():
        i = 10_000
        while not stop.is_set():
            client.post("/api/explain", json={"features": _vary(base, i)}, headers={"X-Client-Id": "bg"})
            i += 1

    # analyze latency alone
    alone = []
    for i in range(15):
        t = time.perf_counter()
        assert client.post("/api/analyze", json={"features": _vary(base, i)}).status_code == 200
        alone.append(time.perf_counter() - t)
    th = threading.Thread(target=hammer, daemon=True)
    th.start()
    time.sleep(0.3)
    busy = []
    for i in range(15, 45):
        t = time.perf_counter()
        assert client.post("/api/analyze", json={"features": _vary(base, i)}).status_code == 200
        busy.append(time.perf_counter() - t)
    stop.set()
    th.join(timeout=10)
    p50_alone, p50_busy = float(np.median(alone)), float(np.median(busy))
    print(f"\nanalyze p50 alone {p50_alone * 1e3:.0f} ms, while explain runs {p50_busy * 1e3:.0f} ms")
    assert p50_busy < max(3 * p50_alone, 0.150)


def test_newer_request_from_same_client_supersedes_queued(client, presets):
    base = presets[1]["features"]
    results = {}

    def call(i):
        r = client.post("/api/explain", json={"features": _vary(base, 20_000 + i)}, headers={"X-Client-Id": "tab-1"})
        results[i] = r.status_code

    threads = [threading.Thread(target=call, args=(i,)) for i in range(5)]
    for i, th in enumerate(threads):
        th.start()
        time.sleep(0.02 if i else 0.05)
    for th in threads:
        th.join(timeout=30)
    codes = [results[i] for i in range(5)]
    assert codes[-1] == 200  # the newest request is always answered
    assert set(codes) <= {200, 409}


def slow_job(i):  # module-level so the spawned worker can import it
    time.sleep(0.4)
    return i


def test_heavy_pool_drops_stale_queued_jobs():
    from backend.app.service import HeavyPool, Superseded, artifacts_dir

    pool = HeavyPool(artifacts_dir())
    try:
        pool.run("t", "warm", slow_job, -1)  # start the worker (model load) before timing anything
        out = {}

        def call(i):
            try:
                out[i] = pool.run("t", "tab", slow_job, i)
            except Superseded:
                out[i] = "superseded"

        threads = [threading.Thread(target=call, args=(i,)) for i in range(6)]
        for th in threads:
            th.start()
            time.sleep(0.05)
        for th in threads:
            th.join(timeout=30)
        assert out[5] == 5  # newest answered
        assert "superseded" in out.values()  # queued stale jobs were cancelled, not computed
        # other clients are never cancelled by this tab
        assert pool.run("t", "other-tab", slow_job, 99) == 99
    finally:
        pool.shutdown()
