import time

import numpy as np
import pytest

from backend.app.schemas import (
    AnalyzeResponse,
    CounterfactualResponse,
    ExplainResponse,
    HealthResponse,
    Preset,
    SimilarResponse,
)
from ml import config

DISCLAIMER = ("Decision support and educational use only. Not a substitute for formal diagnostic imaging "
              "(e.g., coronary angiography) or clinical judgment.")


def test_health_config_metrics(client):
    HealthResponse(**client.get("/api/health").json())
    cfg = client.get("/api/config").json()
    assert set(cfg) == {"features", "targets", "anatomy", "models"}
    assert cfg["features"] == config.features_cfg()
    m = client.get("/api/metrics").json()
    assert "cv" in m and "test" in m
    g = client.get("/api/global-explanations").json()
    assert set(g["targets"]) == {"CAD", "LAD", "LCX", "RCA", "P_GE2"}


def test_presets(presets):
    assert 6 <= len(presets) <= 8
    ps = [Preset(**p) for p in presets]
    ids = {p.id for p in ps}
    assert {"low-risk", "lad-only", "rca-only", "lcx-only", "three-vessel", "uncertain", "synthetic-ood"} <= ids
    assert all(p.true_pattern is not None for p in ps if p.id != "synthetic-ood")
    assert "SYNTHETIC" in next(p for p in ps if p.id == "synthetic-ood").description


def test_analyze_contract_and_invariants(client, presets):
    for p in presets:
        for mode in ("f1", "screening"):
            r = client.post("/api/analyze", json={"features": p["features"], "mode": mode})
            assert r.status_code == 200, r.text
            a = AnalyzeResponse(**r.json())
            assert abs(sum(a.patterns.values()) - 1) < 1e-9
            assert a.cad.p >= max(v.p for v in a.vessels.values()) - 1e-12
            assert abs(a.cad.p - (1 - a.patterns["000"])) < 1e-12
            assert set(a.vessels) == {"LAD", "LCX", "RCA"}
            assert a.mode == mode
            assert a.imputed == []
    ood = next(p for p in presets if p["id"] == "synthetic-ood")
    r = AnalyzeResponse(**client.post("/api/analyze", json={"features": ood["features"]}).json())
    assert not r.applicability.in_domain
    assert len(r.out_of_range) > 0


def test_imputation(client, presets):
    f = dict(presets[0]["features"])
    f.pop("ldl")
    f["hdl"] = None
    r = client.post("/api/analyze", json={"features": f}).json()
    assert set(r["imputed"]) == {"ldl", "hdl"}
    r0 = client.post("/api/analyze", json={"features": {}}).json()
    assert len(r0["imputed"]) == len(config.features_cfg()["features"])


@pytest.mark.parametrize("features", [
    {"LAD": 1}, {"Cath": "CAD"}, {"rca": 0}, {"LCX": "Stenotic"},   # forbidden label columns
    {"unknown_feature": 3},                                           # unknown id
    {"age": 250}, {"ef_tte": -5}, {"bbb": "XBBB"}, {"sex_male": 2},  # outside features.yaml range / levels
])
def test_invalid_input_422(client, features):
    for path in ("/api/analyze", "/api/explain", "/api/similar", "/api/counterfactual", "/api/report"):
        r = client.post(path, json={"features": features})
        assert r.status_code == 422, (path, features, r.text)


def test_bad_request_shapes_422(client):
    assert client.post("/api/analyze", json={"features": {}, "mode": "nope"}).status_code == 422
    assert client.post("/api/analyze", json={"features": {}, "extra": 1}).status_code == 422
    assert client.post("/api/similar", json={"features": {}, "k": 50}).status_code == 422


def test_explain_contract(client, presets):
    r = client.post("/api/explain", json={"features": presets[1]["features"]})
    e = ExplainResponse(**r.json())
    assert set(e.targets) == {"CAD", "LAD", "LCX", "RCA", "P_GE2"}
    a = client.post("/api/analyze", json={"features": presets[1]["features"]}).json()
    for te in e.targets.values():
        assert abs(te.base_value + sum(c.shap for c in te.contributions) - te.output) < 1e-3
        assert len(te.contributions) == len(config.features_cfg()["features"])
        assert abs(sum(te.group_totals.values()) - (te.output - te.base_value)) < 1e-9
        assert te.narrative and te.narrative_patient
    assert abs(e.targets["CAD"].output - a["cad"]["p"]) < 1e-3
    assert abs(e.targets["LAD"].output - a["vessels"]["LAD"]["p"]) < 1e-3
    assert abs(e.targets["P_GE2"].output - a["multivessel"]["p_ge2"]) < 1e-3


def test_counterfactual_and_similar(client, presets):
    fmod = {f["id"] for f in config.features_cfg()["features"] if f["modifiable"]}
    for p in presets[:5]:
        r = CounterfactualResponse(**client.post("/api/counterfactual", json={"features": p["features"]}).json())
        assert r.note == "Model sensitivity, not treatment advice."
        for cf in r.counterfactuals:
            assert {c.feature for c in cf.changes} <= fmod
            assert cf.p_after < cf.p_before
            assert len({("bmi" if c.feature == "weight" else c.feature) for c in cf.changes}) <= 3
    s = SimilarResponse(**client.post("/api/similar", json={"features": presets[0]["features"], "k": 7}).json())
    assert len(s.neighbors) == 7
    assert all(n.pattern in {"000", "001", "010", "011", "100", "101", "110", "111"} for n in s.neighbors)
    d = [n.distance for n in s.neighbors]
    assert d == sorted(d)


def test_report_html(client, presets):
    r = client.post("/api/report", json={"features": presets[0]["features"]})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert DISCLAIMER in r.text


def test_cors(client):
    r = client.options("/api/analyze", headers={"Origin": "http://localhost:5173",
                                                 "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def _p95(fn, n=40):
    for _ in range(3):
        fn()  # warm-up
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.percentile(ts, 50))  # median: a regression guard robust to load spikes on shared machines/CI


def test_latency(client, presets):
    """Uncached latency: the service LRU cache is bypassed by varying age each call."""
    from backend.app.main import STATE

    svc = STATE["svc"]
    if svc.pool is not None:
        svc.pool.warm.result(timeout=120)  # do not time while the explain worker is still loading its model copy
    model = svc.model
    base = dict(presets[1]["features"])
    ages = iter(range(10_000))

    def vary():
        f = dict(base)
        f["age"] = 30 + (next(ages) % 60)
        f["bp"] = 100 + (next(ages) % 97)
        return f

    an = _p95(lambda: model.analyze(vary()))
    ex = _p95(lambda: model.explain(vary()), n=30)
    http = _p95(lambda: client.post("/api/analyze", json={"features": vary()}))
    print(f"\nlatency p50: analyze {an * 1000:.1f} ms, explain {ex * 1000:.1f} ms, http analyze {http * 1000:.1f} ms")
    assert an < 0.300  # generous so CI on shared runners is not flaky; UX targets are measured in e2e/latency.spec.ts
    assert ex < 0.800
