"""The nearest-artery LV territories are checked against the AHA 17-segment standard (tools/territory_check.py)."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_territory_agreement_report_is_complete():
    res = json.loads((ROOT / "frontend/public/models/territory_agreement.json").read_text("utf-8"))
    anatomy = yaml.safe_load((ROOT / "config/anatomy.yaml").read_text("utf-8"))
    standard = {seg: v for v, segs in anatomy["bullseye"].items() for seg in segs}
    assert {int(k) for k in res["per_segment"]} == set(range(1, 18))
    for k, r in res["per_segment"].items():
        assert r["standard"] == standard[int(k)]  # the check compares against the config's standard assignment
        assert r["faces"] > 0
    assert 0.0 <= res["area_weighted_agreement"] <= 1.0
    assert res["segments_majority_agree"] == len(res["agreeing_segments"])
    assert "Voronoi" in res["method"]


def test_disagreeing_segments_are_named_and_explained():
    res = json.loads((ROOT / "frontend/public/models/territory_agreement.json").read_text("utf-8"))
    assert len(res["disagreeing_segments"]) == 17 - res["segments_majority_agree"]
    for d in res["disagreeing_segments"]:
        assert d["name"] and d["reason"] and d["standard"] != d["partition"]
    assert "not independent validation" in res["summary"]
