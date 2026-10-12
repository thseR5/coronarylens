"""Cohort map shows only the training split; the current patient is projected; test rows are never exposed."""

import json

from backend.app.service import artifacts_dir


def _split():
    return json.loads((artifacts_dir() / "split.json").read_text("utf-8"))


def test_cohort_map_is_training_split_only(client, presets):
    r = client.post("/api/cohort-map", json={"features": presets[0]["features"]})
    assert r.status_code == 200
    d = r.json()
    split = _split()
    idx = {p["index"] for p in d["points"]}
    assert idx == set(split["train"])
    assert not idx & set(split["test"])
    assert set(d["patient"]) == {"x", "y"}
    assert all(p["n_vessels"] == p["pattern"].count("1") for p in d["points"])
    assert 0 < sum(d["explained_variance"]) < 1


def test_projection_is_deterministic(client, presets):
    f = presets[2]["features"]
    a = client.post("/api/cohort-map", json={"features": f}).json()["patient"]
    b = client.post("/api/cohort-map", json={"features": dict(f, age=f["age"])}).json()["patient"]
    assert a == b


def test_training_patient_lookup(client):
    split = _split()
    ok = client.get(f"/api/training-patient/{split['train'][0]}")
    assert ok.status_code == 200 and ok.json()["index"] == split["train"][0]
    assert client.get(f"/api/training-patient/{split['test'][0]}").status_code == 404
