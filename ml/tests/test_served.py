"""Invariants of the served function on the test split and on random inputs."""

import json

import numpy as np
import shap
from scipy.stats import spearmanr

from ml import config
from ml.calibrate import Calibrator
from ml.conformal import evaluate as conformal_eval
from ml.conformal import fit_quantiles
from ml.explain import owen_values
from ml.joint import marginals


def _random_inputs(n, seed=0):
    fcfg = config.features_cfg()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d = {}
        for f in fcfg["features"]:
            if f["type"] == "numeric":
                lo, hi = f["range"]
                d[f["id"]] = float(rng.uniform(lo, hi))
            elif f["type"] == "binary":
                d[f["id"]] = int(rng.integers(0, 2))
            elif f["type"] == "ordinal":
                d[f["id"]] = f["levels"][int(rng.integers(0, len(f["levels"])))]
            else:
                d[f["id"]] = f["levels"][int(rng.integers(0, len(f["levels"])))]
        out.append(d)
    return out


def _check_coherent(P):
    m = marginals(P)
    assert np.allclose(P.sum(1), 1, atol=1e-9)
    assert (P >= 0).all()
    assert np.allclose(m["CAD"], 1 - P[:, 0])
    mx = np.max(np.column_stack([m["LAD"], m["LCX"], m["RCA"]]), 1)
    assert (m["CAD"] >= mx - 1e-12).all()
    assert (m["P_GE2"] >= m["P_3VD"] - 1e-12).all()


def test_coherence_test_split(served, test_X):
    _check_coherent(served.patterns(test_X))


def test_coherence_random_inputs(served):
    X = np.vstack([served.encode(d)[0] for d in _random_inputs(200)])
    _check_coherent(served.patterns(X))
    for d in _random_inputs(5, seed=1):
        r = served.analyze(d)
        assert abs(sum(r["patterns"].values()) - 1) < 1e-9
        assert r["cad"]["p"] >= max(v["p"] for v in r["vessels"].values()) - 1e-12
        assert abs(r["cad"]["p"] - (1 - r["patterns"]["000"])) < 1e-12
        assert abs(sum(r["multivessel"]["n_vessels_dist"].values()) - 1) < 1e-9
        for t in [r["cad"], *r["vessels"].values()]:
            assert t["interval"][0] <= t["p"] <= t["interval"][1]


def test_calibrator_sanity():
    rng = np.random.default_rng(0)
    n, k = 4000, 8
    logits = rng.normal(size=(n, k)) * 1.5
    true_p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=p) for p in true_p])
    over = np.exp(logits * 2.5)
    over /= over.sum(1, keepdims=True)  # overconfident version
    cal = Calibrator("temperature").fit(over, y)
    assert 2.0 < cal.T_ < 3.0
    nll = lambda P: -np.mean(np.log(P[np.arange(n), y]))
    assert nll(cal.transform(over)) < nll(over)
    assert np.allclose(Calibrator("none").fit(over, y).transform(over), over)
    vec = Calibrator("vector").fit(over, y)
    assert nll(vec.transform(over)) < nll(over)


def test_served_calibration_reasonable():
    m = json.loads((config.ARTIFACTS / "metrics.json").read_text())
    for t in ["CAD", "LAD", "LCX", "RCA"]:
        c = m["calibration"]["train_oof"][t]
        assert 0.5 < c["slope"] < 2.0, (t, c)
        assert c["ece"] < 0.12, (t, c)


def test_conformal_coverage_synthetic():
    rng = np.random.default_rng(1)
    p = rng.random(4000)
    y = (rng.random(4000) < p).astype(int)
    q = fit_quantiles(y[:2000], p[:2000], 0.1)
    r = conformal_eval(y[2000:], p[2000:], q)
    assert abs(r["coverage_class1"] - 0.9) < 0.03 and abs(r["coverage_class0"] - 0.9) < 0.03


def test_conformal_coverage_test_split():
    m = json.loads((config.ARTIFACTS / "metrics.json").read_text())
    for t, r in m["conformal"]["test"].items():
        # n=61 test patients: binomial SD of coverage ~0.04; allow 3 SD below target
        assert r["coverage"] >= 0.78, (t, r)


def test_shap_additivity_and_determinism(served, test_X):
    for x in test_X[:3]:
        phi, base, _fx = owen_values(served.served, x, served.background, served.groups, 4, 42)
        assert np.abs(base + phi.sum(0) - served.served(x[None])[0]).max() < 1e-3
        phi2, _, _ = owen_values(served.served, x, served.background, served.groups, 4, 42)
        assert np.array_equal(phi, phi2)
    e = served.explain({})
    for te in e["targets"].values():
        assert abs(te["base_value"] + sum(c["shap"] for c in te["contributions"]) - te["output"]) < 1e-9
    assert abs(e["targets"]["CAD"]["output"] - served.analyze({})["cad"]["p"]) < 1e-3


def test_owen_agrees_with_shap_permutation(served, test_X):
    x = test_X[0]
    phi, _, _ = owen_values(served.served, x, served.background, served.groups, 40, 0)
    ex = shap.PermutationExplainer(served.served, shap.maskers.Independent(served.background,
                                                                           max_samples=len(served.background)),
                                   seed=0)
    sv = ex(x[None], max_evals=20 * (2 * len(x) + 1), silent=True).values[0]
    rho = spearmanr(np.abs(phi[:, 0]), np.abs(sv[:, 0])).statistic
    assert rho > 0.7
    assert np.argmax(np.abs(phi[:, 0])) in np.argsort(-np.abs(sv[:, 0]))[:3]
