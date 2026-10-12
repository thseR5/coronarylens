import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from ml import config
from ml.evaluate import auc, average_precision, delong_paired
from ml.fastforest import CompiledForest
from ml.joint import BITS, marginals
from ml.models import PatternModel, build
from ml.thresholds import max_f1_threshold, min_sensitivity_threshold


def test_pattern_model_handles_missing_classes(dataset):
    X = dataset.X.to_numpy()
    y = dataset.T["pattern_code"].to_numpy()
    keep = y != 3  # class 011 absent from this "fold"
    m = build("joint_ensemble", config.models_cfg(), n_estimators=10)
    assert isinstance(m, PatternModel)
    m.fit(X[keep], y[keep])
    P = m.predict_patterns(X[:20])
    assert P.shape == (20, 8)
    assert np.allclose(P.sum(1), 1)
    assert np.allclose(P[:, 3], m.member_patterns(X[:20])[0][:, 3] / 3)  # forests give 0 to the unseen class; LR mass / 3


def test_marginals_bit_order():
    P = np.eye(8)
    m = marginals(P)
    assert list(m["LAD"]) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert list(m["RCA"]) == [0, 1, 0, 1, 0, 1, 0, 1]
    assert list(m["CAD"]) == [0, 1, 1, 1, 1, 1, 1, 1]
    assert list(m["P_GE2"]) == [0, 0, 0, 1, 0, 1, 1, 1]
    assert (BITS.sum(1) == [0, 1, 1, 2, 1, 2, 2, 3]).all()


def test_compiled_forest_matches_sklearn(dataset):
    X = dataset.X.to_numpy()
    y = dataset.T["pattern_code"].to_numpy()
    rf = RandomForestClassifier(50, min_samples_leaf=3, random_state=0).fit(X[y != 2], y[y != 2])
    cf = CompiledForest(rf, 8)
    ref = np.zeros((len(X), 8))
    ref[:, rf.classes_] = rf.predict_proba(X)
    assert np.abs(cf.predict_proba(X) - ref).max() < 1e-12


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.round(rng.random(200), 2) * 0.5 + y * 0.3
    assert auc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert average_precision(y, p) == pytest.approx(average_precision_score(y, p))
    r = delong_paired(y, p, p)
    assert r["diff"] == 0


def test_thresholds():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 1])
    p = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, .95])
    t = min_sensitivity_threshold(y, p, 0.9)
    assert np.mean(p[y == 1] >= t) >= 0.9
    assert 0 < max_f1_threshold(y, p) <= 1
