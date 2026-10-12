"""FastSVC reproduces sklearn's SVC.predict_proba (libsvm Platt + pairwise coupling) exactly."""

import numpy as np
from sklearn.svm import SVC

from ml.fastsvm import FastSVC


def test_fastsvc_matches_sklearn_multiclass():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 6))
    y = (X[:, 0] > 0).astype(int) * 2 + (X[:, 1] + 0.5 * rng.normal(size=160) > 0).astype(int)  # 4 classes
    y[:5] = 4  # a rare 5th class
    svc = SVC(probability=True, random_state=0).fit(X, y)
    Z = rng.normal(size=(300, 6))
    fs = FastSVC(svc)
    svc.decision_function_shape = "ovo"
    np.testing.assert_allclose(fs.decision_ovo(Z), svc.decision_function(Z), atol=1e-10)
    np.testing.assert_allclose(fs.predict_proba(Z), svc.predict_proba(Z), atol=1e-8)
