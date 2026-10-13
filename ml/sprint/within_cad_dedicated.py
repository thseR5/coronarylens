"""Exploratory: can dedicated models trained ONLY on CAD patients localise the artery better than the joint model?
Training split only, 5x5 repeated CV. Output: reports/sprint/within_cad_dedicated.json (exploratory, not adopted)."""

from __future__ import annotations

import json

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml import config
from ml.evaluate import auc
from ml.sprint.core import labels, training_split

MODELS = {
    "l2_logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)),
    "random_forest": lambda: RandomForestClassifier(300, min_samples_leaf=5, random_state=0, n_jobs=4),
    "catboost_d3": lambda: CatBoostClassifier(iterations=300, depth=3, l2_leaf_reg=10, verbose=False, random_seed=0,
                                              thread_count=4, allow_writing_files=False),
}


def main() -> dict:
    X, T, _ = training_split()
    Y = labels(T)
    cad = Y["CAD"] == 1
    out = {"note": "exploratory; best-of-3 is optimistic; not adopted (would need a new nested run + test look)",
           "n_cad_patients": int(cad.sum()), "results": {}}
    for v in ("LAD", "LCX", "RCA"):
        y = Y[v][cad]
        out["results"][v] = {}
        for name, mk in MODELS.items():
            a = [auc(y[te], mk().fit(X[cad][tr], y[tr]).predict_proba(X[cad][te])[:, 1])
                 for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1).split(X[cad], y)]
            out["results"][v][name] = {"auc_mean": float(np.mean(a)), "auc_sd": float(np.std(a))}
    (config.REPORTS / "sprint" / "within_cad_dedicated.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    r = main()
    print({v: {k: round(x["auc_mean"], 3) for k, x in d.items()} for v, d in r["results"].items()})
