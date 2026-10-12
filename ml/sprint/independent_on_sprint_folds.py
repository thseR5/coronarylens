"""Score the independent binary ensemble (LR+RF+ET per target) on EXACTLY the sprint's 10x5 outer folds, so the served
procedure can be compared with it fold by fold (Nadeau-Bengio corrected paired test). Output: reports/sprint/indep_B1.json
"""

from __future__ import annotations

import json

import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import RepeatedStratifiedKFold

from ml import config
from ml import evaluate as E
from ml.ingest import load_dataset, model_columns
from ml.train import FULL


def main(n_jobs: int = 3) -> dict:
    fcfg, tcfg, mcfg = config.features_cfg(), config.targets_cfg(), config.models_cfg()
    seed = int(mcfg["seed"])
    ds = load_dataset(fcfg, tcfg)
    tr = np.asarray(json.loads((config.ARTIFACTS / "split.json").read_text("utf-8"))["train"])
    X, T = ds.X.to_numpy(dtype=float)[tr], ds.T.iloc[tr].reset_index(drop=True)
    # identical to ml/sprint/nested.py: RepeatedStratifiedKFold(5, 10, seed) on the pattern code
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed).split(X, T["pattern_code"]))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        res = Parallel(n_jobs=n_jobs)(delayed(E.run_outer_fold)(i, X, T, a, b, mcfg, ["independent"], FULL["cv_trees"],
                                                                  model_columns(fcfg), seed)
                                      for i, (a, b) in enumerate(splits))
    folds = []
    for i, te, r in sorted(res, key=lambda x: x[0]):
        mt = r["independent"]["metrics"]
        folds.append({"fold": i, "test_index": [int(j) for j in te],
                      **{f"auc_{t}": mt[t]["roc_auc"] for t in ["CAD", "LAD", "LCX", "RCA"]},
                      "brier_mean": float(np.mean([mt[t]["brier"] for t in ["CAD", "LAD", "LCX", "RCA"]])),
                      "pattern_log_loss": mt["pattern_log_loss"], "coherence": mt["coherence_violation_rate"]})
    out = {"model": "independent ensemble (elastic-net LR + RF + ET per target, 200 trees)",
           "folds_definition": "RepeatedStratifiedKFold(5, 10, seed) on pattern code, as ml/sprint/nested.py", "folds": folds}
    (config.REPORTS / "sprint" / "indep_B1.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = main()
    f = o["folds"]
    print({k: round(float(np.mean([x[k] for x in f])), 4) for k in f[0] if k not in ("fold", "test_index")})
