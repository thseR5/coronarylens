"""Score extra benchmark families on EXACTLY the outer folds of ml/train.py's nested CV, without retraining
everything or touching metrics.json. Output: reports/benchmarks/<family>.json.

Usage: uv run python -m ml.benchmark_extra regularized_lr
"""

from __future__ import annotations

import json
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold

from ml import config
from ml import evaluate as E
from ml.ingest import load_dataset, model_columns
from ml.train import FULL


def main(family: str, n_jobs: int = 4) -> dict:
    fcfg, tcfg, mcfg = config.features_cfg(), config.targets_cfg(), config.models_cfg()
    seed, val = int(mcfg["seed"]), mcfg["validation"]
    ds = load_dataset(fcfg, tcfg)
    split = json.loads((config.ARTIFACTS / "split.json").read_text("utf-8"))
    tr = np.asarray(split["train"])
    X, T = ds.X.to_numpy(dtype=float)[tr], ds.T.iloc[tr].reset_index(drop=True)
    rskf = RepeatedStratifiedKFold(n_splits=int(val["outer_folds"]), n_repeats=int(val["outer_repeats"]),
                                   random_state=seed)
    splits = list(rskf.split(X, T["pattern_code"]))  # identical to ml/train.py
    res = Parallel(n_jobs=n_jobs)(delayed(E.run_outer_fold)(i, X, T, a, b, mcfg, [family], FULL["cv_trees"],
                                                            model_columns(fcfg), seed)
                                  for i, (a, b) in enumerate(splits))
    folds = [r[2][family]["metrics"] for r in sorted(res, key=lambda r: r[0])]
    summary = E.summarize_folds(folds)
    out = {"family": family, "folds": "same 5x5 outer folds as metrics.json cv", "summary": summary}
    d = config.REPORTS / "benchmarks"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{family}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = main(sys.argv[1])
    s = o["summary"]
    print({t: round(s[t]["roc_auc"]["mean"], 3) for t in ["CAD", "LAD", "LCX", "RCA"]},
          "pattern log-loss", round(s["pattern_log_loss"]["mean"], 3),
          f"contradictions {100 * s['coherence_violation_rate']['mean']:.1f}% "
          f"± {100 * s['coherence_violation_rate']['sd']:.1f}")
