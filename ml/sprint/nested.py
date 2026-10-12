"""Stage B: nested CV of the WHOLE selection procedure (the headline number of the sprint).

Outer: 10x5 repeated stratified CV on the training split (seed from models.yaml). In every outer fold:
  1. each candidate learner produces inner 5-fold OOF predictions on the outer-training part (+ a refit);
  2. every candidate procedure is scored on those inner OOF predictions with cross-fitted post-processing;
  3. the procedure with the lowest inner score is selected, its post-processing is fitted on the full inner OOF,
     and it predicts the outer-test part, which nothing above has seen;
  4. the current served recipe (lr_en+rf+et | avg | vector) is scored on the same outer fold, for a paired comparison.
Because selection happens inside every outer fold, the outer score accounts for the whole search.

Usage: uv run python -m ml.sprint.nested --tag B1 --candidates <file.json>
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import RepeatedStratifiedKFold

from ml import config
from ml.evaluate import threshold_metrics
from ml.joint import TARGETS, marginals
from ml.sprint.core import (
    Procedure,
    apply_post,
    count_model_outputs,
    crossfit_oof_score,
    fit_post,
    labels,
    learner_outputs,
    score,
    training_split,
    transform_features,
)
from ml.sprint.screen import SPECS
from ml.thresholds import max_f1_threshold

OUT = config.REPORTS / "sprint"
BASELINE = Procedure(("lr_en", "rf", "et"), "avg", "vector")


def _outer_fold(i, a, b, X, Y, cols, procs: list[Procedure], inner: int, seed: int):
    y = Y["pattern"]
    caches, counts = {}, {}
    for feat in sorted({p.features for p in procs}):
        Xa, Xb = transform_features(feat, X[a], X[b], cols)
        names = sorted({n for p in procs if p.features == feat for n in p.learners})
        caches[feat] = {n: learner_outputs(SPECS[n], Xa, y[a], Xb, seed + i, inner) for n in names}
        if any(p.count_lambda for p in procs if p.features == feat):
            counts[feat] = count_model_outputs(Xa, y[a], Xb, seed + i, inner)

    def parts(p):
        c, cnt = caches[p.features], counts.get(p.features)
        return ([c[n].oof for n in p.learners], [c[n].test for n in p.learners],
                None if cnt is None else cnt.oof, None if cnt is None else cnt.test)

    inner_scores = {}
    for p in procs:
        oofs, _, c_oof, _ = parts(p)
        inner_scores[p.name] = crossfit_oof_score(p, oofs, c_oof, y[a], seed + i)
    chosen = min(procs, key=lambda p: inner_scores[p.name])
    Yb = {k: v[b] for k, v in Y.items()}
    out = {"fold": i, "chosen": chosen.name, "inner_scores": inner_scores}
    for label, p in (("selected", chosen), ("baseline", BASELINE)):
        oofs, tests, c_oof, c_test = parts(p)
        fit = fit_post(p, oofs, c_oof, y[a])
        P = apply_post(fit, tests, c_test)
        m = score(Yb, P)
        # max-F1 thresholds from the post-processed inner OOF (training part only)
        P_oof = apply_post(fit, oofs, c_oof)
        mo, mt = marginals(P_oof), marginals(P)
        for t in TARGETS:
            thr = max_f1_threshold(Y[t][a], mo[t])
            for k, v in threshold_metrics(Yb[t], mt[t], thr).items():
                m[f"{k}_{t}"] = v
        out[label] = m
    return out


def run(tag: str, procs: list[Procedure], folds: int = 5, repeats: int = 10, inner: int = 5, n_jobs: int = 3) -> dict:
    if BASELINE not in procs:
        procs = [*procs, BASELINE]
    X, T, cols = training_split()
    Y = labels(T)
    seed = int(config.models_cfg()["seed"])
    splits = list(RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed).split(X, Y["pattern"]))
    t0 = time.time()
    # no nested parallelism: every learner/BLAS/OpenMP pool inside a worker is limited to one thread
    with parallel_config(backend="loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=n_jobs)(delayed(_outer_fold)(i, a, b, X, Y, cols, procs, inner, seed)
                                       for i, (a, b) in enumerate(splits))
    rows = sorted(rows, key=lambda r: r["fold"])
    keys = list(rows[0]["selected"])

    def agg(label):
        return {k: {"mean": float(np.nanmean([r[label][k] for r in rows])),
                    "sd": float(np.nanstd([r[label][k] for r in rows])),
                    "se": float(np.nanstd([r[label][k] for r in rows]) / np.sqrt(len(rows)))} for k in keys}

    diff = {}
    for k in keys:
        d = np.array([r["selected"][k] - r["baseline"][k] for r in rows], dtype=float)
        d = d[~np.isnan(d)]
        lower_is_better = k.startswith(("pattern_log_loss", "brier"))
        improved = int(np.sum(d < 0)) if lower_is_better else int(np.sum(d > 0))
        diff[k] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "se": float(d.std(ddof=1) / np.sqrt(d.size)),
                   "folds_improved": improved, "folds_equal": int(np.sum(d == 0)), "n_folds": int(d.size),
                   "better_is": "lower" if lower_is_better else "higher"}
    out = {"tag": tag, "stage": "nested CV of the selection procedure (headline)",
           "outer": f"{repeats}x{folds} repeated stratified (seed {seed})", "inner": f"{inner}-fold OOF + 5-fold cross-fitted post-processing",
           "n_candidates": len(procs), "candidates": [p.name for p in procs], "seconds": round(time.time() - t0, 1),
           "selected": agg("selected"), "baseline": agg("baseline"), "paired_diff_selected_minus_baseline": diff,
           "chosen_counts": dict(Counter(r["chosen"] for r in rows).most_common()), "folds": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"nested_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def parse_procs(spec: list[dict]) -> list[Procedure]:
    return [Procedure(tuple(d["learners"]), d.get("combine", "avg"), d.get("calib", "vector"),
                      float(d.get("count_lambda", 0.0)), d.get("features", "base")) for d in spec]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--candidates", required=True, help="JSON list of procedure dicts")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=3)
    a = ap.parse_args()
    with open(a.candidates, encoding="utf-8") as fh:
        procs = parse_procs(json.load(fh))
    out = run(a.tag, procs, repeats=a.repeats, n_jobs=a.n_jobs)
    s, bl, d = out["selected"], out["baseline"], out["paired_diff_selected_minus_baseline"]
    print(f"{out['n_candidates']} candidates, {out['seconds']} s; chosen: {out['chosen_counts']}")
    for k in ["auc_CAD", "auc_LAD", "auc_LCX", "auc_RCA", "pattern_log_loss", "brier_mean"]:
        print(f"{k:18s} selected {s[k]['mean']:.4f}±{s[k]['sd']:.3f}  baseline {bl[k]['mean']:.4f}±{bl[k]['sd']:.3f}  "
              f"paired diff {d[k]['mean']:+.4f} ± {d[k]['sd']:.4f} (SE {d[k]['se']:.4f}); "
              f"improved in {d[k]['folds_improved']}/{d[k]['n_folds']} folds ({d[k]['better_is']} is better)")


if __name__ == "__main__":
    main()
