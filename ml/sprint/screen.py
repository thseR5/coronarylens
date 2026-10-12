"""Stage A screening: repeated CV of fixed procedures on the TRAINING split only (never a headline number).

For each outer fold, each learner's inner-OOF + refit predictions are computed once and cached; every procedure is
then a cheap function of that cache. Results -> reports/sprint/screen_<tag>.json and a table for EXPERIMENTS.md.

Usage: uv run python -m ml.sprint.screen --tag A1 --learners lr_en rf et catboost --procedures default
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import RepeatedStratifiedKFold

from ml import config
from ml.sprint.core import (
    Procedure,
    apply_post,
    count_model_outputs,
    fit_post,
    labels,
    learner_outputs,
    score,
    training_split,
    transform_features,
)

OUT = config.REPORTS / "sprint"
SEED = 1042  # screening folds differ from the nested-CV folds (seed 42 family)

# Learner specs available to screening; key = cache name.
SPECS: dict[str, dict] = {
    "lr_en": {"learner": "lr_en", "params": {"C": 0.3, "l1_ratio": 0.5}},
    "lr_l2": {"learner": "lr_l2", "params": {"C": 0.1}},
    "lr_l2_strong": {"learner": "lr_l2", "params": {"C": 0.02}},
    "rf": {"learner": "rf", "params": {"n_estimators": 300, "min_samples_leaf": 3}},
    "et": {"learner": "et", "params": {"n_estimators": 300, "min_samples_leaf": 3}},
    "lgbm": {"learner": "lgbm", "params": {}},
    "catboost": {"learner": "catboost", "params": {}},
    "catboost_d3": {"learner": "catboost", "params": {"depth": 3, "l2_leaf_reg": 10.0}},
    "ebm": {"learner": "ebm", "params": {}},
    "tabpfn": {"learner": "tabpfn", "params": {}},
    # batch 2 (ml/sprint/learners2.py)
    "bayes_lr": {"learner": "bayes_lr", "params": {"sigma": 0.5, "tau": 0.15}},
    "bayes_lr_flat": {"learner": "bayes_lr", "params": {"sigma": 0.5, "tau": 0.0}},  # same model, no clinical prior
    "sel_joint": {"learner": "sel_joint", "params": {}},
    "stack": {"learner": "stack", "params": {}},
    "svm": {"learner": "svm", "params": {}},
}


CACHE = Path(".scratch/sprint_cache")  # gitignored; learner outputs are deterministic given spec/fold/seed


def _cached_learner(name, i, Xa, ya, Xb, features, inner, seed):
    key = hashlib.sha256(json.dumps([SPECS[name], features, i, seed, inner, SEED, ya.tolist()],
                                    sort_keys=True).encode()).hexdigest()[:16]
    f = CACHE / f"{name}_{key}.joblib"
    if f.exists():
        return joblib.load(f)
    out = learner_outputs(SPECS[name], Xa, ya, Xb, seed + i, inner)
    CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, f)
    return out


def fold_work(i, a, b, X, Y, cols, names, features, inner, seed, procedures):
    """One outer fold: learner caches (from disk when available) + scores of every procedure."""
    y = Y["pattern"]
    Xa, Xb = transform_features(features, X[a], X[b], cols)
    cache = {n: _cached_learner(n, i, Xa, y[a], Xb, features, inner, seed) for n in names}
    cnt = count_model_outputs(Xa, y[a], Xb, seed + i, inner) if any(p.count_lambda for p in procedures) else None
    Yb = {k: v[b] for k, v in Y.items()}
    scores = {}
    for p in procedures:
        fit = fit_post(p, [cache[n].oof for n in p.learners], None if cnt is None else cnt.oof, y[a])
        P = apply_post(fit, [cache[n].test for n in p.learners], None if cnt is None else cnt.test)
        scores[p.name] = score(Yb, P)
    return i, scores


def run(tag: str, learners: list[str], procedures: list[Procedure], features: str = "base", folds: int = 5,
        repeats: int = 5, inner: int = 3, n_jobs: int = 3) -> dict:
    X, T, cols = training_split()
    Y = labels(T)
    y = Y["pattern"]
    splits = list(RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=SEED).split(X, y))
    t0 = time.time()
    # no nested parallelism: every learner/BLAS/OpenMP pool inside a worker is limited to one thread
    with parallel_config(backend="loky", inner_max_num_threads=1):
        res = Parallel(n_jobs=n_jobs)(
            delayed(fold_work)(i, a, b, X, Y, cols, learners, features, inner, SEED, procedures)
            for i, (a, b) in enumerate(splits))
    t_cache = time.time() - t0
    per_proc: dict[str, list[dict]] = {p.name: [] for p in procedures}
    for _, scores in sorted(res, key=lambda r: r[0]):
        for name, m in scores.items():
            per_proc[name].append(m)
    summary = {}
    for name, rows in per_proc.items():
        summary[name] = {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows])),
                             "se": float(np.std([r[k] for r in rows]) / np.sqrt(len(rows)))} for k in rows[0]}
        summary[name]["_folds"] = rows
    out = {"tag": tag, "stage": "screening (training split only, never headline)", "features": features,
           "cv": f"{repeats}x{folds} repeated stratified, inner {inner}-fold", "learners": {n: SPECS[n] for n in learners},
           "seconds_total": round(t_cache, 1), "procedures": summary}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"screen_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def table(out: dict) -> str:
    rows = ["| Procedure | CAD | LAD | LCX | RCA | Pattern log-loss | Mean Brier |", "|---|---|---|---|---|---|---|"]
    items = sorted(out["procedures"].items(), key=lambda kv: kv[1]["pattern_log_loss"]["mean"])
    for name, s in items:
        ms = {k: f"{s[k]['mean']:.3f}±{s[k]['sd']:.3f}" for k in s if not k.startswith("_")}
        rows.append(f"| {name} | {ms['auc_CAD']} | {ms['auc_LAD']} | {ms['auc_LCX']} | {ms['auc_RCA']} | "
                    f"{ms['pattern_log_loss']} | {ms['brier_mean']} |")
    return "\n".join(rows)


def default_procedures(learners: list[str]) -> list[Procedure]:
    """Each learner alone (vector calibration) + the current served recipe for reference."""
    procs = [Procedure((n,), "avg", "vector") for n in learners]
    if {"lr_en", "rf", "et"} <= set(learners):
        procs.append(Procedure(("lr_en", "rf", "et"), "avg", "vector"))
    return procs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--learners", nargs="+", required=True)
    ap.add_argument("--features", default="base")
    ap.add_argument("--combos", action="store_true", help="also all 2-3 learner averages/learned weights")
    ap.add_argument("--calibs", nargs="+", default=["vector"])
    ap.add_argument("--count", nargs="*", type=float, default=[])
    ap.add_argument("--combo-calibs", nargs="+", default=None, help="calibrations for combos (default: first of --calibs)")
    ap.add_argument("--n-jobs", type=int, default=3)
    a = ap.parse_args()
    procs = []
    for c in a.calibs:
        procs += [Procedure((n,), "avg", c) for n in a.learners]
    if a.combos:
        for r in (2, 3):
            for combo in itertools.combinations(a.learners, r):
                for comb in ("avg", "learned"):
                    for cc in (a.combo_calibs or a.calibs[:1]):
                        procs.append(Procedure(combo, comb, cc))
    base = tuple(n for n in ("lr_en", "rf", "et") if n in a.learners)
    for lam in a.count:
        if base:
            procs.append(Procedure(base, "avg", a.calibs[0], count_lambda=lam))
    procs = list(dict.fromkeys(procs))
    out = run(a.tag, a.learners, procs, a.features, n_jobs=a.n_jobs)
    print(f"{out['seconds_total']} s; {len(procs)} procedures")
    print(table(out))


if __name__ == "__main__":
    main()
