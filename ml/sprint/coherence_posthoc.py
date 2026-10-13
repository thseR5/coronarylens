"""How large are the independent ensemble's contradictions, and what does a post-hoc patch do?

Patch: P(CAD)' = max(P(CAD), max vessel probability). It removes contradictions in CAD vs vessels, but the outputs still
do not form one distribution (no P(>=2 vessels) / P(3VD) consistent with them). Compared on the sprint's 10x5 folds with
the served procedure. Output: reports/sprint/coherence_posthoc.json
"""

from __future__ import annotations

import json

import numpy as np

from ml import config
from ml.evaluate import auc, brier, log_loss
from ml.joint import marginals
from ml.sprint.core import labels, training_split
from ml.sprint.stats import corrected


def main() -> dict:
    _, T, _ = training_split()
    Y = labels(T)
    ind = json.loads((config.REPORTS / "sprint" / "indep_B1.json").read_text("utf-8"))["folds"]
    nb = json.loads((config.REPORTS / "sprint" / "nested_B1.json").read_text("utf-8"))["folds"]
    gaps, rows = [], []
    for fi, fs in zip(ind, nb, strict=True):
        idx = np.asarray(fi["test_index"])
        p = {t: np.asarray(fi["pred"][t]) for t in ["CAD", "LAD", "LCX", "RCA"]}
        mx = np.max(np.column_stack([p["LAD"], p["LCX"], p["RCA"]]), axis=1)
        gap = mx - p["CAD"]
        gaps += gap[gap > 1e-9].tolist()
        patched = np.maximum(p["CAD"], mx)
        served = marginals(np.asarray(fs["selected_pred"]["P"]))["CAD"]
        y = Y["CAD"][idx]
        rows.append({"orig": {"ll": log_loss(y, p["CAD"]), "brier": brier(y, p["CAD"]), "auc": auc(y, p["CAD"])},
                     "patched": {"ll": log_loss(y, patched), "brier": brier(y, patched), "auc": auc(y, patched)},
                     "served": {"ll": log_loss(y, served), "brier": brier(y, served), "auc": auc(y, served)}})
    n_test = float(np.mean([len(f["test_index"]) for f in ind]))
    n_train = 242 - n_test

    def cmp(a, b, k):
        return corrected([r[a][k] - r[b][k] for r in rows], n_train, n_test)

    g = np.asarray(gaps)
    out = {"violations": {"n_patient_folds_violating": int(g.size), "gap_mean": float(g.mean()),
                          "gap_median": float(np.median(g)), "gap_max": float(g.max())},
           "cad_metrics_mean": {lab: {k: float(np.mean([r[lab][k] for r in rows])) for k in ("ll", "brier", "auc")}
                                for lab in ("orig", "patched", "served")},
           "served_minus_patched": {k: cmp("served", "patched", k) for k in ("ll", "brier", "auc")},
           "note": "patch fixes CAD-vs-vessel contradictions only; P(>=2)/P(3VD) remain undefined for separate models"}
    (config.REPORTS / "sprint" / "coherence_posthoc.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = main()
    print(o["violations"])
    print({k: {m: round(v, 4) for m, v in d.items()} for k, d in o["cad_metrics_mean"].items()})
    print({k: (round(v["mean"], 4), round(v["p"], 4)) for k, v in o["served_minus_patched"].items()})
