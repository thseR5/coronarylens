"""Turn the sprint's nested-CV record (reports/sprint/nested_<tag>.json) into the metrics.json schema used by the
dashboard and report: the headline (outer-fold metrics of the whole selection procedure), the paired comparison with
the previous served recipe, and subgroup AUCs from the stored outer-test predictions."""

from __future__ import annotations

import json

import numpy as np

from ml import config
from ml import evaluate as E
from ml.joint import TARGETS, marginals

NESTED = config.REPORTS / "sprint"


def load(tag: str = "B1") -> dict:
    return json.loads((NESTED / f"nested_{tag}.json").read_text("utf-8"))


def _stat(vals) -> dict:
    a = np.asarray(vals, float)
    a = a[~np.isnan(a)]
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "se": float(a.std(ddof=1) / np.sqrt(a.size)),
            "n": int(a.size)}


def variant_summary(rec: dict, label: str = "selected") -> dict:
    """Per-target dict in the cv.variants schema (roc_auc, pr_auc, brier, accuracy, ... as mean/sd/se/n)."""
    rows = [f[label] for f in rec["folds"]]
    out: dict = {}
    for t in TARGETS:
        out[t] = {"roc_auc": _stat([r[f"auc_{t}"] for r in rows]),
                  "pr_auc": _stat([r[f"pr_auc_{t}"] for r in rows]),
                  "brier": _stat([r[f"brier_{t}"] for r in rows])}
        for k in ("accuracy", "precision", "recall", "f1", "specificity"):
            out[t][k] = _stat([r[f"{k}_{t}"] for r in rows])
    out["P_GE2"] = {"roc_auc": _stat([r["auc_P_GE2"] for r in rows])}
    out["P_3VD"] = {"roc_auc": _stat([r["auc_P_3VD"] for r in rows])}
    out["pattern_log_loss"] = _stat([r["pattern_log_loss"] for r in rows])
    out["brier_mean"] = _stat([r["brier_mean"] for r in rows])
    out["coherence_violation_rate"] = {"mean": 0.0, "sd": 0.0, "se": 0.0, "n": len(rows)}  # by construction
    return out


def oof_by_repeat(rec: dict, n_train: int, n_folds: int, label: str = "selected") -> list[dict[str, np.ndarray]]:
    """Outer-test predictions reassembled per repeat (each repeat covers every training patient once)."""
    reps = len(rec["folds"]) // n_folds
    out = []
    for r in range(reps):
        P = np.full((n_train, 8), np.nan)
        for f in rec["folds"][r * n_folds:(r + 1) * n_folds]:
            pred = f[f"{label}_pred"]
            P[np.asarray(pred["test_index"])] = np.asarray(pred["P"])
        out.append(marginals(P))
    return out


def subgroups(rec: dict, Xtr, Ytr, cols, n_folds: int) -> dict:
    oof = oof_by_repeat(rec, len(Xtr), n_folds)
    age = Xtr[:, cols.index("age")]
    male = Xtr[:, cols.index("sex_male")] == 1
    groups = {"sex=male": male, "sex=female": ~male, "age<50": age < 50, "age50-64": (age >= 50) & (age < 65),
              "age>=65": age >= 65}
    out = {}
    for g, mask in groups.items():
        out[g] = {"n": int(mask.sum())}
        for t in TARGETS:
            aucs = [E.auc(Ytr[t][mask], o[t][mask]) for o in oof]
            out[g][t] = {"prevalence": float(Ytr[t][mask].mean()), "roc_auc": float(np.nanmean(aucs)),
                         "roc_auc_sd_repeats": float(np.nanstd(aucs)),
                         "brier": float(np.mean([E.brier(Ytr[t][mask], o[t][mask]) for o in oof]))}
    return out


def sprint_section(rec: dict) -> dict:
    fin = json.loads((NESTED / "finalists.json").read_text("utf-8"))
    return {"n_screened": fin["n_screened"], "n_finalists": len(fin["finalists"]),
            "finalists": [f["name"] for f in fin["finalists"]], "outer": rec["outer"], "inner": rec["inner"],
            "chosen_counts": rec["chosen_counts"],
            "paired_diff_vs_previous_served": rec["paired_diff_selected_minus_baseline"],
            "previous_served": variant_summary(rec, "baseline"),
            "rule": "finalists by the pre-registered rule; inner selection by cross-fitted inner-OOF pattern log-loss"}
