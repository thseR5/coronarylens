"""Turn the sprint's nested-CV record (reports/sprint/nested_<tag>.json) into the metrics.json schema used by the
dashboard and report: the headline (outer-fold metrics of the whole selection procedure), the paired comparison with
the previous served recipe, and subgroup AUCs from the stored outer-test predictions."""

from __future__ import annotations

import json

import numpy as np

from ml import config
from ml import evaluate as E
from ml.joint import TARGETS, marginals
from ml.sprint.stats import corrected, paired

NESTED = config.REPORTS / "sprint"


def load(tag: str = "B1") -> dict:
    return json.loads((NESTED / f"nested_{tag}.json").read_text("utf-8"))


def fold_sizes(rec: dict, n_train_total: int = 242) -> tuple[float, float]:
    """Mean outer-train and outer-test sizes (for the Nadeau-Bengio correction)."""
    n_test = float(np.mean([len(f["selected_pred"]["test_index"]) for f in rec["folds"]]))
    return n_train_total - n_test, n_test


_SIZES = (193.6, 48.4)  # set per record in variant_summary / sprint_section


def _stat(vals) -> dict:
    a = np.asarray(vals, float)
    a = a[~np.isnan(a)]
    c = corrected(a, *_SIZES)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "se": c["se_corrected"], "ci95": c["ci95"],
            "n": int(a.size), "se_note": "Nadeau-Bengio corrected"}


def variant_summary(rec: dict, label: str = "selected") -> dict:
    """Per-target dict in the cv.variants schema (roc_auc, pr_auc, brier, accuracy, ... as mean/sd/se/n)."""
    global _SIZES
    _SIZES = fold_sizes(rec)
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


METRICS = ["pattern_log_loss", "brier_mean", "auc_CAD", "auc_LAD", "auc_LCX", "auc_RCA", "auc_P_GE2", "auc_P_3VD"]


def corrected_paired(rec: dict) -> dict:
    n_train, n_test = fold_sizes(rec)
    return {k: paired(rec["folds"], "selected", "baseline", k, k.startswith(("pattern", "brier")), n_train, n_test)
            for k in METRICS}


def vs_independent(rec: dict) -> dict | None:
    """Served procedure vs the independent binary ensemble on IDENTICAL outer folds (reports/sprint/indep_B1.json)."""
    f = NESTED / "indep_B1.json"
    if not f.exists():
        return None
    ind = json.loads(f.read_text("utf-8"))["folds"]
    if not all(b["selected_pred"]["test_index"] == i["test_index"] for b, i in zip(rec["folds"], ind, strict=True)):
        raise ValueError("independent-ensemble folds differ from the sprint folds")
    rows = [{"served": b["selected"], "independent": i} for b, i in zip(rec["folds"], ind, strict=True)]
    n_train, n_test = fold_sizes(rec)
    out = {k: paired(rows, "served", "independent", k, k.startswith(("pattern", "brier")), n_train, n_test)
           for k in ["pattern_log_loss", "brier_mean", "auc_CAD", "auc_LAD", "auc_LCX", "auc_RCA"]}
    out["coherence_independent"] = corrected([i["coherence"] for i in ind], n_train, n_test)
    return out


def served_coherence(rec: dict) -> dict:
    """Measured share of outer-test patients with P(CAD) < max vessel probability (0 by construction)."""
    vals = []
    for f in rec["folds"]:
        m = marginals(np.asarray(f["selected_pred"]["P"]))
        mx = np.max(np.column_stack([m["LAD"], m["LCX"], m["RCA"]]), axis=1)
        vals.append(float(np.mean(m["CAD"] < mx - 1e-9)))
    return {"mean": float(np.mean(vals)), "max": float(np.max(vals)), "n_folds": len(vals)}


def within_cad_localisation(rec: dict) -> dict:
    """How well vessel probabilities pick the diseased artery AMONG patients with CAD (outer-test predictions)."""
    from ml.sprint.core import labels, training_split

    _, T, _ = training_split()
    Y = labels(T)
    pat = T["pattern"].to_numpy()
    aucs: dict[str, list[float]] = {v: [] for v in ("LAD", "LCX", "RCA")}
    top = {"100": [0, 0], "010": [0, 0], "001": [0, 0]}
    for f in rec["folds"]:
        idx = np.asarray(f["selected_pred"]["test_index"])
        m = marginals(np.asarray(f["selected_pred"]["P"]))
        cad = Y["CAD"][idx] == 1
        for v, lst in aucs.items():
            y = Y[v][idx][cad]
            if 0 < y.sum() < y.size:
                lst.append(E.auc(y, m[v][cad]))
        V = np.column_stack([m["LAD"], m["LCX"], m["RCA"]])
        for k, col in (("100", 0), ("010", 1), ("001", 2)):
            s = pat[idx] == k
            top[k][0] += int((V[s].argmax(1) == col).sum())
            top[k][1] += int(s.sum())
    return {"within_cad_auc": {v: float(np.mean(a)) for v, a in aucs.items()},
            "single_vessel_true_vessel_ranked_highest": {k: {"hits": a, "n": n} for k, (a, n) in top.items()},
            "note": "outer-test predictions of the 10x5 nested CV; patients with CAD only"}


def sprint_section(rec: dict) -> dict:
    fin = json.loads((NESTED / "finalists.json").read_text("utf-8"))
    return {"n_screened": fin["n_screened"], "n_finalists": len(fin["finalists"]),
            "finalists": [f["name"] for f in fin["finalists"]], "outer": rec["outer"], "inner": rec["inner"],
            "chosen_counts": rec["chosen_counts"],
            "paired_diff_vs_previous_served": corrected_paired(rec),
            "stats": "paired differences over 50 outer folds; Nadeau-Bengio corrected SE, t (df 49) and two-sided p",
            "previous_served": variant_summary(rec, "baseline"),
            "vs_independent_same_folds": vs_independent(rec),
            "served_coherence_measured": served_coherence(rec),
            "within_cad_localisation": within_cad_localisation(rec),
            "coherence_posthoc": (json.loads((NESTED / "coherence_posthoc.json").read_text("utf-8"))
                                  if (NESTED / "coherence_posthoc.json").exists() else None),
            "rule": "finalists by the pre-registered rule; inner selection by cross-fitted inner-OOF pattern log-loss"}
