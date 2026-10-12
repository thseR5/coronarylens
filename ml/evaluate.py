"""Metrics, nested cross-validation harness, bootstrap CIs, DeLong test, calibration statistics."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold

from ml import models as M
from ml.calibrate import Calibrator
from ml.joint import N_PATTERNS, TARGETS, marginals
from ml.thresholds import fit_thresholds

EVAL_TARGETS = TARGETS + ["P_GE2", "P_3VD"]
EPS = 1e-6


# ----------------------------------------------------------------------------- labels
def labels(T) -> dict[str, np.ndarray]:
    nv = T[["LAD", "LCX", "RCA"]].sum(axis=1).to_numpy()
    return {"CAD": T["cad"].to_numpy(), "LAD": T["LAD"].to_numpy(), "LCX": T["LCX"].to_numpy(),
            "RCA": T["RCA"].to_numpy(), "P_GE2": (nv >= 2).astype(int), "P_3VD": (nv == 3).astype(int),
            "pattern": T["pattern_code"].to_numpy()}


# ----------------------------------------------------------------------------- fast metrics
def auc(y, p) -> float:
    y = np.asarray(y)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-p, kind="mergesort")
    y, p = y[o], p[o]
    distinct = np.r_[np.where(np.diff(p))[0], len(p) - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    prec = tp / (tp + fp)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def multiclass_log_loss(y, P) -> float:
    P = np.clip(np.asarray(P), EPS, 1)
    P = P / P.sum(1, keepdims=True)
    return float(-np.mean(np.log(P[np.arange(len(y)), np.asarray(y, dtype=int)])))


def threshold_metrics(y, p, t) -> dict[str, float]:
    y, pred = np.asarray(y), np.asarray(p) >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "specificity": tn / (tn + fp) if tn + fp else float("nan")}


def target_metrics(y, p, t: float | None) -> dict[str, float]:
    out = {"roc_auc": auc(y, p), "pr_auc": average_precision(y, p), "brier": brier(y, p), "log_loss": log_loss(y, p)}
    if t is not None:
        out.update(threshold_metrics(y, p, t))
    return out


def coherence_violation_rate(Pm: dict[str, np.ndarray], tol: float = 1e-9) -> float:
    mx = np.max(np.column_stack([Pm["LAD"], Pm["LCX"], Pm["RCA"]]), axis=1)
    return float(np.mean(Pm["CAD"] < mx - tol))


def all_metrics(Y, Pm, P, thr) -> dict:
    out = {t: target_metrics(Y[t], Pm[t], thr.get(t, {}).get("f1") if thr else None) for t in EVAL_TARGETS}
    out["pattern_log_loss"] = multiclass_log_loss(Y["pattern"], P)
    out["coherence_violation_rate"] = coherence_violation_rate(Pm)
    return out


# ----------------------------------------------------------------------------- nested CV
FAMILIES = {
    # family -> (registry name, kwargs, calibrations to evaluate)
    "joint": ("joint_ensemble", {"fe": False}, ["none", "temperature", "vector"]),
    "joint_fe": ("joint_ensemble", {"fe": True}, ["none", "temperature", "vector"]),
    "lightgbm": ("lightgbm", {}, ["none", "temperature"]),
    "independent": ("independent_ensemble", {}, ["none"]),
    "naive_lr": ("naive_logreg", {}, ["none"]),
    "regularized_lr": ("regularized_logreg", {}, ["none"]),
    "chains": ("classifier_chains", {}, ["none"]),
}


def _build(family, mcfg, n_estimators, columns):
    name, kw, _ = FAMILIES[family]
    if family == "chains":
        return M.build(name, mcfg)
    if family == "lightgbm":
        return M.build(name, mcfg)
    kw = dict(kw)
    if FAMILIES[family][0] == "joint_ensemble":
        kw["columns"] = columns
    return M.build(name, mcfg, n_estimators=n_estimators, **kw)


def predict_family(model, X):
    """-> (patterns (n,8), targets dict)."""
    if getattr(model, "kind", "") == "joint":
        P = model.predict_patterns(X)
        return P, marginals(P)
    Pm = model.predict_targets(X)
    return model.predict_patterns(X), Pm


def apply_calibration(family_kind, cal: Calibrator | None, P, Pm):
    if cal is None or family_kind != "joint":
        return P, Pm
    Pc = cal.transform(P)
    return Pc, marginals(Pc)


def fit_with_inner(family, mcfg, X, T, seed, inner_folds, n_estimators, columns):
    """Inner OOF on (X, T) -> calibrators + thresholds; then refit on all of (X, T)."""
    skf = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    n = len(X)
    oof_P = np.zeros((n, N_PATTERNS))
    oof_m = {t: np.zeros(n) for t in EVAL_TARGETS}
    for tr, va in skf.split(X, T["pattern_code"]):
        m = _build(family, mcfg, n_estimators, columns).fit_targets(X[tr], T.iloc[tr])
        P, Pm = predict_family(m, X[va])
        oof_P[va] = P
        for t in EVAL_TARGETS:
            oof_m[t][va] = Pm[t]
    final = _build(family, mcfg, n_estimators, columns).fit_targets(X, T)
    Y = labels(T)
    out = {}
    for cal_name in FAMILIES[family][2]:
        cal = None if cal_name == "none" else Calibrator(cal_name).fit(oof_P, Y["pattern"])
        if cal is not None:
            Pc, Pmc = apply_calibration("joint", cal, oof_P, None)
        else:
            Pc, Pmc = oof_P, oof_m
        thr = fit_thresholds(Y, Pmc, EVAL_TARGETS)
        out[cal_name] = {"cal": cal, "thr": thr, "oof_P": Pc, "oof_m": Pmc}
    return final, out


def run_outer_fold(fold_id, X, T, tr, te, mcfg, families, n_estimators, columns, seed):
    inner = int(mcfg["validation"]["inner_folds"])
    Yte = labels(T.iloc[te])
    res = {}
    for fam in families:
        final, cals = fit_with_inner(fam, mcfg, X[tr], T.iloc[tr], seed + fold_id, inner, n_estimators, columns)
        P, Pm = predict_family(final, X[te])
        kind = getattr(final, "kind", "")
        for cal_name, c in cals.items():
            Pc, Pmc = apply_calibration(kind, c["cal"], P, Pm)
            key = fam if cal_name == "none" and len(cals) == 1 else f"{fam}_{cal_name}"
            res[key] = {"metrics": all_metrics(Yte, Pmc, Pc, c["thr"]),
                        "pred": {t: Pmc[t] for t in EVAL_TARGETS}, "thr": c["thr"],
                        "cal": c["cal"].params() if c["cal"] else None}
    return fold_id, te, res


def summarize_folds(fold_metrics: list[dict]) -> dict:
    """List of all_metrics dicts -> mean/sd/se per leaf."""
    def leaves(d, prefix=()):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from leaves(v, prefix + (k,))
            else:
                yield prefix + (k,), v

    acc: dict = {}
    for fm in fold_metrics:
        for key, v in leaves(fm):
            acc.setdefault(key, []).append(v)
    out: dict = {}
    for key, vals in acc.items():
        a = np.array(vals, dtype=float)
        a = a[~np.isnan(a)]
        node = out
        for k in key[:-1]:
            node = node.setdefault(k, {})
        node[key[-1]] = {"mean": float(a.mean()) if a.size else None,
                         "sd": float(a.std(ddof=1)) if a.size > 1 else None,
                         "se": float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else None, "n": int(a.size)}
    return out


def headline(m: dict, targets=TARGETS) -> dict[str, float]:
    """Fold-level scalar summaries used by the selection rule (means over the 4 primary targets)."""
    return {"log_loss": float(np.mean([m[t]["log_loss"] for t in targets])),
            "brier": float(np.mean([m[t]["brier"] for t in targets])),
            "roc_auc": float(np.mean([m[t]["roc_auc"] for t in targets]))}


# ----------------------------------------------------------------------------- bootstrap / DeLong
def bootstrap_ci(Y, Pm, thr, n_boot: int, seed: int, targets=EVAL_TARGETS) -> dict:
    rng = np.random.default_rng(seed)
    n = len(Y["CAD"])
    idx = rng.integers(0, n, size=(n_boot, n))
    out = {}
    for t in targets:
        y, p = Y[t], Pm[t]
        th = thr[t]["f1"] if thr and t in thr else None
        point = target_metrics(y, p, th)
        samples: dict[str, list] = {k: [] for k in point}
        for b in range(n_boot):
            ii = idx[b]
            yb = y[ii]
            if yb.min() == yb.max():
                continue
            for k, v in target_metrics(yb, p[ii], th).items():
                samples[k].append(v)
        out[t] = {k: {"value": float(point[k]),
                      "ci95": [float(np.nanpercentile(s, 2.5)), float(np.nanpercentile(s, 97.5))]}
                  for k, s in samples.items()}
    return out


def _midrank_auc_components(y, p):
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    allp = np.r_[pos, neg]
    r = stats.rankdata(allp)
    rp, rn = stats.rankdata(pos), stats.rankdata(neg)
    a = (r[:m].sum() - m * (m + 1) / 2) / (m * n)
    v10 = (r[:m] - rp) / n          # structural components for positives
    v01 = 1 - (r[m:] - rn) / m      # for negatives
    return a, v10, v01


def delong_paired(y, p1, p2) -> dict:
    """DeLong et al. (1988) test for two correlated ROC AUCs (fast midrank form)."""
    y = np.asarray(y)
    a1, v10_1, v01_1 = _midrank_auc_components(y, np.asarray(p1))
    a2, v10_2, v01_2 = _midrank_auc_components(y, np.asarray(p2))
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    S = s10 / len(v10_1) + s01 / len(v01_1)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    diff = a1 - a2
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return {"auc_a": float(a1), "auc_b": float(a2), "diff": float(diff), "z": float(z),
            "p_value": float(2 * stats.norm.sf(abs(z)))}


def bootstrap_auc_diff(y, p1, p2, n_boot, seed) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        ii = rng.integers(0, n, n)
        if y[ii].min() == y[ii].max():
            continue
        d.append(auc(y[ii], p1[ii]) - auc(y[ii], p2[ii]))
    d = np.array(d)
    return {"diff": float(auc(y, p1) - auc(y, p2)), "ci95": [float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5))], "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


# ----------------------------------------------------------------------------- calibration stats
def calibration_stats(y, p, n_bins: int = 10) -> dict:
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    lp = logit(p)

    def nll(theta, offset_only=False):
        a, b = (theta[0], 1.0) if offset_only else (theta[0], theta[1])
        q = np.clip(expit(a + b * lp), EPS, 1 - EPS)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    slope_fit = minimize(nll, [0.0, 1.0], method="BFGS")
    int_fit = minimize(lambda th: nll(th, True), [0.0], method="BFGS")
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bins, ece = [], 0.0
    for i in range(n_bins):
        mask = b == i
        if mask.sum() == 0:
            continue
        mp, fy = float(p[mask].mean()), float(y[mask].mean())
        ece += mask.mean() * abs(mp - fy)
        bins.append({"bin": i, "n": int(mask.sum()), "mean_pred": mp, "frac_pos": fy})
    return {"slope": float(slope_fit.x[1]), "intercept_citl": float(int_fit.x[0]), "ece": float(ece),
            "reliability": bins}
