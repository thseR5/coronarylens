"""SHAP-family explanations of the final served function (impute -> encode -> joint ensemble -> calibration ->
marginalization), multi-output [CAD, LAD, LCX, RCA, P_GE2] on the probability scale.

Method "owen-partition-sampled": Owen values for the two-level clinical hierarchy of features.yaml
(group -> feature). Estimated by sampling hierarchy-respecting permutations (a random order of the groups,
then a random order of the features inside each group) with antithetic pairs; for each permutation the
marginal contributions telescope, so base + sum(phi) == f(x) exactly (additivity, tested). Value function is
interventional: v(S) = mean over background rows b of f(x_S, b_~S). Background = k-means centroids of the
training split. Group totals are the Owen group values (= Shapley values of the groups as players).

shap.PartitionExplainer was measured at 0.7-3.3 s per patient on this laptop (too slow for the 400 ms
budget); shap.PermutationExplainer (plain Shapley, no hierarchy) is used as a cross-check in tests.
"""

from __future__ import annotations

import numpy as np

from ml.ingest import column_to_feature
from ml.joint import EXPLAIN_OUTPUTS

METHOD = "owen-partition-sampled"


def column_groups(columns: list[str], fcfg: dict) -> list[list[int]]:
    """Model column indices per features.yaml group (engineered features live inside the pipeline)."""
    c2f = column_to_feature(fcfg)
    f2g = {f["id"]: f["group"] for f in fcfg["features"]}
    groups = [g["id"] for g in fcfg["groups"]]
    out = [[i for i, c in enumerate(columns) if f2g[c2f[c]] == g] for g in groups]
    return [g for g in out if g]


def _orders(groups: list[list[int]], n_perm: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(max(1, n_perm // 2)):
        g_order = rng.permutation(len(groups))
        order = np.concatenate([rng.permutation(groups[g]) for g in g_order])
        out += [order, order[::-1]]  # antithetic: the reverse is also hierarchy-respecting
    return out


def owen_values(f, x: np.ndarray, background: np.ndarray, groups: list[list[int]], n_perm: int = 8,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (phi (d, k), base (k,), fx (k,)). f maps (n, d) -> (n, k)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    bg = np.asarray(background, dtype=float)
    d, nb = x.size, bg.shape[0]
    orders = _orders(groups, n_perm, seed)
    P = len(orders)
    # coalition j of permutation p = first j features of the order come from x (j = 1..d-1; 0 and d shared)
    masks = np.zeros((P, d - 1, d), dtype=bool)
    for p, order in enumerate(orders):
        for j in range(1, d):
            masks[p, j - 1, order[:j]] = True
    masks = masks.reshape(-1, d)
    Z = np.where(masks[:, None, :], x[None, None, :], bg[None, :, :]).reshape(-1, d)
    Z = np.vstack([bg, x[None, :], Z])
    out = f(Z)
    k = out.shape[1]
    v0 = out[:nb].mean(0)
    fx = out[nb]
    vmid = out[nb + 1:].reshape(P, d - 1, nb, k).mean(2)
    phi = np.zeros((d, k))
    for p, order in enumerate(orders):
        v = np.vstack([v0, vmid[p], fx])
        phi[order] += np.diff(v, axis=0)
    phi /= P
    return phi, v0, fx


def aggregate_features(phi: np.ndarray, columns: list[str], fcfg: dict) -> dict[str, np.ndarray]:
    """Model-column SHAP -> feature-id SHAP (one-hot columns of a categorical are summed)."""
    c2f = column_to_feature(fcfg)
    out: dict[str, np.ndarray] = {}
    for i, c in enumerate(columns):
        out[c2f[c]] = out.get(c2f[c], 0) + phi[i]
    return out


# ----------------------------------------------------------------------------- narratives
TARGET_LABELS = {
    "CAD": ("coronary artery disease (any vessel)", "narrowing in at least one heart artery"),
    "LAD": ("stenosis of the left anterior descending artery", "narrowing in the artery at the front of the heart (LAD)"),
    "LCX": ("stenosis of the left circumflex artery", "narrowing in the artery at the side/back of the heart (LCX)"),
    "RCA": ("stenosis of the right coronary artery", "narrowing in the right heart artery (RCA)"),
    "P_GE2": ("multi-vessel disease (two or more vessels)", "narrowing in two or more heart arteries"),
}


def _fmt_value(fid: str, value, fmap: dict) -> str:
    f = fmap.get(fid, {})
    if f.get("type") == "binary":
        if fid == "sex_male":
            return "male" if float(value) >= 0.5 else "female"
        return "yes" if float(value) >= 0.5 else "no"
    if f.get("type") == "categorical":
        return str(value)
    unit = f.get("unit", "")
    try:
        v = float(value)
        s = f"{v:g}"
    except (TypeError, ValueError):
        s = str(value)
    return f"{s} {unit}".strip()


def narratives(target: str, p: float, base: float, contribs: list[dict], group_totals: dict[str, float],
               fmap: dict, group_labels: dict[str, str], top: int = 3) -> tuple[list[str], list[str]]:
    clin_label, plain_label = TARGET_LABELS[target]
    ups = [c for c in contribs if c["shap"] > 0.005][:top]
    downs = [c for c in contribs if c["shap"] < -0.005][:top]

    def item(c, plain=False):
        lab = fmap[c["feature"]]["label"]
        val = _fmt_value(c["feature"], c["value"], fmap)
        if plain:
            return f"{lab.lower()} ({val})"
        return f"{lab} = {val} ({c['shap'] * 100:+.1f} pp)"

    clin = [(f"Estimated probability of {clin_label}: {p * 100:.0f}% "
            f"(average over the reference population: {base * 100:.0f}%).")]
    if ups:
        clin.append("Factors raising the estimate: " + "; ".join(item(c) for c in ups) + ".")
    if downs:
        clin.append("Factors lowering the estimate: " + "; ".join(item(c) for c in downs) + ".")
    if group_totals:
        g, v = max(group_totals.items(), key=lambda kv: abs(kv[1]))
        clin.append(f"Largest contributing domain: {group_labels.get(g, g)} ({v * 100:+.1f} pp in total).")
    clin.append("Contributions describe this model's behaviour (SHAP/Owen values), not causal effects.")

    n = round(p * 100)
    pat = [(f"Based on the information entered, the model estimates about {n} in 100 people like you "
           f"would have {plain_label}.")]
    if ups:
        pat.append("Results that pushed this estimate up: " + ", ".join(item(c, True) for c in ups) + ".")
    if downs:
        pat.append("Results that pushed this estimate down: " + ", ".join(item(c, True) for c in downs) + ".")
    pat.append("This is an educational estimate, not a diagnosis. Only tests such as coronary angiography can "
               "show whether an artery is narrowed; please discuss results with your doctor.")
    return clin, pat


def build_target_explanations(phi_feat: dict[str, np.ndarray], base: np.ndarray, fx: np.ndarray,
                              raw_values: dict, fcfg: dict) -> dict[str, dict]:
    fmap = {f["id"]: f for f in fcfg["features"]}
    glabels = {g["id"]: g["label"] for g in fcfg["groups"]}
    out = {}
    for k, t in enumerate(EXPLAIN_OUTPUTS):
        contribs = [{"feature": fid, "value": raw_values.get(fid), "shap": float(v[k])} for fid, v in phi_feat.items()]
        contribs.sort(key=lambda c: (-abs(c["shap"]), c["feature"]))
        gt: dict[str, float] = {}
        for c in contribs:
            g = fmap[c["feature"]]["group"]
            gt[g] = gt.get(g, 0.0) + c["shap"]
        clin, pat = narratives(t, float(fx[k]), float(base[k]), contribs, gt, fmap, glabels)
        out[t] = {"base_value": float(base[k]), "output": float(base[k] + sum(c["shap"] for c in contribs)),
                  "contributions": contribs, "group_totals": gt, "narrative": clin, "narrative_patient": pat}
    return out


# ----------------------------------------------------------------------------- global
def feature_column_index(columns: list[str], fcfg: dict) -> dict[str, list[int]]:
    c2f = column_to_feature(fcfg)
    out: dict[str, list[int]] = {}
    for i, c in enumerate(columns):
        out.setdefault(c2f[c], []).append(i)
    return out


def local_explanations(f, Xs, background, groups, n_perm, seed) -> np.ndarray:
    """-> (N, d, k) Owen values for each row of Xs."""
    return np.stack([owen_values(f, x, background, groups, n_perm, seed)[0] for x in Xs])


def global_summary(phis: np.ndarray, columns: list[str], fcfg: dict) -> dict:
    fidx = feature_column_index(columns, fcfg)
    f2g = {f["id"]: f["group"] for f in fcfg["features"]}
    out = {}
    for k, t in enumerate(EXPLAIN_OUTPUTS):
        feat = {fid: phis[:, ix, k].sum(1) for fid, ix in fidx.items()}
        mean_abs = {fid: float(np.mean(np.abs(v))) for fid, v in feat.items()}
        mean_signed = {fid: float(np.mean(v)) for fid, v in feat.items()}
        grp: dict[str, np.ndarray] = {}
        for fid, v in feat.items():
            grp[f2g[fid]] = grp.get(f2g[fid], 0) + v
        out[t] = {"mean_abs_shap": dict(sorted(mean_abs.items(), key=lambda kv: -kv[1])),
                  "mean_shap": mean_signed,
                  "group_mean_abs": {g: float(np.mean(np.abs(v))) for g, v in grp.items()}}
    return out


def permutation_importance(f, X, Y: dict[str, np.ndarray], columns, fcfg, n_repeats=20, seed=0) -> dict:
    """Mean drop in ROC-AUC when a feature's column(s) are permuted jointly (rows shuffled)."""
    from ml.evaluate import auc

    rng = np.random.default_rng(seed)
    fidx = feature_column_index(columns, fcfg)
    base = f(X)
    base_auc = {t: auc(Y[t], base[:, k]) for k, t in enumerate(EXPLAIN_OUTPUTS)}
    out = {t: {} for t in EXPLAIN_OUTPUTS}
    for fid, ix in fidx.items():
        drops = {t: [] for t in EXPLAIN_OUTPUTS}
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, ix] = X[rng.permutation(len(X))][:, ix]
            pp = f(Xp)
            for k, t in enumerate(EXPLAIN_OUTPUTS):
                drops[t].append(base_auc[t] - auc(Y[t], pp[:, k]))
        for t in EXPLAIN_OUTPUTS:
            out[t][fid] = float(np.mean(drops[t]))
    return out


def deletion_curve(f, Xs, phis, background, columns, fcfg, k_max=15, target=0, seed=0) -> dict:
    """Remove features (interventional: replaced by background rows) in |SHAP| order vs random order;
    report mean |f(x) - f(x without top-k)| for the CAD output. Faithful attributions rise faster."""
    rng = np.random.default_rng(seed)
    fidx = feature_column_index(columns, fcfg)
    fids = list(fidx)
    bg = np.asarray(background)
    res = {"shap_order": np.zeros(k_max + 1), "random_order": np.zeros(k_max + 1)}
    for x, phi in zip(Xs, phis, strict=True):
        imp = np.array([abs(phi[fidx[fid], target].sum()) for fid in fids])
        orders = {"shap_order": np.argsort(-imp, kind="mergesort"), "random_order": rng.permutation(len(fids))}
        fx = f(x[None])[0, target]
        for name, order in orders.items():
            rows = []
            for k in range(k_max + 1):
                mask = np.zeros(len(x), dtype=bool)
                for j in order[:k]:
                    mask[fidx[fids[j]]] = True
                rows.append(np.where(mask[None, :], bg, x[None, :]))
            out = f(np.vstack(rows))[:, target].reshape(k_max + 1, len(bg)).mean(1)
            res[name] += np.abs(fx - out)
    return {k: (v / len(Xs)).tolist() for k, v in res.items()} | {"k": list(range(k_max + 1))}
