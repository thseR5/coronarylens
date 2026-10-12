"""End-to-end training: `uv run python -m ml.train [--quick]`.

1. locked stratified 80/20 split on vessel pattern (indices -> ml/artifacts/split.json)
2. repeated nested CV (5x5 outer, 3 inner) on the 80%: joint (+/- engineered features, 3 calibrations)
   vs LightGBM joint, independent ensembles, classifier chains
3. selection rule among the coherent joint variants
4. final refit on the 80% (5-fold inner OOF for calibrator / thresholds / conformal), test evaluated once
5. uncertainty (bootstrap ensemble, conformal, applicability domain), explanations, presets, figures
Outputs are deterministic: running twice gives an identical metrics.json (timings go to reports/train_log.txt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

from ml import config, conformal, dca, plots
from ml import evaluate as E
from ml.calibrate import Calibrator
from ml.domain import DomainModel
from ml.explain import (
    METHOD,
    column_groups,
    deletion_curve,
    global_summary,
    local_explanations,
    permutation_importance,
    representative_background,
)
from ml.ingest import data_sha256, decode_row, imputation_values, load_dataset, model_columns, training_ranges
from ml.joint import EXPLAIN_OUTPUTS, TARGETS, explain_matrix, marginals
from ml.leakage import assert_clean, proxy_scan
from ml.neighbors import NeighborIndex
from ml.procedure import ProcedureBootstrap, ProcedureModel, candidates_from_config
from ml.sprint import headline as SH
from ml.thresholds import fit_thresholds
from ml.uncertainty import BootstrapEnsemble, fit_bootstrap

JOINT_VARIANTS = ["joint_none", "joint_temperature", "joint_vector",
                  "joint_fe_none", "joint_fe_temperature", "joint_fe_vector"]  # simplest first

FULL = {"cv_trees": 200, "final_trees": None, "boot_B": None, "boot_trees": 50, "n_boot_ci": None,
        "bg": 50, "n_perm": 4, "global_n": None, "repeats": None, "folds": None, "lc": True}
QUICK = {"cv_trees": 25, "final_trees": 40, "boot_B": 5, "boot_trees": 10, "n_boot_ci": 100,
         "bg": 10, "n_perm": 2, "global_n": 15, "repeats": 1, "folds": 3, "lc": False}


def _r(x, nd=6):
    """Round floats recursively (stable, readable JSON)."""
    if isinstance(x, dict):
        return {str(k): _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if np.isnan(x) else round(x, nd)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return _r(x.tolist(), nd)
    return x


def _dump(obj, path: Path):
    path.write_text(json.dumps(_r(obj), indent=1, sort_keys=False, ensure_ascii=False) + "\n", encoding="utf-8")


def select_variant(headline_summary: dict[str, dict]) -> dict:
    """Among coherent joint variants: eligible = within 1 SE of the best mean log-loss AND within 1 SE of the
    best mean ROC-AUC; pick the simplest eligible (JOINT_VARIANTS order). Brier reported alongside."""
    hs = {v: headline_summary[v] for v in JOINT_VARIANTS if v in headline_summary}
    best_ll = min(hs, key=lambda v: hs[v]["log_loss"]["mean"])
    best_auc = max(hs, key=lambda v: hs[v]["roc_auc"]["mean"])
    ll_cut = hs[best_ll]["log_loss"]["mean"] + hs[best_ll]["log_loss"]["se"]
    auc_cut = hs[best_auc]["roc_auc"]["mean"] - hs[best_auc]["roc_auc"]["se"]
    eligible = [v for v in hs if hs[v]["log_loss"]["mean"] <= ll_cut and hs[v]["roc_auc"]["mean"] >= auc_cut]
    chosen = eligible[0] if eligible else best_ll
    return {"rule": "simplest joint variant within 1 SE of best mean log-loss and within 1 SE of best mean "
                    "ROC-AUC (means over CAD/LAD/LCX/RCA per outer fold)",
            "best_log_loss": best_ll, "best_roc_auc": best_auc, "log_loss_cutoff": ll_cut, "auc_cutoff": auc_cut,
            "eligible": eligible, "chosen": chosen,
            "feature_engineering": "_fe_" in chosen, "calibration": chosen.rsplit("_", 1)[1]}


def main(quick: bool = False, out_dir: Path = config.ARTIFACTS, reports: Path = config.REPORTS,
         n_jobs: int = 4) -> dict:
    t_start = time.time()
    S = QUICK if quick else FULL
    timings: dict[str, float] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fcfg, tcfg, mcfg = config.features_cfg(), config.targets_cfg(), config.models_cfg()
    seed = int(mcfg["seed"])
    val = mcfg["validation"]
    n_folds = S["folds"] or int(val["outer_folds"])
    n_rep = S["repeats"] or int(val["outer_repeats"])
    n_boot_ci = S["n_boot_ci"] or int(val["bootstrap_ci_resamples"])
    boot_B = S["boot_B"] or int(val["uncertainty_bootstrap_models"])

    ds = load_dataset(fcfg, tcfg)
    assert_clean(ds.X, tcfg)
    cols = model_columns(fcfg)
    X_all, T_all = ds.X.to_numpy(dtype=float), ds.T

    # ---------------------------------------------------------------- 1. locked split
    idx = np.arange(len(X_all))
    tr_idx, te_idx = train_test_split(idx, test_size=float(val["test_fraction"]),
                                      stratify=T_all["pattern_code"], random_state=seed)
    tr_idx, te_idx = np.sort(tr_idx), np.sort(te_idx)
    _dump({"seed": seed, "stratify": "pattern_code", "test_fraction": val["test_fraction"],
           "train": tr_idx.tolist(), "test": te_idx.tolist()}, out_dir / "split.json")
    Xtr, Ttr = X_all[tr_idx], T_all.iloc[tr_idx].reset_index(drop=True)
    Xte, Tte = X_all[te_idx], T_all.iloc[te_idx].reset_index(drop=True)
    Ytr, Yte = E.labels(Ttr), E.labels(Tte)
    metrics: dict = {"data": {"sha256": data_sha256(), "n": len(X_all), "n_train": len(tr_idx),
                              "n_test": len(te_idx), "n_model_columns": len(cols),
                              "test_pattern_counts": Tte["pattern"].value_counts().sort_index().to_dict()}}
    metrics["leakage"] = {"forbidden_columns_absent": True,
                          "proxy_scan_auc_gt_0.95": {t: proxy_scan(pd.DataFrame(Xtr, columns=cols), Ytr[t], seed=seed)
                                                     for t in TARGETS}}

    # ---------------------------------------------------------------- 2. nested CV
    t0 = time.time()
    families = list(E.FAMILIES) if not quick else ["joint", "joint_fe", "independent"]
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_rep, random_state=seed)
    splits = list(rskf.split(Xtr, Ttr["pattern_code"]))
    results = Parallel(n_jobs=n_jobs)(
        delayed(E.run_outer_fold)(i, Xtr, Ttr, a, b, mcfg, families, S["cv_trees"], cols, seed)
        for i, (a, b) in enumerate(splits))
    timings["nested_cv_s"] = time.time() - t0
    variants = list(results[0][2])
    fold_metrics = {v: [r[2][v]["metrics"] for r in results] for v in variants}
    oof = {v: [{t: np.zeros(len(Xtr)) for t in E.EVAL_TARGETS} for _ in range(n_rep)] for v in variants}
    for fid, te, res in results:
        rep = fid // n_folds
        for v in variants:
            for t in E.EVAL_TARGETS:
                oof[v][rep][t][te] = res[v]["pred"][t]
    summary = {v: E.summarize_folds(fold_metrics[v]) for v in variants}
    head = {v: E.summarize_folds([E.headline(m) for m in fold_metrics[v]]) for v in variants}
    sel = select_variant(head)
    chosen = sel["chosen"]
    fe = sel["feature_engineering"]
    cal_method = sel["calibration"]
    # engineered features: paired fold-level comparison (same outer folds)
    ll_diff = [E.headline(a)["log_loss"] - E.headline(b)["log_loss"]
               for a, b in zip(fold_metrics[f"joint_fe_{cal_method}"], fold_metrics[f"joint_{cal_method}"],
                               strict=True)]
    auc_diff = [E.headline(a)["roc_auc"] - E.headline(b)["roc_auc"]
                for a, b in zip(fold_metrics[f"joint_fe_{cal_method}"], fold_metrics[f"joint_{cal_method}"],
                                strict=True)]
    metrics["cv"] = {
        "settings": {"outer_folds": n_folds, "outer_repeats": n_rep, "inner_folds": int(val["inner_folds"]),
                     "trees_per_forest_in_cv": S["cv_trees"], "stratify": "pattern_code",
                     "thresholds": "max-F1 from inner OOF", "families": families},
        "variants": summary, "headline": head, "selection": sel,
        "feature_engineering_paired": {
            "calibration": cal_method,
            "delta_log_loss_fe_minus_base": {"mean": float(np.mean(ll_diff)), "sd": float(np.std(ll_diff, ddof=1))},
            "delta_roc_auc_fe_minus_base": {"mean": float(np.mean(auc_diff)), "sd": float(np.std(auc_diff, ddof=1))},
            "kept": fe},
    }

    # ---------------------------------------------------------------- 3/4. final refit + test
    t0 = time.time()
    fam = "joint_fe" if fe else "joint"
    final_inner = int(val["outer_folds"])
    procedure = mcfg["served"] == "procedure"
    if procedure:
        # Part 2 sprint: serve the validated selection procedure (same inner selection as the 10x5 nested CV)
        final = ProcedureModel(candidates_from_config(mcfg), seed,
                               int(mcfg["served_procedure"]["inner_folds"])).fit(Xtr, Ttr["pattern_code"].to_numpy())
        final.compile()
        calibrator = final.calibrator_
        oof_m = marginals(final.oof_P_)
        thr = fit_thresholds(Ytr, oof_m, E.EVAL_TARGETS)
        c = {"cal": calibrator, "thr": thr, "oof_P": final.oof_P_, "oof_m": oof_m}
        chosen, cal_method = "sprint_procedure", calibrator.method
        rec = SH.load("B1")
        metrics["cv"]["variants"]["sprint_procedure"] = SH.variant_summary(rec)
        metrics["sprint"] = SH.sprint_section(rec) | {"served_on_full_training_split": final.describe()}
    else:
        final, cals = E.fit_with_inner(fam, mcfg, Xtr, Ttr, seed, final_inner, S["final_trees"], cols)
        c = cals[cal_method]
        calibrator = c["cal"] or Calibrator("none").fit(c["oof_P"], Ytr["pattern"])
        thr = c["thr"]
        final.compile()
    Pte = calibrator.transform(final.predict_patterns(Xte))
    Mte = marginals(Pte)
    ind, ind_cals = E.fit_with_inner("independent", mcfg, Xtr, Ttr, seed, final_inner, S["final_trees"], cols)
    Mind = ind.predict_targets(Xte)
    Pind = ind.predict_patterns(Xte)
    timings["final_fit_s"] = time.time() - t0
    first_look = config.REPORTS / "sprint" / "test_first_look.json"
    if procedure and first_look.exists():
        metrics["test_first_look"] = json.loads(first_look.read_text("utf-8"))
    metrics["test"] = {
        "note": ("locked 20% split; SECOND look (after the Part 2 sprint; the first look, before the sprint, is in "
                 "test_first_look); 95% CIs from bootstrap resamples of test patients") if procedure else
                "locked 20% split, evaluated once; 95% CIs from bootstrap resamples of test patients",
        "n_bootstrap": n_boot_ci,
        "served": {"variant": chosen, "metrics": E.bootstrap_ci(Yte, Mte, thr, n_boot_ci, seed),
                   "pattern_log_loss": E.multiclass_log_loss(Yte["pattern"], Pte),
                   "coherence_violation_rate": E.coherence_violation_rate(Mte)},
        "independent": {"metrics": E.bootstrap_ci(Yte, Mind, ind_cals["none"]["thr"], n_boot_ci, seed),
                        "pattern_log_loss": E.multiclass_log_loss(Yte["pattern"], Pind),
                        "coherence_violation_rate": E.coherence_violation_rate(Mind)},
        "comparison_served_vs_independent": {
            t: {"delong": E.delong_paired(Yte[t], Mte[t], Mind[t]),
                "bootstrap": E.bootstrap_auc_diff(Yte[t], Mte[t], Mind[t], n_boot_ci, seed)}
            for t in TARGETS + ["P_GE2"]},
    }
    # coherence on CV OOF predictions, all variants
    metrics["coherence"] = {v: summary[v]["coherence_violation_rate"] for v in variants}
    metrics["thresholds"] = thr
    _dump({t: thr[t] for t in TARGETS + ["P_GE2", "P_3VD"]}, out_dir / "thresholds.json")

    # calibration statistics: test (served) and training OOF
    metrics["calibration"] = {
        "method": cal_method, "params": calibrator.params(),
        "test": {t: E.calibration_stats(Yte[t], Mte[t]) for t in TARGETS + ["P_GE2"]},
        "train_oof": {t: E.calibration_stats(Ytr[t], c["oof_m"][t]) for t in TARGETS + ["P_GE2"]},
        "cv_by_method": {m: {"log_loss": head[f"{fam}_{m}"]["log_loss"], "brier": head[f"{fam}_{m}"]["brier"]}
                         for m in ["none", "temperature", "vector"]},
    }
    plots.reliability({t: metrics["calibration"]["test"][t]["reliability"] for t in TARGETS},
                      figs / "reliability_test.png", f"Reliability, test split (n={len(te_idx)}), served model")
    plots.reliability({t: metrics["calibration"]["train_oof"][t]["reliability"] for t in TARGETS},
                      figs / "reliability_train_oof.png", "Reliability, training out-of-fold predictions")
    show = [v for v in [chosen, "lightgbm_temperature", "independent", "chains"] if v in summary]
    plots.cv_comparison(summary, show, TARGETS + ["P_GE2", "P_3VD"], figs / "cv_auc_comparison.png")

    # ---------------------------------------------------------------- conformal
    alpha = 1 - float(tcfg["conformal"]["target_coverage"])
    q = {t: conformal.fit_quantiles(Ytr[t], c["oof_m"][t], alpha) for t in TARGETS}
    _dump(q, out_dir / "conformal.json")
    metrics["conformal"] = {"target_coverage": 1 - alpha, "method": "Mondrian (class-conditional), K-fold OOF scores",
                            "quantiles": q, "test": {t: conformal.evaluate(Yte[t], Mte[t], q[t]) for t in TARGETS}}

    # ---------------------------------------------------------------- DCA
    metrics["dca"] = {"test": {t: dca.net_benefit(Yte[t], Mte[t]) for t in TARGETS},
                      "train_oof": {t: dca.net_benefit(Ytr[t], c["oof_m"][t]) for t in TARGETS}}
    dca.plot_dca(metrics["dca"]["train_oof"], figs / "dca_train_oof.png",
                 "Decision curves, training OOF predictions (served model)")
    dca.plot_dca(metrics["dca"]["test"], figs / "dca_test.png", "Decision curves, test split")

    # ---------------------------------------------------------------- subgroups (CV OOF of chosen variant)
    age = Xtr[:, cols.index("age")]
    groups_def = {"sex=male": Xtr[:, cols.index("sex_male")] == 1, "sex=female": Xtr[:, cols.index("sex_male")] == 0,
                  "age<50": age < 50, "age50-64": (age >= 50) & (age < 65), "age>=65": age >= 65}
    sub = SH.subgroups(rec, Xtr, Ytr, cols, n_folds) if procedure else {}
    for g, mask in ({} if procedure else groups_def).items():
        sub[g] = {"n": int(mask.sum())}
        for t in TARGETS:
            aucs = [E.auc(Ytr[t][mask], oof[chosen][r][t][mask]) for r in range(n_rep)]
            briers = [E.brier(Ytr[t][mask], oof[chosen][r][t][mask]) for r in range(n_rep)]
            sub[g][t] = {"prevalence": float(Ytr[t][mask].mean()), "roc_auc": float(np.nanmean(aucs)),
                         "roc_auc_sd_repeats": float(np.nanstd(aucs)), "brier": float(np.mean(briers))}
    metrics["subgroups"] = {"source": ("outer-test predictions of the sprint's 10x5 nested CV of the served procedure, "
                                       "averaged over repeats") if procedure else
                            f"nested-CV out-of-fold predictions ({chosen}), averaged over repeats",
                            "groups": sub}

    # ---------------------------------------------------------------- learning curve
    if S["lc"]:
        t0 = time.time()
        fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
        lc = {t: [] for t in TARGETS}
        lc_ll = []
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        from ml.models import build

        for fr in fracs:
            res = {t: [] for t in TARGETS}
            lls = []
            for k, (a, b) in enumerate(skf.split(Xtr, Ttr["pattern_code"])):
                if fr < 1:
                    a, _ = train_test_split(a, train_size=fr, stratify=Ttr["pattern_code"].iloc[a],
                                            random_state=seed + k)
                m = build("joint_ensemble", mcfg, n_estimators=S["cv_trees"], fe=fe, columns=cols)
                m.fit(Xtr[a], Ttr["pattern_code"].to_numpy()[a])
                Pm = marginals(m.predict_patterns(Xtr[b]))
                for t in TARGETS:
                    res[t].append(E.auc(Ytr[t][b], Pm[t]))
                lls.append(E.headline({t: E.target_metrics(Ytr[t][b], Pm[t], None) for t in TARGETS})["log_loss"])
            for t in TARGETS:
                lc[t].append(float(np.mean(res[t])))
            lc_ll.append(float(np.mean(lls)))
        n_train_fold = round(len(Xtr) * (n_folds - 1) / n_folds)
        metrics["learning_curve"] = {"train_fraction": fracs, "n_train": [round(f * n_train_fold) for f in fracs],
                                     "roc_auc": lc, "mean_log_loss": lc_ll,
                                     "note": "joint ensemble, uncalibrated, single 5-fold CV on the training split"}
        plots.lines({t: (metrics["learning_curve"]["n_train"], lc[t]) for t in TARGETS},
                    figs / "learning_curve.png", "training patients", "ROC-AUC (5-fold CV)", "Learning curve")
        timings["learning_curve_s"] = time.time() - t0

    # ---------------------------------------------------------------- uncertainty ensemble, domain, kNN
    t0 = time.time()
    if procedure:
        boot = ProcedureBootstrap(final, Xtr, Ttr["pattern_code"].to_numpy(), boot_B, seed,
                                  mcfg["served_procedure"].get("bootstrap_overrides"))
        be = boot
    else:
        specs = mcfg["pattern_models"]["joint_ensemble"]["members"]
        boot = fit_bootstrap(specs, Xtr, Ttr["pattern_code"].to_numpy(), boot_B, S["boot_trees"], seed, fe, cols)
        be = BootstrapEnsemble(boot)
    iv = be.intervals(Xte, calibrator)
    cover = {t: float(np.mean((Mte[t] >= iv[t][:, 0]) & (Mte[t] <= iv[t][:, 1]))) for t in TARGETS}
    width = {t: float(np.mean(iv[t][:, 1] - iv[t][:, 0])) for t in TARGETS}
    metrics["bootstrap_intervals"] = {"B": boot_B, "level": 0.90,
                                      "refits": (mcfg["served_procedure"].get("bootstrap_overrides") if procedure
                                                 else {"trees_per_forest": S["boot_trees"]}),
                                      "test_mean_width": width, "test_frac_point_inside_raw_interval": cover}
    timings["bootstrap_s"] = time.time() - t0
    domain = DomainModel(0.99, seed).fit(Xtr)
    dte = domain.score(Xte)
    metrics["applicability_domain"] = {"method": "Mahalanobis (Ledoit-Wolf), 99th pct of OOF training scores",
                                       "threshold": domain.threshold_,
                                       "test_frac_in_domain": float(np.mean(dte <= domain.threshold_))}
    knn = NeighborIndex().fit(Xtr, tr_idx, Ttr["pattern"].to_numpy(), Ttr["cad"].to_numpy())
    bg = representative_background(Xtr, S["bg"], seed)

    # ---------------------------------------------------------------- explanations
    t0 = time.time()

    def served(Z):
        return explain_matrix(calibrator.transform(final.predict_patterns(Z)))

    groups = column_groups(cols, fcfg)
    Xg = Xtr if S["global_n"] is None else Xtr[: S["global_n"]]
    phis = local_explanations(served, Xg, bg, groups, S["n_perm"], seed)
    add_err = float(np.max(np.abs(phis.sum(1) + served(bg).mean(0) - served(Xg))))
    gsum = global_summary(phis, cols, fcfg)
    Yte5 = {t: Yte[t] for t in EXPLAIN_OUTPUTS}
    pimp = permutation_importance(served, Xte, Yte5, cols, fcfg, n_repeats=10 if not quick else 2, seed=seed)
    from scipy.stats import spearmanr

    agree = {}
    for t in EXPLAIN_OUTPUTS:
        fids = list(gsum[t]["mean_abs_shap"])
        rho = spearmanr([gsum[t]["mean_abs_shap"][f] for f in fids], [pimp[t][f] for f in fids]).statistic
        top10_shap = fids[:10]
        top10_perm = sorted(pimp[t], key=lambda f: -pimp[t][f])[:10]
        agree[t] = {"spearman_rho": float(rho), "top10_overlap": len(set(top10_shap) & set(top10_perm))}
    Xd = Xte if S["global_n"] is None else Xte[:5]
    phis_te = local_explanations(served, Xd, bg, groups, S["n_perm"], seed)
    deletion = deletion_curve(served, Xd, phis_te, bg, cols, fcfg, k_max=15, seed=seed)
    plots.lines({"remove by |SHAP| (CAD)": (deletion["k"], deletion["shap_order"]),
                 "remove in random order": (deletion["k"], deletion["random_order"])},
                figs / "deletion_curve.png", "features removed", "mean |Δ P(CAD)|",
                "Fidelity: deletion curve (test split)")
    plots.hbar(gsum["CAD"]["mean_abs_shap"], figs / "global_shap_cad.png", "mean |Owen value| on P(CAD)",
               "Global importance, CAD (training split)")
    glob = {"method": METHOD, "scale": "probability", "n_patients": len(Xg), "background": S["bg"],
            "n_permutations": S["n_perm"], "targets": gsum, "permutation_importance_test_auc_drop": pimp,
            "agreement_shap_vs_permutation": agree, "max_additivity_error": add_err, "deletion_curve": deletion,
            "group_labels": {g["id"]: g["label"] for g in fcfg["groups"]}}
    _dump(glob, out_dir / "global_explanations.json")
    metrics["explanations"] = {"max_additivity_error": add_err, "agreement": agree,
                               "deletion_auc_shap": float(np.mean(deletion["shap_order"])),
                               "deletion_auc_random": float(np.mean(deletion["random_order"]))}
    timings["explain_s"] = time.time() - t0

    # ---------------------------------------------------------------- presets (test split)
    presets = build_presets(Xte, Tte, te_idx, Mte, q, domain, cols, fcfg, served)
    _dump(presets, out_dir / "presets.json")

    # ---------------------------------------------------------------- artifacts
    joblib.dump(final, out_dir / "joint_model.joblib", compress=3)
    joblib.dump(boot, out_dir / "bootstrap_ensemble.joblib", compress=3)
    joblib.dump(domain, out_dir / "domain.joblib", compress=3)
    joblib.dump(knn, out_dir / "knn.joblib", compress=3)
    np.save(out_dir / "shap_background.npy", bg)
    _dump(calibrator.params(), out_dir / "calibrator.json")
    _dump({"columns": cols, "imputation": imputation_values(pd.DataFrame(Xtr, columns=cols), fcfg),
           "train_ranges": training_ranges(pd.DataFrame(Xtr, columns=cols), fcfg)}, out_dir / "preprocess.json")
    _dump(metrics, out_dir / "metrics.json")
    mhash = hashlib.sha256((out_dir / "metrics.json").read_bytes()).hexdigest()[:8]
    import lightgbm
    import scipy
    import shap
    import sklearn

    meta = {"served": mcfg["served"], "variant": chosen, "feature_engineering": fe, "calibration": cal_method,
            "model_version": f"{mcfg['served']}-{chosen}-{data_sha256()[:8]}-{mhash}{'-quick' if quick else ''}",
            "seed": seed, "data_sha256": data_sha256(), "quick": quick,
            "explain": {"method": METHOD, "n_perm": S["n_perm"], "seed": seed, "background": S["bg"]},
            "trees": {"cv": S["cv_trees"], "final": S["final_trees"] or "models.yaml (500)",
                      "bootstrap": S["boot_trees"]},
            "procedure": final.describe() if procedure else None,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "shap": shap.__version__,
                         "lightgbm": lightgbm.__version__, "joblib": joblib.__version__}}
    _dump(meta, out_dir / "model_meta.json")
    timings["total_s"] = time.time() - t_start
    log = reports / ("train_log_quick.txt" if quick else "train_log.txt")
    log.write_text("\n".join(f"{k}: {v:.1f}" for k, v in timings.items()) + "\n")
    write_report(metrics, glob, reports / ("model_report_quick.md" if quick else "model_report.md"), variants)
    return metrics


def build_presets(Xte, Tte, te_idx, Mte, q, domain, cols, fcfg, served) -> list[dict]:
    """Representative test-split patients, chosen by a fixed rule (no cherry-picking on correctness): for each
    true pattern, the patient at a fixed quantile of served P(CAD) among test patients with that pattern
    (median; 10th percentile for the low-risk preset). Uncertain: most targets with a {0,1} conformal set."""
    out = []
    used: set[int] = set()

    def pick(mask, key, label, desc, quant=0.5):
        ids = [i for i in np.where(mask)[0] if i not in used]
        if not ids:
            return
        vals = np.array([key[i] for i in ids])
        i = ids[int(np.argsort(vals, kind="mergesort")[int(quant * (len(vals) - 1))])]
        used.add(i)
        out.append({"id": label, "label": desc[0], "description": desc[1] + f" Test-split row {int(te_idx[i])}.",
                    "features": decode_row(pd.Series(Xte[i], index=cols), fcfg),
                    "true_pattern": str(Tte["pattern"].iloc[i])})

    pat = Tte["pattern"].to_numpy()
    pcad = Mte["CAD"]
    pick(pat == "000", pcad, "low-risk", ("Low risk", ("Normal angiography (000); 10th percentile of predicted "
                                               "P(CAD) among such test patients.")), quant=0.1)
    pick(pat == "100", pcad, "lad-only", ("LAD only", "Single-vessel LAD stenosis on angiography (100)."))
    pick(pat == "001", pcad, "rca-only", ("RCA only", "Single-vessel RCA stenosis on angiography (001)."))
    pick(pat == "010", pcad, "lcx-only", ("LCX only", "Single-vessel LCX stenosis on angiography (010)."))
    pick(pat == "111", pcad, "three-vessel", ("Three-vessel disease", "Stenosis of LAD, LCX and RCA (111)."))
    unc = np.array([sum(conformal.status(float(Mte[t][i]), q[t]) == "uncertain" for t in TARGETS)
                    for i in range(len(Xte))])
    pick(unc == unc.max(), -unc + np.abs(pcad - 0.5), "uncertain",
         ("Uncertain case", f"Conformal prediction set is {{0,1}} for {int(unc.max())} of 4 targets."))
    # synthetic out-of-domain example: perturb the low-risk patient to implausible-but-valid combinations
    base = dict(out[0]["features"]) if out else decode_row(pd.Series(Xte[0], index=cols), fcfg)
    ood = dict(base)
    ood.update({"age": 31, "weight": 150, "bmi": 52.0, "fbs": 480, "tg": 1500, "cr": 6.5, "bun": 110, "esr": 120,
                "hb": 7.5, "k": 6.8, "na": 125, "wbc": 32000, "plt": 850, "ef_tte": 15, "bp": 210, "pr": 150})
    xo = np.zeros(len(cols))
    from ml.ingest import encode_value

    fmap = {f["id"]: f for f in fcfg["features"]}
    for fid, v in ood.items():
        for col, val in encode_value(v, fmap[fid]).items():
            xo[cols.index(col)] = val
    score = float(domain.score(xo[None])[0])
    out.append({"id": "synthetic-ood", "label": "Synthetic out-of-domain",
                "description": "SYNTHETIC: a test patient with labs/vitals perturbed to extreme but valid values, "
                               f"to demonstrate the applicability-domain warning (score {score:.2f} vs threshold "
                               f"{domain.threshold_:.2f}). Not a real patient.",
                "features": ood, "true_pattern": None})
    return out


def _sprint_lines(m: dict) -> list[str]:
    sp, hv = m["sprint"], m["cv"]["variants"]["sprint_procedure"]
    d = sp["paired_diff_vs_previous_served"]
    L = ["## Served model: Part 2 sprint procedure (headline)", "",
         (f"{sp['n_screened']} procedures screened on the training split; {sp['n_finalists']} finalists (pre-registered "
          f"rule) went into nested CV ({sp['outer']}; {sp['inner']}); selection happens inside every outer fold, so the "
          "outer score accounts for the search."), "",
         f"Served on the full training split: **{sp['served_on_full_training_split']['chosen']}**.", "",
         "| metric | served procedure (mean ± SD) | previous served model | paired diff (mean ± SD) | folds improved |",
         "|---|---|---|---|---|"]
    prev = sp["previous_served"]
    for t in ["CAD", "LAD", "LCX", "RCA"]:
        k = f"auc_{t}"
        L.append(f"| {t} ROC-AUC | {hv[t]['roc_auc']['mean']:.3f} ± {hv[t]['roc_auc']['sd']:.3f} | "
                 f"{prev[t]['roc_auc']['mean']:.3f} ± {prev[t]['roc_auc']['sd']:.3f} | "
                 f"{d[k]['mean']:+.4f} ± {d[k]['sd']:.4f} | {d[k]['folds_improved']}/{d[k]['n_folds']} |")
    for k, label in (("pattern_log_loss", "pattern log-loss"), ("brier_mean", "mean Brier")):
        L.append(f"| {label} | {hv[k]['mean']:.4f} ± {hv[k]['sd']:.4f} | {prev[k]['mean']:.4f} ± {prev[k]['sd']:.4f} | "
                 f"{d[k]['mean']:+.4f} ± {d[k]['sd']:.4f} | {d[k]['folds_improved']}/{d[k]['n_folds']} |")
    return [*L, ""]


def write_report(m: dict, glob: dict, path: Path, variants: list[str]) -> None:
    L = ["# Model report (generated by `python -m ml.train`)", "",
         ("Decision support and educational use only. Not a substitute for formal diagnostic imaging "
         "(e.g., coronary angiography) or clinical judgment."), "",
         (f"Data SHA-256 `{m['data']['sha256'][:16]}…`, n={m['data']['n']} (train {m['data']['n_train']}, "
         f"test {m['data']['n_test']})."), "", "## Nested CV (5×5 outer, 3 inner) on the training split", "",
         "Mean ± SD over outer folds. Thresholds (max-F1) and calibration fitted on inner OOF predictions.", "",
         ("| variant | CAD AUC | LAD AUC | LCX AUC | RCA AUC | ≥2 AUC | 3VD AUC | mean log-loss | mean Brier | "
         "pattern log-loss | coherence viol. |"), "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in variants:
        s, h = m["cv"]["variants"][v], m["cv"]["headline"][v]

        def f(t, k="roc_auc", s=s):
            return f"{s[t][k]['mean']:.3f} ± {s[t][k]['sd']:.3f}"

        L.append(f"| {v} | {f('CAD')} | {f('LAD')} | {f('LCX')} | {f('RCA')} | {f('P_GE2')} | {f('P_3VD')} | "
                 f"{h['log_loss']['mean']:.3f} | {h['brier']['mean']:.3f} | "
                 f"{s['pattern_log_loss']['mean']:.3f} | {s['coherence_violation_rate']['mean']:.3f} |")
    sel = m["cv"]["selection"]
    L += ["", f"**Selection:** {sel['rule']}. Eligible: {', '.join(sel['eligible'])}. **Chosen: {sel['chosen']}**.",
          "", (f"Engineered features (paired, same folds): Δ log-loss {m['cv']['feature_engineering_paired']['delta_log_loss_fe_minus_base']['mean']:+.4f}, "
          f"Δ AUC {m['cv']['feature_engineering_paired']['delta_roc_auc_fe_minus_base']['mean']:+.4f} → kept: "
          f"{m['cv']['feature_engineering_paired']['kept']}."), "",
          *(_sprint_lines(m) if m.get("sprint") else []),
          ("## Locked test split, 95% bootstrap CIs (second look: after the Part 2 sprint)" if m.get("sprint")
           else "## Locked test split (evaluated once), 95% bootstrap CIs"), "",
          "| model | target | ROC-AUC | PR-AUC | Brier | F1 | sensitivity | specificity |", "|---|---|---|---|---|---|---|---|"]
    for name in ("served", "independent"):
        for t in ["CAD", "LAD", "LCX", "RCA", "P_GE2", "P_3VD"]:
            x = m["test"][name]["metrics"][t]

            def ci(k, x=x):
                return f"{x[k]['value']:.3f} [{x[k]['ci95'][0]:.3f}, {x[k]['ci95'][1]:.3f}]"

            L.append(f"| {name} | {t} | {ci('roc_auc')} | {ci('pr_auc')} | {ci('brier')} | {ci('f1')} | "
                     f"{ci('recall')} | {ci('specificity')} |")
    L += ["", (f"Coherence violation rate on test: served {m['test']['served']['coherence_violation_rate']:.3f}, "
              f"independent {m['test']['independent']['coherence_violation_rate']:.3f}."), "",
          "DeLong (served − independent): " + "; ".join(
              f"{t} Δ={d['delong']['diff']:+.3f} (p={d['delong']['p_value']:.2f})"
              for t, d in m["test"]["comparison_served_vs_independent"].items()), "",
          "## Conformal (90% target), test split", "", ("| target | coverage | class-1 cov. | class-0 cov. | "
          "mean set size | uncertain | empty |"), "|---|---|---|---|---|---|---|"]
    for t, c in m["conformal"]["test"].items():
        L.append(f"| {t} | {c['coverage']:.3f} | {c['coverage_class1']:.3f} | {c['coverage_class0']:.3f} | "
                 f"{c['mean_set_size']:.2f} | {c['frac_uncertain']:.2f} | {c['frac_empty']:.2f} |")
    L += ["", "## Calibration (test)", "", "| target | slope | intercept (CITL) | ECE |", "|---|---|---|---|"]
    for t, c in m["calibration"]["test"].items():
        L.append(f"| {t} | {c['slope']:.2f} | {c['intercept_citl']:+.2f} | {c['ece']:.3f} |")
    L += ["", "## Subgroups (CV OOF ROC-AUC)", "", "| group | n | CAD | LAD | LCX | RCA |", "|---|---|---|---|---|---|"]
    for g, d in m["subgroups"]["groups"].items():
        L.append(f"| {g} | {d['n']} | " + " | ".join(f"{d[t]['roc_auc']:.3f}" for t in TARGETS) + " |")
    L += ["", "## Explanations", "", f"Method {glob['method']}; additivity max error {glob['max_additivity_error']:.2e}.",
          "Agreement of global mean |Owen| with permutation importance (test AUC drop): " + "; ".join(
              f"{t} ρ={a['spearman_rho']:.2f}, top-10 overlap {a['top10_overlap']}"
              for t, a in glob["agreement_shap_vs_permutation"].items()), "",
          "Top CAD features (mean |Owen value|, probability scale): " + ", ".join(
              f"{k} {v:.3f}" for k, v in list(glob["targets"]["CAD"]["mean_abs_shap"].items())[:8]), "",
          ("Figures: `reports/figures/` (cv_auc_comparison, reliability_*, dca_*, learning_curve, deletion_curve, "
          "global_shap_cad).")]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny settings for smoke tests (not for reporting)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args()
    out = a.out or (config.ARTIFACTS if not a.quick else Path(tempfile.gettempdir()) / "coronarylens_quick")
    main(quick=a.quick, out_dir=out, reports=config.REPORTS if not a.quick else out / "reports", n_jobs=a.n_jobs)
