"""The served function and everything the API needs, loaded once from ml/artifacts/.

served(X) = marginals(calibrator(joint_ensemble.predict_patterns(X)))
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml import config, conformal
from ml.calibrate import Calibrator
from ml.explain import METHOD, aggregate_features, build_target_explanations, column_groups, owen_values
from ml.ingest import InputError, decode_row, encode_input, encode_value, feature_map
from ml.joint import EXPLAIN_OUTPUTS, PATTERNS, VESSELS, explain_matrix, marginals
from ml.uncertainty import BootstrapEnsemble, centered_intervals

TARGET_KEYS = {"CAD": "cad", "LAD": "LAD", "LCX": "LCX", "RCA": "RCA"}


def load_calibrator(d: dict) -> Calibrator:
    c = Calibrator(d["method"])
    c.T_, c.w_, c.b_ = float(d["T"]), np.array(d["w"]), np.array(d["b"])
    if d.get("W") is not None:
        c.W_ = np.array(d["W"])
    return c


class ServedModel:
    def __init__(self, art_dir: Path | str = config.ARTIFACTS, n_threads: int = 4):
        a = Path(art_dir)
        self.fcfg = config.features_cfg()
        self.tcfg = config.targets_cfg()
        self.model = joblib.load(a / "joint_model.joblib").compile(n_threads)
        self.calibrator = load_calibrator(json.loads((a / "calibrator.json").read_text()))
        self.thresholds = json.loads((a / "thresholds.json").read_text())
        self.conformal = json.loads((a / "conformal.json").read_text())
        pre = json.loads((a / "preprocess.json").read_text())
        self.columns: list[str] = pre["columns"]
        self.imputation: dict[str, Any] = pre["imputation"]
        self.train_ranges: dict[str, list[float]] = pre["train_ranges"]
        boot = joblib.load(a / "bootstrap_ensemble.joblib")
        self.bootstrap = boot if hasattr(boot, "intervals") else BootstrapEnsemble(boot)
        self.domain = joblib.load(a / "domain.joblib")
        self.knn = joblib.load(a / "knn.joblib")
        self.background = np.load(a / "shap_background.npy")
        self.meta = json.loads((a / "model_meta.json").read_text())
        self.explain_cfg = self.meta["explain"]
        self.model_version = self.meta["model_version"]
        self.groups = column_groups(self.columns, self.fcfg)
        self.fmap = feature_map(self.fcfg)

    # ------------------------------------------------------------------ core
    def patterns(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.transform(self.model.predict_patterns(np.asarray(X, dtype=float)))

    def served(self, X: np.ndarray) -> np.ndarray:
        """(n, d) -> (n, 5) [CAD, LAD, LCX, RCA, P_GE2]: the function SHAP explains."""
        return explain_matrix(self.patterns(X))

    def encode(self, features: dict[str, Any]):
        X, imputed, oor = encode_input(features, self.imputation, self.train_ranges, self.fcfg, self.tcfg)
        return X.to_numpy(dtype=float), imputed, oor

    def raw_values(self, features: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for fid in self.fmap:
            v = features.get(fid)
            out[fid] = self.imputation[fid] if v is None or (isinstance(v, str) and not v.strip()) else v
        return out

    # ------------------------------------------------------------------ analyze
    def analyze(self, features: dict[str, Any], mode: str = "f1") -> dict:
        x, imputed, oor = self.encode(features)
        P = self.patterns(x)
        m = marginals(P)
        Pb = self.bootstrap.predict_patterns(x)  # (B, 1, 8) raw bootstrap members
        B = Pb.shape[0]
        Pc = self.calibrator.transform(Pb.reshape(B, -1)).reshape(B, 1, -1)
        iv, _ = centered_intervals(Pc, {t: m[t] for t in ["CAD", *VESSELS]})
        dom = float(self.domain.score(x)[0])

        def risk(t):
            p = float(m[t][0])
            lo, hi = float(iv[t][0, 0]), float(iv[t][0, 1])  # centred bootstrap interval (contains p by design)
            thr = float(self.thresholds[t][mode])
            positive = bool(p >= thr)
            return {"p": p, "interval": (lo, hi), "status": conformal.reconciled_status(p, self.conformal[t], positive),
                    "threshold": thr, "positive": positive}

        nv = {str(n): float(m[f"N{n}"][0]) for n in range(len(VESSELS) + 1)}
        return {
            "patterns": {s: float(P[0, i]) for i, s in enumerate(PATTERNS)},
            "cad": risk("CAD"),
            "vessels": {v: risk(v) for v in VESSELS},
            "multivessel": {"p_ge2": float(m["P_GE2"][0]), "p_3vd": float(m["P_3VD"][0]), "n_vessels_dist": nv},
            "applicability": {"in_domain": bool(dom <= self.domain.threshold_), "score": dom,
                              "threshold": float(self.domain.threshold_)},
            "imputed": imputed,
            "out_of_range": oor,
            "mode": mode,
            "model_version": self.model_version,
        }

    # ------------------------------------------------------------------ explain
    def explain(self, features: dict[str, Any]) -> dict:
        x, _, _ = self.encode(features)
        phi, base, fx = owen_values(self.served, x[0], self.background, self.groups,
                                    n_perm=int(self.explain_cfg["n_perm"]), seed=int(self.explain_cfg["seed"]))
        phi_feat = aggregate_features(phi, self.columns, self.fcfg)
        targets = build_target_explanations(phi_feat, base, fx, self.raw_values(features), self.fcfg)
        return {"method": METHOD, "scale": "probability", "targets": targets, "model_version": self.model_version}

    # ------------------------------------------------------------------ similar
    def similar(self, features: dict[str, Any], k: int = 5) -> dict:
        x, _, _ = self.encode(features)
        out = []
        for nb in self.knn.query(x[0], k):
            row = pd.Series(self.knn.X_[nb["pos"]], index=self.columns)
            out.append({"index": nb["index"], "distance": nb["distance"], "features": decode_row(row, self.fcfg),
                        "pattern": nb["pattern"], "cad": nb["cad"]})
        return {"neighbors": out}

    # ------------------------------------------------------------------ cohort map
    def _cohort_pca(self):
        """Deterministic 2-D PCA of the standardized TRAINING split (test patients never shown)."""
        if getattr(self, "_pca", None) is None:
            from sklearn.decomposition import PCA

            Z = self.knn.sc_.transform(self.knn.X_)
            self._pca = PCA(2, svd_solver="full").fit(Z)
            xy = self._pca.transform(Z)
            n_vessels = np.array([p.count("1") for p in self.knn.patterns_])
            self._cohort_points = [
                {"index": int(i), "x": float(a), "y": float(b), "pattern": str(p), "n_vessels": int(n), "cad": bool(c)}
                for i, (a, b), p, n, c in zip(self.knn.row_index_, xy, self.knn.patterns_, n_vessels, self.knn.cad_,
                                              strict=True)]
        return self._pca

    def cohort_map(self, features: dict[str, Any]) -> dict:
        pca = self._cohort_pca()
        x, _, _ = self.encode(features)
        px, py = pca.transform(self.knn.sc_.transform(x))[0]
        return {"method": "PCA (2 components) of standardized features, training split",
                "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
                "points": self._cohort_points, "patient": {"x": float(px), "y": float(py)}}

    def training_patient(self, index: int) -> dict:
        pos = np.flatnonzero(self.knn.row_index_ == int(index))
        if pos.size == 0:
            raise KeyError(index)
        row = pd.Series(self.knn.X_[pos[0]], index=self.columns)
        return {"index": int(index), "features": decode_row(row, self.fcfg),
                "pattern": str(self.knn.patterns_[pos[0]]), "cad": bool(self.knn.cad_[pos[0]])}

    # ------------------------------------------------------------------ counterfactual
    def _candidates(self, raw: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        """Per modifiable unit: candidate value changes, within plausibility range, towards the ref range."""
        cands: dict[str, list[dict[str, Any]]] = {}
        h = float(raw["height"]) / 100.0
        for fid, f in self.fmap.items():
            if not f.get("modifiable") or fid == "weight":
                continue  # weight is coupled to BMI (height fixed): one unit, handled under bmi
            cur = raw[fid]
            opts: list[dict[str, Any]] = []
            if f["type"] == "binary":
                good = 0 if f.get("monotone", 0) >= 0 else 1
                if float(cur) != good:
                    opts.append({fid: good})
            elif f["type"] == "numeric" and f.get("ref"):
                lo_r, hi_r = (float(v) for v in f["ref"])
                lo_p, hi_p = (float(v) for v in f["range"])
                cur = float(cur)
                vals = []
                if cur > hi_r:
                    vals = [hi_r, (cur + hi_r) / 2]
                elif cur < lo_r:
                    vals = [lo_r, (cur + lo_r) / 2]
                for v in vals:
                    v = float(np.clip(round(v, 1), lo_p, hi_p))
                    if abs(v - cur) > 1e-9:
                        ch = {fid: v}
                        if fid == "bmi":
                            ch["weight"] = float(np.clip(round(v * h * h, 1), *self.fmap["weight"]["range"]))
                        opts.append(ch)
            if opts:
                cands[fid] = opts
        return cands

    def counterfactual(self, features: dict[str, Any], target: str = "CAD", max_changes: int = 3) -> dict:
        x, _, _ = self.encode(features)
        raw = self.raw_values(features)
        k = EXPLAIN_OUTPUTS.index(target)
        p0 = float(self.served(x)[0, k])
        cands = self._candidates(raw)
        units = sorted(cands)
        combos: list[dict[str, Any]] = []
        for r in range(1, max_changes + 1):
            for subset in itertools.combinations(units, r):
                for choice in itertools.product(*(cands[u] for u in subset)):
                    ch: dict[str, Any] = {}
                    for c in choice:
                        ch.update(c)
                    combos.append(ch)
        if not combos:
            return {"target": target, "counterfactuals": [], "note": "Model sensitivity, not treatment advice."}
        rows = []
        for ch in combos:
            r = dict(raw)
            r.update(ch)
            xr = x[0].copy()
            for fid, v in ch.items():
                for col, val in encode_value(v, self.fmap[fid]).items():
                    xr[self.columns.index(col)] = val
            rows.append(xr)
        p = self.served(np.array(rows))[:, k]
        best: dict[int, tuple[float, dict]] = {}
        for ch, pa in zip(combos, p, strict=True):
            n_units = len({"bmi" if f == "weight" else f for f in ch})
            if pa < p0 - 1e-4 and (n_units not in best or pa < best[n_units][0]):
                best[n_units] = (float(pa), ch)
        out = []
        for n_units in sorted(best):
            pa, ch = best[n_units]
            out.append({"changes": [{"feature": f, "from_value": raw[f], "to_value": v} for f, v in ch.items()],
                        "p_before": p0, "p_after": pa})
        return {"target": target, "counterfactuals": out, "note": "Model sensitivity, not treatment advice."}


__all__ = ["InputError", "ServedModel"]
