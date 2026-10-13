"""Stability of the served explanations: recompute Owen-value SHAP with different permutation seeds for every preset
and report the spread of contributions and how stable the top-5 features are. Output: reports/explain_stability.json"""

from __future__ import annotations

import json

import numpy as np

from ml import config
from ml.explain import aggregate_features, owen_values
from ml.joint import EXPLAIN_OUTPUTS
from ml.predictor import ServedModel

SEEDS = list(range(8))


def main() -> dict:
    m = ServedModel(config.ARTIFACTS)
    presets = json.loads((config.ARTIFACTS / "presets.json").read_text("utf-8"))
    out = {"seeds": len(SEEDS), "n_perm": int(m.explain_cfg["n_perm"]), "presets": {}}
    sds, overlaps = [], []
    for p in presets:
        x, _, _ = m.encode(p["features"])
        runs = []
        for s in SEEDS:
            phi, _, _ = owen_values(m.served, x[0], m.background, m.groups, n_perm=int(m.explain_cfg["n_perm"]), seed=s)
            feat = aggregate_features(phi, m.columns, m.fcfg)
            runs.append(np.array([feat[f] for f in sorted(feat)]))  # (n_features, n_outputs)
        R = np.stack(runs)  # (seeds, features, outputs)
        sd = R.std(0).max(0)  # worst-case sd per output (probability scale)
        ref = R.mean(0)
        top = [set(np.argsort(-np.abs(ref[:, k]))[:5]) for k in range(R.shape[2])]
        ov = [np.mean([len(set(np.argsort(-np.abs(r[:, k]))[:5]) & top[k]) / 5 for r in R]) for k in range(R.shape[2])]
        out["presets"][p["id"]] = {"max_sd_by_output": dict(zip(EXPLAIN_OUTPUTS, map(float, sd), strict=True)),
                                   "top5_overlap_by_output": dict(zip(EXPLAIN_OUTPUTS, map(float, ov), strict=True))}
        sds.append(sd.max())
        overlaps.append(min(ov))
    out["summary"] = {"max_contribution_sd": float(max(sds)), "min_top5_overlap": float(min(overlaps))}
    (config.REPORTS / "explain_stability.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(main()["summary"])
