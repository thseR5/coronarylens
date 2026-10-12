"""What-if plausibility check of the SERVED model: sweep key clinical features across their range for several presets
and record P(CAD) and each vessel probability. Flags
  - direction: the overall change (end - start) has the opposite sign to the clinical expectation (`monotone` in
    features.yaml), larger than a tolerance;
  - jumps: one grid step changes a probability by more than JUMP (would look odd as a live colour jump).
Outputs reports/whatif/whatif.json, reports/whatif/whatif_<preset>.png and a summary for the docs.

Usage: uv run python -m ml.whatif_check
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml import config
from ml.predictor import ServedModel

FEATURES = ["age", "typical_chest_pain", "dm", "htn", "st_depression", "t_inversion", "ef_tte", "region_rwma", "bp",
            "ldl", "fbs"]
OUTPUTS = ["CAD", "LAD", "LCX", "RCA"]
PRESETS = ["low-risk", "lad-only", "rca-only", "three-vessel", "uncertain"]
JUMP = 0.15  # max acceptable change in one grid step (probability scale)
DIR_TOL = 0.02  # a reversal smaller than this is treated as flat
GRID_N = 25


def grid(f: dict, lo_hi: tuple[float, float]) -> np.ndarray:
    if f["type"] == "binary":
        return np.array([0, 1])
    if f["type"] == "ordinal":
        return np.arange(len(f["levels"])) if not isinstance(f["levels"][0], (int, float)) else np.array(f["levels"])
    return np.linspace(lo_hi[0], lo_hi[1], GRID_N)


def main() -> dict:
    m = ServedModel(config.ARTIFACTS)
    fmap = {f["id"]: f for f in config.features_cfg()["features"]}
    presets = {p["id"]: p for p in json.loads((config.ARTIFACTS / "presets.json").read_text("utf-8"))}
    prep = json.loads((config.ARTIFACTS / "preprocess.json").read_text("utf-8"))
    ranges = prep.get("train_ranges", {})  # observed training-data range (sweeps stay inside the data)
    out_dir = config.REPORTS / "whatif"
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"jump_threshold": JUMP, "direction_tolerance": DIR_TOL, "presets": PRESETS, "features": FEATURES,
                 "curves": {}, "flags": []}
    for pid in PRESETS:
        base = dict(presets[pid]["features"])
        fig, axes = plt.subplots(3, 4, figsize=(11, 7.2), dpi=110)
        res["curves"][pid] = {}
        for ax, fid in zip(axes.ravel(), FEATURES + [None], strict=True):
            if fid is None:
                ax.axis("off")
                continue
            f = fmap[fid]
            r = ranges.get(fid) or f["range"]
            xs = grid(f, (float(r[0]), float(r[1])))
            rows = []
            for x in xs:
                feats = dict(base)
                feats[fid] = f["levels"][int(x)] if f["type"] == "ordinal" and isinstance(f["levels"][0], str) else \
                    (int(x) if f["type"] in ("binary", "ordinal") else float(x))
                a = m.analyze(feats, "f1")
                rows.append([a["cad"]["p"]] + [a["vessels"][v]["p"] for v in ["LAD", "LCX", "RCA"]])
            P = np.array(rows)
            res["curves"][pid][fid] = {"x": xs.tolist(), **{o: P[:, i].round(4).tolist() for i, o in enumerate(OUTPUTS)}}
            mono = int(f.get("monotone", 0))
            for i, o in enumerate(OUTPUTS):
                ax.plot(xs, P[:, i], label=o, lw=1.4, marker="o" if len(xs) <= 5 else None, ms=3)
                delta = float(P[-1, i] - P[0, i])
                steps = np.abs(np.diff(P[:, i]))
                if mono != 0 and delta * mono < -DIR_TOL:
                    res["flags"].append({"preset": pid, "feature": fid, "output": o, "kind": "direction",
                                         "expected": "increase" if mono > 0 else "decrease", "delta": round(delta, 3)})
                if f["type"] == "numeric" and steps.max() > JUMP:
                    res["flags"].append({"preset": pid, "feature": fid, "output": o, "kind": "jump",
                                         "max_step": round(float(steps.max()), 3),
                                         "at": float(xs[int(steps.argmax())])})
            ax.set_title(f"{f['label']}" + (f" (expected {'↑' if mono > 0 else '↓'})" if mono else ""), fontsize=8)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7)
        axes.ravel()[0].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"What-if curves, served model, preset '{presets[pid]['label']}'", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"whatif_{pid}.png")
        plt.close(fig)
    (out_dir / "whatif.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = main()
    print(f"{len(r['flags'])} flags")
    for fl in r["flags"]:
        print(" ", fl)
