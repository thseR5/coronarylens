"""Apply the pre-registered finalist-selection rule (docs/EXPERIMENTS.md) to the screening results.

Usage: uv run python -m ml.sprint.finalists A1 A2  -> reports/sprint/finalists.json (+ printed table)
"""

from __future__ import annotations

import json
import sys

import numpy as np

from ml import config

OUT = config.REPORTS / "sprint"
BASELINE = "lr_en+rf+et|avg|vector"


def parse(name: str) -> dict:
    parts = name.split("|")
    d = {"learners": parts[0].split("+"), "combine": parts[1], "calib": parts[2]}
    for p in parts[3:]:
        if p.startswith("count"):
            d["count_lambda"] = float(p[5:])
        else:
            d["features"] = p
    return d


def main(tags: list[str]) -> dict:
    pool: dict[str, dict] = {}
    for t in tags:
        screen = json.loads((OUT / f"screen_{t}.json").read_text("utf-8"))
        for name, s in screen["procedures"].items():
            if name not in pool or s["pattern_log_loss"]["mean"] < pool[name]["pattern_log_loss"]["mean"]:
                pool[name] = {k: v for k, v in s.items() if not k.startswith("_")} | {"_tag": t}

    def ll(n):
        return pool[n]["pattern_log_loss"]["mean"]

    def auc(n, ts):
        return float(np.mean([pool[n][f"auc_{t}"]["mean"] for t in ts]))

    chosen: list[tuple[str, str]] = [(n, "top-8 log-loss") for n in sorted(pool, key=ll)[:8]]
    names = {n for n, _ in chosen}
    best_lr = max(pool, key=lambda n: auc(n, ["LCX", "RCA"]))
    if best_lr not in names:
        chosen.append((best_lr, "best LCX+RCA AUC"))
        names.add(best_lr)
    best_all = max(pool, key=lambda n: auc(n, ["CAD", "LAD", "LCX", "RCA"]))
    if best_all not in names:
        chosen.append((best_all, "best mean AUC"))
        names.add(best_all)
    if BASELINE not in names:
        chosen.append((BASELINE, "baseline (served recipe)"))
    out = {"n_screened": len(pool), "tags": tags, "rule": "see docs/EXPERIMENTS.md 'Finalist-selection rule'",
           "finalists": [{"name": n, "reason": r, **parse(n),
                          "screen": {"pattern_log_loss": ll(n), "auc_CAD": pool[n]["auc_CAD"]["mean"],
                                     "auc_LAD": pool[n]["auc_LAD"]["mean"], "auc_LCX": pool[n]["auc_LCX"]["mean"],
                                     "auc_RCA": pool[n]["auc_RCA"]["mean"], "from": pool[n]["_tag"]}}
                         for n, r in chosen]}
    (OUT / "finalists.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (OUT / "finalists_candidates.json").write_text(
        json.dumps([{k: f[k] for k in ("learners", "combine", "calib", "count_lambda", "features") if k in f}
                    for f in out["finalists"]], indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = main(sys.argv[1:] or ["A1", "A2"])
    print(f"{o['n_screened']} procedures screened; {len(o['finalists'])} finalists")
    for f in o["finalists"]:
        s = f["screen"]
        print(f"  {f['name']:42s} {f['reason']:24s} ll {s['pattern_log_loss']:.4f}  CAD {s['auc_CAD']:.3f} "
              f"LAD {s['auc_LAD']:.3f} LCX {s['auc_LCX']:.3f} RCA {s['auc_RCA']:.3f} ({s['from']})")
