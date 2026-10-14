"""Check that every number in the judge-facing docs and the dashboard text is backed by a measured artifact.

1. Every numeric leaf of the measured JSON artifacts (metrics, sprint summaries, latency, stability, what-if,
   benchmarks, territory agreement, anatomy manifest) is rendered the ways the docs write numbers: 1-3 decimals,
   percentages with 0-1 decimals, and integers.
2. Every decimal number (e.g. 0.934, 1.508) and every percentage (e.g. 54%, 5.4%) in the checked files must match one
   of those renderings, or a documented constant in CONSTANTS (design choices, versions, sizes - not results).
3. CLAIMS pins the headline numbers to their exact JSON value, so a stale headline fails even if the stale value
   happens to exist elsewhere in the artifacts.

Usage: uv run python tools/check_numbers.py   (exit 1 and a list of unmatched numbers on failure)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    "ml/artifacts/metrics.json",
    "ml/artifacts/ablation.json",
    "ml/artifacts/thresholds.json",
    "ml/artifacts/model_meta.json",
    "reports/sprint/coherence_posthoc.json",
    "reports/sprint/finalists.json",
    "reports/sprint/test_first_look.json",
    "reports/sprint/within_cad_dedicated.json",
    "reports/explain_stability.json",
    "reports/whatif/whatif.json",
    "reports/benchmarks/naive_lr.json",
    "reports/benchmarks/regularized_lr.json",
    "docs/latency.json",
    "frontend/public/models/territory_agreement.json",
    "frontend/public/models/anatomy_manifest.json",
]

FILES = [
    "README.md",
    "docs/REPORT.md",
    # submission material kept outside the repo (checked when present on this machine)
    "../coronarylens-private/docs/DEVPOST.md",
    "../coronarylens-private/docs/JUDGE_QA.md",
    "../coronarylens-private/docs/DEMO_SCRIPT.md",
    "docs/MODEL_CARD.md",
    "frontend/src/dashboard/*.tsx",
    "frontend/src/lib/*.ts",
    "frontend/src/three/Legend.tsx",
    "backend/app/report.py",
]

# Design choices, versions and fixed settings (not results). Each entry: value -> why it may appear.
CONSTANTS = {
    "0.90": "screening sensitivity target / conformal 90% level",
    "90%": "conformal coverage target, screening sensitivity target",
    "95%": "confidence-interval level",
    "0.95": "proxy-scan AUC threshold / CI level",
    "0.5": "chance AUC",
    "0.50": "input-confidence 'low' threshold",
    "50%": "input-confidence 'low' threshold",
    "0.25": "input-confidence 'forced uncertain' threshold",
    "25%": "input-confidence 'forced uncertain' threshold",
    "0.10": "input-confidence 'reduced' threshold",
    "10%": "input-confidence 'reduced' threshold",
    "0.02": "what-if direction tolerance",
    "0.15": "what-if jump threshold",
    "0.30": "what-if: binary features move P(CAD) by up to 0.30 (whatif.json max step, checked in CLAIMS)",
    "0.3": "elastic-net C",
    "4.0": "BodyParts3D 4.0 / CC BY 4.0",
    "1.9.1": "scikit-learn version",
    "1.2.10": "CatBoost version",
    "3.12": "Python version",
    "1.5": "max device pixel ratio",
    "0.45": "anatomy.glb size in MB",
    "2.47": "Docker image size in GB",
    "4.44": "Docker image size before slimming (GB)",
    "4.4": "FastSVC speed-up (measured in tests/benchmarks, see DECISIONS)",
    "3.7": "corrected/naive SE ratio sqrt(1 + J*n_test/n_train) = sqrt(1 + 50*48.4/193.6)",
    "100%": "percentage scale end / share of all inputs",
    "0%": "percentage scale start / coherence 0% by construction",
    "80%": "train share of the 80/20 split",
    "20%": "test share of the 80/20 split",
    "13%": "cohort-map PCA variance (logged in DECISIONS)",
    "97.7%": "accuracy reported by a 2025 study (cited, not ours)",
    "74.2": "published accuracy (Alizadehsani 2012)",
    "63.8": "published accuracy (Alizadehsani 2012)",
    "68.3": "published accuracy (Alizadehsani 2012)",
    "79.5": "published accuracy (Alizadehsani 2013)",
    "65.1": "published accuracy (Alizadehsani 2013)",
    "68.0": "published accuracy (Alizadehsani 2013)",
    "86.6": "published accuracy (Alizadehsani 2020)",
    "83.5": "published accuracy (Alizadehsani 2020)",
    "82.9": "published accuracy (Alizadehsani 2020)",
    "10.24432": "dataset DOI prefix",
    "10.1111": "DOI prefix in a reference",
    "10.3389": "DOI prefix in a reference",
    "0.85": "slider span when a feature has no reference range (code constant)",
    "1.15": "slider span when a feature has no reference range (code constant)",
    "18.365473": "code-comment example of rounding (RiskSummary.tsx)",
    "18.4": "code-comment example of rounding (RiskSummary.tsx)",
    "0.0": "additivity error / zero",
    "1.0": "probability one",
    "1.1": "API/report version strings",
    "2.1": "legacy OBJ header licence text (CC BY-SA 2.1 Japan)",
    "97–99%": "range of published accuracies",
    "97": "range of published accuracies",
    "99%": "range of published accuracies",
}

# Headline claims pinned to exact JSON values: (text that must appear in a checked file, file, path, format).
# The number inside `text` must equal format(value); a stale headline fails even if its old value exists elsewhere.
M, ABL, LAT = "ml/artifacts/metrics.json", "ml/artifacts/ablation.json", "docs/latency.json"
NEST = "cv/variants/sprint_procedure"
WC = "sprint/within_cad_localisation/within_cad_auc_ci"
CLAIMS = [
    ("CAD 0.934", M, f"{NEST}/CAD/roc_auc/mean", "{:.3f}"),
    ("LAD 0.866", M, f"{NEST}/LAD/roc_auc/mean", "{:.3f}"),
    ("LCX 0.708", M, f"{NEST}/LCX/roc_auc/mean", "{:.3f}"),
    ("RCA 0.737", M, f"{NEST}/RCA/roc_auc/mean", "{:.3f}"),
    ("0.934 [0.896, 0.973]", M, f"{NEST}/CAD/roc_auc/ci95/0", "{:.3f}", 1),
    ("0.934 [0.896, 0.973]", M, f"{NEST}/CAD/roc_auc/ci95/1", "{:.3f}", 2),
    *[("CAD 0.93 · LAD 0.87 · LCX 0.71 · RCA 0.74", M, f"{NEST}/{t}/roc_auc/mean", "{:.2f}", i)
      for i, t in enumerate(("CAD", "LAD", "LCX", "RCA"))],
    ("1.508", M, f"{NEST}/pattern_log_loss/mean", "{:.3f}"),
    ("CAD 0.898", M, "test/served/metrics/CAD/roc_auc/value", "{:.3f}"),
    ("LAD 0.829", M, "test/served/metrics/LAD/roc_auc/value", "{:.3f}"),
    ("LCX 0.761", M, "test/served/metrics/LCX/roc_auc/value", "{:.3f}"),
    ("RCA 0.661", M, "test/served/metrics/RCA/roc_auc/value", "{:.3f}"),
    ("0.69 [0.59–0.78]", M, f"{WC}/LAD/mean", "{:.2f}"),
    ("0.52 [0.42–0.61]", M, f"{WC}/LCX/mean", "{:.2f}"),
    ("0.57 [0.47–0.66]", M, f"{WC}/RCA/mean", "{:.2f}"),
    ("p = 0.0003", M, "sprint/vs_independent_same_folds/pattern_log_loss/p", "{:.4f}"),
    ("p = 0.049", M, "sprint/vs_independent_same_folds/brier_mean/p", "{:.3f}"),
    ("p = 0.002", M, "sprint/paired_diff_vs_previous_served/pattern_log_loss/p", "{:.3f}"),
    ("748", M, "sprint/n_screened", "{:.0f}"),
    ("32.1%", M, "coherence/naive_lr/mean", "{:.1%}"),
    ("5.4%", M, "sprint/vs_independent_same_folds/coherence_independent/mean", "{:.1%}"),
    ("0.930 → 0.968", ABL, "targets/CAD/D_correct_selection_and_smote_in_fold/roc_auc/mean", "{:.3f}"),
    ("0.930 → 0.968", ABL, "targets/CAD/A_smote_and_selection_before_cv/roc_auc/mean", "{:.3f}", 1),
    ("226 ms", LAT, "runs_after_judge_fixes/0/analyze_path_ms/p95", "{:.0f}"),
    ("360 ms", LAT, "runs_after_judge_fixes/0/explain_path_ms/p95", "{:.0f}"),
]

NUM = re.compile(r"(?<![\w.#/-])(\d+(?:\.\d+)?%|\d+\.\d+)(?![\w.]*\d)")


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        if len(x) > 4 and all(isinstance(v, (int, float)) for v in x):
            return  # curves, bins and per-fold vectors: plotted, never quoted; skipping keeps the check strict
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}/{i}")
    elif isinstance(x, bool):
        return
    elif isinstance(x, (int, float)):
        yield path, float(x)


def renderings(v: float) -> set[str]:
    out = set()
    for d in (1, 2, 3, 4):
        out.add(f"{v:.{d}f}")
        out.add(f"{abs(v):.{d}f}")
    for d in (0, 1):
        out.add(f"{v * 100:.{d}f}")  # e.g. accuracy 0.805 written as 80.5 in a "%" table
        out.add(f"{v * 100:.{d}f}%")
        out.add(f"{abs(v) * 100:.{d}f}%")
        out.add(f"{v:.{d}f}%")  # values already stored in percent
    if abs(v) >= 1 and float(v).is_integer():
        out.add(f"{int(v)}%")
    return out


def derived() -> dict[str, float]:
    """Values the docs quote that are computed from stored records rather than stored directly."""
    import numpy as np

    out = {}
    ind = json.loads((ROOT / "reports/sprint/indep_B1.json").read_text("utf-8"))["folds"]
    for k in ("pattern_log_loss", "brier_mean", "auc_CAD", "auc_LAD", "auc_LCX", "auc_RCA", "coherence"):
        out[f"independent same folds: mean {k}"] = float(np.mean([f[k] for f in ind]))
    m = json.loads((ROOT / "ml/artifacts/metrics.json").read_text("utf-8"))
    sc = m["sprint"]["served_on_full_training_split"]["inner_scores"]
    best = sorted(sc.values())
    out["inner log-loss margin, chosen vs runner-up"] = best[1] - best[0]
    return out


TARGET_CTX = re.compile(r"\b(CAD|LAD|LCX|RCA)\W{1,3}$")
P_CTX = re.compile(r"\bp\s*(?:=|<|≤)\s*$")


def allowed() -> dict[str, set[str]]:
    """Renderings allowed globally ("*"), per target ("CAD", ...) and for p-values ("p")."""
    ok: dict[str, set[str]] = {"*": set(CONSTANTS), "p": set(), "CAD": set(), "LAD": set(), "LCX": set(), "RCA": set()}
    for src in SOURCES:
        data = json.loads((ROOT / src).read_text("utf-8"))
        for path, v in leaves(data):
            r = renderings(v)
            ok["*"] |= r
            parts = re.split(r"[/_]", path)
            for t in ("CAD", "LAD", "LCX", "RCA"):
                if t in parts or t.lower() in parts:
                    ok[t] |= r
            if parts[-1] in ("p", "p_value", "p_vs_chance"):
                ok["p"] |= r
    for name, v in derived().items():
        r = renderings(v)
        ok["*"] |= r
        for t in ("CAD", "LAD", "LCX", "RCA"):
            if t in name:
                ok[t] |= r
    return ok


def context(prefix: str) -> str:
    if P_CTX.search(prefix):
        return "p"
    m = TARGET_CTX.search(prefix)
    return m.group(1) if m else "*"


def get(data, path: str):
    for k in path.split("/"):
        data = data[int(k)] if isinstance(data, list) else data[k]
    return data


def check_claims(texts: dict[str, str]) -> list[str]:
    """Each claim: the n-th number in `text` (default first) must equal format(value from JSON) and `text` must be used."""
    errs = []
    for claim in CLAIMS:
        text, src, path, fmt = claim[:4]
        idx = claim[4] if len(claim) > 4 else 0
        v = float(get(json.loads((ROOT / src).read_text("utf-8")), path))
        expected = fmt.format(v)
        numbers = re.findall(r"\d+(?:\.\d+)?%?", text)
        stated = numbers[idx] if idx < len(numbers) else None
        if stated != expected:
            errs.append(f"claim {text!r}: states {stated}, JSON {src}:{path} = {v} -> {expected}")
        if not any(text in t for t in texts.values()):
            errs.append(f"claim {text!r} not found in any checked file")
    return errs


def main() -> int:
    ok = allowed()
    texts = {}
    for pat in FILES:
        for f in sorted(ROOT.glob(pat)):
            texts[str(f.relative_to(ROOT))] = f.read_text("utf-8")
    bad = []
    for name, t in texts.items():
        for i, line in enumerate(t.splitlines(), 1):
            if name.endswith((".tsx", ".ts", ".py")) and re.search(r"className=|import |toFixed|\bpx\b|rgba?\(|#[0-9a-f]{3,6}", line):
                line = re.sub(r'className="[^"]*"|className=\{`[^`]*`\}|rgba?\([^)]*\)|style=\{\{[^}]*\}\}', "", line)
            for m in NUM.finditer(line):
                s, ctx = m.group(1), context(line[: m.start()])
                if s not in ok[ctx] and s not in CONSTANTS:
                    bad.append(f"{name}:{i}: {s} [{ctx}]  | {line.strip()[:140]}")
    errs = check_claims(texts)
    for e in errs:
        print("CLAIM", e)
    for b in bad:
        print("UNMATCHED", b)
    print(f"checked {len(texts)} files; {len(bad)} unmatched numbers; {len(errs)} claim errors")
    return 1 if bad or errs else 0


if __name__ == "__main__":
    sys.exit(main())
