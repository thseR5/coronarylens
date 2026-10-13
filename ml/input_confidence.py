"""How much of the model's usual signal comes from cohort averages because inputs are missing.

Each missing (imputed) input is weighted by its global importance for P(CAD): mean |Owen value| over the
SHAP evaluation patients (`global_explanations.json`). The share therefore grows smoothly with *which* inputs are
missing, not with how many: leaving out ten minor labs barely matters, leaving out chest-pain type matters a lot.

    missing_signal_share = sum(importance of imputed inputs) / sum(importance of all inputs)

Levels (also used by the UI to grey the gauge and vessel colours progressively):
  < 0.10  full     - shown normally
  < 0.50  reduced  - colours fade in proportion to the share; statuses forced to "uncertain" from 0.25
  >= 0.50 low      - "estimate based mostly on cohort averages": fully greyed and hatched, statuses uncertain
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

REDUCED_FROM = 0.10
UNCERTAIN_FROM = 0.25
LOW_FROM = 0.50


def missing_signal_share(imputed: Iterable[str], importance: Mapping[str, float]) -> float:
    total = float(sum(importance.values()))
    if total <= 0:
        return 0.0
    missing = sum(float(importance.get(f, 0.0)) for f in set(imputed))
    return min(1.0, max(0.0, missing / total))


def level(share: float) -> str:
    return "low" if share >= LOW_FROM else "reduced" if share >= REDUCED_FROM else "full"


def input_confidence(imputed: Iterable[str], importance: Mapping[str, float]) -> dict:
    s = missing_signal_share(imputed, importance)
    return {"missing_signal_share": s, "level": level(s), "forces_uncertain": s >= UNCERTAIN_FROM}
