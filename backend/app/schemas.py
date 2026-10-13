"""Frozen API contract (v1). Mirrored by frontend/src/types/api.ts — change both together.

All probabilities are on the probability scale [0, 1]. Every displayed probability is a marginal of
one calibrated joint distribution over the 2^k vessel patterns (see config/targets.yaml).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

API_VERSION = "1"

Status = Literal["likely", "uncertain", "unlikely"]
ThresholdMode = Literal["f1", "screening"]
StatusReason = Literal[
    "single_outcome", "both_outcomes", "empty_set", "threshold_disagrees", "out_of_distribution", "missing_inputs"
]
TargetId = Literal["CAD", "LAD", "LCX", "RCA", "P_GE2"]

# Feature values: numeric for numeric/binary/ordinal features, string for categorical (e.g. bbb = "LBBB").
# Ordinal string-coded features (vhd) accept either the raw level ("Moderate") or its code (2).
FeatureValue = float | int | str | None


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- requests
class AnalyzeRequest(_Model):
    features: dict[str, FeatureValue] = Field(
        description="Feature id -> value. Missing or null features are imputed and listed in `imputed`."
    )
    mode: ThresholdMode = "f1"


class ExplainRequest(_Model):
    features: dict[str, FeatureValue]


class CounterfactualRequest(_Model):
    features: dict[str, FeatureValue]
    target: TargetId = "CAD"
    max_changes: int = Field(default=3, ge=1, le=3)


class SimilarRequest(_Model):
    features: dict[str, FeatureValue]
    k: int = Field(default=5, ge=1, le=20)


# ---------------------------------------------------------------- analyze
class TargetRisk(_Model):
    p: float = Field(ge=0, le=1)
    interval: tuple[float, float] = Field(description="Bootstrap 90% interval for p")
    status: Status = Field(description="Conformal prediction set: {1}=likely, {0,1}=uncertain, {0}=unlikely")
    reason: StatusReason = Field(
        description="Why this status: single_outcome (90% set is one outcome, agrees with threshold), both_outcomes, "
        "empty_set, threshold_disagrees, out_of_distribution, missing_inputs (>= 25% of signal imputed)"
    )
    threshold: float = Field(ge=0, le=1, description="Operating threshold for the requested mode")
    positive: bool = Field(description="p >= threshold")


class MultiVessel(_Model):
    p_ge2: float = Field(ge=0, le=1)
    p_3vd: float = Field(ge=0, le=1)
    n_vessels_dist: dict[str, float] = Field(description='{"0": p, "1": p, "2": p, "3": p}, sums to 1')


class Applicability(_Model):
    in_domain: bool
    score: float = Field(description="Distance score; higher = less like the training data")
    threshold: float = Field(description="99th percentile of training scores")


class OutOfRange(_Model):
    feature: str
    value: float
    range: tuple[float, float]


class InputConfidence(_Model):
    missing_signal_share: float = Field(
        ge=0, le=1, description="Share of the model's usual signal (global mean |SHAP| for CAD) from imputed inputs"
    )
    level: Literal["full", "reduced", "low"] = Field(description="full < 0.10 <= reduced < 0.50 <= low")
    forces_uncertain: bool = Field(description="share >= 0.25: every status is shown as uncertain")


class AnalyzeResponse(_Model):
    patterns: dict[str, float] = Field(description='Pattern string (bit order from targets.yaml, e.g. "101") -> p')
    cad: TargetRisk
    vessels: dict[str, TargetRisk] = Field(description="Vessel id -> risk, keys from targets.yaml")
    multivessel: MultiVessel
    applicability: Applicability
    imputed: list[str] = Field(description="Feature ids that were missing and imputed with training median/mode")
    input_confidence: InputConfidence
    out_of_range: list[OutOfRange] = Field(description="Values outside the training data range (accepted, flagged)")
    mode: ThresholdMode
    model_version: str


# ---------------------------------------------------------------- explain
class Contribution(_Model):
    feature: str = Field(description="Feature id (or engineered/group id)")
    value: FeatureValue = Field(description="Patient's raw input value")
    shap: float = Field(description="Signed contribution on the probability scale")


class TargetExplanation(_Model):
    base_value: float = Field(description="Expected output over the background set")
    output: float = Field(description="base_value + sum(contributions) (additivity tested)")
    contributions: list[Contribution] = Field(description="All features, sorted by |shap| desc")
    group_totals: dict[str, float] = Field(description="Group id (features.yaml) -> summed shap")
    narrative: list[str] = Field(description="Deterministic template sentences, clinician wording")
    narrative_patient: list[str] = Field(description="Deterministic template sentences, plain language")


class ExplainResponse(_Model):
    method: str = Field(description='e.g. "owen-partition" or "permutation-grouped"')
    scale: Literal["probability"] = "probability"
    targets: dict[str, TargetExplanation] = Field(description="Keys: CAD, LAD, LCX, RCA, P_GE2")
    model_version: str


# ---------------------------------------------------------------- counterfactual / similar
class FeatureChange(_Model):
    feature: str
    from_value: FeatureValue
    to_value: FeatureValue


class Counterfactual(_Model):
    changes: list[FeatureChange]
    p_before: float
    p_after: float


class CounterfactualResponse(_Model):
    target: TargetId
    counterfactuals: list[Counterfactual]
    note: str = "Model sensitivity, not treatment advice."


class SimilarPatient(_Model):
    index: int = Field(description="Row index in the training split")
    distance: float
    features: dict[str, FeatureValue]
    pattern: str = Field(description='True angiography vessel pattern, e.g. "110"')
    cad: bool = Field(description="True Cath label")


class SimilarResponse(_Model):
    neighbors: list[SimilarPatient]


# ---------------------------------------------------------------- cohort map
class CohortPoint(_Model):
    index: int = Field(description="Row index in the source data (training split only)")
    x: float
    y: float
    pattern: str = Field(description="True angiography vessel pattern")
    n_vessels: int
    cad: bool


class CohortMapResponse(_Model):
    method: str
    explained_variance: list[float]
    points: list[CohortPoint]
    patient: dict[str, float] = Field(description='{"x": .., "y": ..} of the current patient, projected')


class TrainingPatient(_Model):
    index: int
    features: dict[str, FeatureValue]
    pattern: str
    cad: bool


# ---------------------------------------------------------------- misc
class Preset(_Model):
    id: str
    label: str
    description: str
    features: dict[str, FeatureValue]
    true_pattern: str | None = Field(default=None, description="Actual angiography pattern (test split)")


class HealthResponse(_Model):
    status: Literal["ok"]
    api_version: str = API_VERSION
    model_version: str
