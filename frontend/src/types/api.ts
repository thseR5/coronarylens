// Frozen API contract (v1). Mirrors backend/app/schemas.py — change both together.
// All probabilities are on the [0, 1] scale and are marginals of one joint vessel-pattern distribution.

export type Status = 'likely' | 'uncertain' | 'unlikely'
export type ThresholdMode = 'f1' | 'screening'
export type TargetId = 'CAD' | 'LAD' | 'LCX' | 'RCA' | 'P_GE2'
export type FeatureValue = number | string | null
export type Features = Record<string, FeatureValue>

export interface TargetRisk {
  p: number
  interval: [number, number]
  status: Status
  threshold: number
  positive: boolean
}

export interface MultiVessel {
  p_ge2: number
  p_3vd: number
  n_vessels_dist: Record<'0' | '1' | '2' | '3', number>
}

export interface Applicability {
  in_domain: boolean
  score: number
  threshold: number
}

export interface OutOfRange {
  feature: string
  value: number
  range: [number, number]
}

export interface AnalyzeResponse {
  patterns: Record<string, number>
  cad: TargetRisk
  vessels: Record<string, TargetRisk>
  multivessel: MultiVessel
  applicability: Applicability
  imputed: string[]
  out_of_range: OutOfRange[]
  mode: ThresholdMode
  model_version: string
}

export interface Contribution {
  feature: string
  value: FeatureValue
  shap: number
}

export interface TargetExplanation {
  base_value: number
  output: number
  contributions: Contribution[]
  group_totals: Record<string, number>
  narrative: string[]
  narrative_patient: string[]
}

export interface ExplainResponse {
  method: string
  scale: 'probability'
  targets: Record<TargetId, TargetExplanation>
  model_version: string
}

export interface FeatureChange {
  feature: string
  from_value: FeatureValue
  to_value: FeatureValue
}

export interface CounterfactualResponse {
  target: TargetId
  counterfactuals: { changes: FeatureChange[]; p_before: number; p_after: number }[]
  note: string
}

export interface SimilarPatient {
  index: number
  distance: number
  features: Features
  pattern: string
  cad: boolean
}

export interface SimilarResponse {
  neighbors: SimilarPatient[]
}

export interface Preset {
  id: string
  label: string
  description: string
  features: Features
  true_pattern: string | null
}

export interface CohortPoint {
  index: number
  x: number
  y: number
  pattern: string
  n_vessels: number
  cad: boolean
}

export interface CohortMapResponse {
  method: string
  explained_variance: number[]
  points: CohortPoint[]
  patient: { x: number; y: number }
}

export interface TrainingPatient {
  index: number
  features: Features
  pattern: string
  cad: boolean
}
