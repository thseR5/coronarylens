import type { AnalyzeResponse, Contribution, ExplainResponse, TargetExplanation } from '../types/api'


/** Fixture matching the API contract; used by tests and the 3D scene before the backend is up. */
export const mockAnalysis: AnalyzeResponse = {
  patterns: { '000': 0.08, '100': 0.12, '010': 0.03, '001': 0.05, '110': 0.14, '101': 0.1, '011': 0.06, '111': 0.42 },
  cad: { p: 0.92, interval: [0.86, 0.96], status: 'likely', reason: 'single_outcome', threshold: 0.55, positive: true },
  vessels: {
    LAD: { p: 0.78, interval: [0.69, 0.86], status: 'likely', reason: 'single_outcome', threshold: 0.5, positive: true },
    LCX: { p: 0.65, interval: [0.52, 0.76], status: 'uncertain', reason: 'both_outcomes', threshold: 0.42, positive: true },
    RCA: { p: 0.63, interval: [0.5, 0.74], status: 'uncertain', reason: 'both_outcomes', threshold: 0.4, positive: true },
  },
  multivessel: { p_ge2: 0.72, p_3vd: 0.42, n_vessels_dist: { '0': 0.08, '1': 0.2, '2': 0.3, '3': 0.42 } },
  applicability: { in_domain: true, score: 4.1, threshold: 9.8 },
  imputed: [],
  input_confidence: { missing_signal_share: 0, level: 'full', forces_uncertain: false },
  out_of_range: [],
  mode: 'f1',
  model_version: 'mock',
}

const mockTarget = (base: number, output: number): TargetExplanation => {
  const contributions: Contribution[] = [
    { feature: 'typical_chest_pain', value: 1, shap: 0.12 * (output - base) / 0.3 },
    { feature: 'age', value: 67, shap: 0.08 * (output - base) / 0.3 },
    { feature: 'region_rwma', value: 2, shap: 0.06 * (output - base) / 0.3 },
    { feature: 'dm', value: 1, shap: 0.05 * (output - base) / 0.3 },
    { feature: 'ef_tte', value: 45, shap: 0.03 * (output - base) / 0.3 },
    { feature: 'atypical', value: 0, shap: 0.02 * (output - base) / 0.3 },
    { feature: 'hdl', value: 52, shap: -0.04 * (output - base) / 0.3 },
    { feature: 'bp', value: 140, shap: 0.02 * (output - base) / 0.3 },
  ]
  const sum = contributions.reduce((a, c) => a + c.shap, 0)
  contributions.push({ feature: 'other', value: null, shap: output - base - sum })
  return {
    base_value: base,
    output,
    contributions,
    group_totals: { demographics: 0.08, risk_factors: 0.05, symptoms_exam: 0.14, ecg: 0.0, labs: -0.02, echo: 0.09 },
    narrative: ['Typical chest pain is the largest factor raising the estimate.', 'Higher HDL lowers it.'],
    narrative_patient: ['Your chest-pain pattern is the main reason the estimate is high.'],
  }
}

export const mockExplanation: ExplainResponse = {
  method: 'mock',
  scale: 'probability',
  targets: {
    CAD: mockTarget(0.71, 0.92),
    LAD: mockTarget(0.58, 0.78),
    LCX: mockTarget(0.39, 0.65),
    RCA: mockTarget(0.38, 0.63),
    P_GE2: mockTarget(0.39, 0.72),
  },
  model_version: 'mock',
}
