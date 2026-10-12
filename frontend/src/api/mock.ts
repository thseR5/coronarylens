import type { AnalyzeResponse } from '../types/api'

/** Fixture matching the API contract; used by tests and the 3D scene before the backend is up. */
export const mockAnalysis: AnalyzeResponse = {
  patterns: { '000': 0.08, '100': 0.12, '010': 0.03, '001': 0.05, '110': 0.14, '101': 0.1, '011': 0.06, '111': 0.42 },
  cad: { p: 0.92, interval: [0.86, 0.96], status: 'likely', threshold: 0.55, positive: true },
  vessels: {
    LAD: { p: 0.78, interval: [0.69, 0.86], status: 'likely', threshold: 0.5, positive: true },
    LCX: { p: 0.65, interval: [0.52, 0.76], status: 'uncertain', threshold: 0.42, positive: true },
    RCA: { p: 0.63, interval: [0.5, 0.74], status: 'uncertain', threshold: 0.4, positive: true },
  },
  multivessel: { p_ge2: 0.72, p_3vd: 0.42, n_vessels_dist: { '0': 0.08, '1': 0.2, '2': 0.3, '3': 0.42 } },
  applicability: { in_domain: true, score: 4.1, threshold: 9.8 },
  imputed: [],
  out_of_range: [],
  mode: 'f1',
  model_version: 'mock',
}
