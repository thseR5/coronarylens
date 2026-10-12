import { create } from 'zustand'
import type { AnalyzeResponse, Features } from '../types/api'

/** Pinned baseline for what-if comparison; dashboard-local, the 3D scene does not need it. */
interface WhatIfState {
  baseline: { features: Features; analysis: AnalyzeResponse } | null
  pin: (features: Features, analysis: AnalyzeResponse) => void
  clear: () => void
}

export const useWhatIf = create<WhatIfState>((set) => ({
  baseline: null,
  pin: (features, analysis) => set({ baseline: { features, analysis } }),
  clear: () => set({ baseline: null }),
}))
