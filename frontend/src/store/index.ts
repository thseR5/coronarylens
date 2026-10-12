import { create } from 'zustand'
import type { AnalyzeResponse, ExplainResponse, Features, TargetId, ThresholdMode } from '../types/api'
import type { AppConfig } from '../types/config'

/**
 * One store drives both the 3D scene and the dashboard, so they can never disagree.
 * The 3D scene reads `analysis`, `selected`, `hovered` and `view`; it writes only `selected` / `hovered`.
 */
export interface ViewState {
  showRibs: boolean
  showLungs: boolean
  showAorta: boolean
  showTerritories: boolean
  xray: boolean
  autoRotate: boolean
  cameraPreset: string | null // id from anatomy.yaml camera_presets; consumed then cleared by the scene
}

export interface AppState {
  config: AppConfig | null
  features: Features
  mode: ThresholdMode
  analysis: AnalyzeResponse | null
  explanation: ExplainResponse | null
  loading: { analyze: boolean; explain: boolean }
  error: string | null
  selected: TargetId | null // selected vessel (LAD/LCX/RCA) or null = CAD overview
  hovered: string | null // vessel id hovered in 3D, bullseye or cards
  view: ViewState

  setConfig: (c: AppConfig) => void
  setFeatures: (f: Features) => void
  setFeature: (id: string, v: Features[string]) => void
  setMode: (m: ThresholdMode) => void
  setAnalysis: (a: AnalyzeResponse | null) => void
  setExplanation: (e: ExplainResponse | null) => void
  setLoading: (k: 'analyze' | 'explain', v: boolean) => void
  setError: (e: string | null) => void
  select: (t: TargetId | null) => void
  hover: (v: string | null) => void
  setView: (patch: Partial<ViewState>) => void
}

export const useStore = create<AppState>((set) => ({
  config: null,
  features: {},
  mode: 'f1',
  analysis: null,
  explanation: null,
  loading: { analyze: false, explain: false },
  error: null,
  selected: null,
  hovered: null,
  view: {
    showRibs: true,
    showLungs: false,
    showAorta: true,
    showTerritories: true,
    xray: false,
    autoRotate: false,
    cameraPreset: null,
  },

  setConfig: (config) => set({ config }),
  setFeatures: (features) => set({ features }),
  setFeature: (id, v) => set((s) => ({ features: { ...s.features, [id]: v } })),
  setMode: (mode) => set({ mode }),
  setAnalysis: (analysis) => set({ analysis }),
  setExplanation: (explanation) => set({ explanation }),
  setLoading: (k, v) => set((s) => ({ loading: { ...s.loading, [k]: v } })),
  setError: (error) => set({ error }),
  select: (selected) => set({ selected }),
  hover: (hovered) => set({ hovered }),
  setView: (patch) => set((s) => ({ view: { ...s.view, ...patch } })),
}))

/** Probability shown for a vessel node, or null before the first analysis. */
export const vesselProbability = (s: AppState, vessel: string): number | null =>
  s.analysis?.vessels[vessel]?.p ?? null
