// Shapes of config/*.yaml as served by GET /api/config (or loaded raw as the offline fallback).

export interface FeatureGroup {
  id: string
  label: string
}

export interface FeatureSpec {
  id: string
  column: string
  label: string
  unit: string
  type: 'numeric' | 'binary' | 'ordinal' | 'categorical'
  encoding?: string
  levels?: (string | number)[]
  /** Display names for a binary feature's values, e.g. sex: { 0: 'Female', 1: 'Male' } (default No / Yes). */
  value_labels?: Record<string, string>
  group: string
  range: [number, number] | null
  ref: [number, number] | null
  default: number | string
  modifiable: boolean
  monotone: -1 | 0 | 1
  quick: boolean
  description: string
}

export interface FeaturesConfig {
  groups: FeatureGroup[]
  features: FeatureSpec[]
  engineered: { id: string; label: string; unit: string; group: string; inputs: string[] }[]
}

export interface VesselSpec {
  id: string
  column: string
  label: string
}

export interface TargetsConfig {
  forbidden_columns: string[]
  cad: { id: string; column: string; label: string }
  vessels: VesselSpec[]
  derived: { id: string; label: string }[]
  threshold_modes: { id: string; label: string }[]
}

export type StructureRole = 'vessel' | 'territory' | 'segment' | 'context'

export interface StructureSpec {
  node: string
  role: StructureRole
  target: string | null
  label: string
  fma: string[]
  note?: string
}

export interface CameraPreset {
  id: string
  label: string
  rao_lao: number
  cran_caud: number
}

export interface AnatomyConfig {
  source: { name: string; license: string; url: string; attribution: string }
  renderer: 'bodyparts3d' | 'procedural'
  glb_path: string
  structures: StructureSpec[]
  territory_note: string
  aha_segments: { node_prefix: string; source: string }
  voronoi_label: string
  bullseye: Record<string, number[]>
  camera_presets: CameraPreset[]
}

export interface AppConfig {
  features: FeaturesConfig
  targets: TargetsConfig
  anatomy: AnatomyConfig
}
