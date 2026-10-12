import { create } from 'zustand'
import type { Mesh, Vector3 } from 'three'

/**
 * 3D-only UI state that the frozen app store (src/store) does not carry: heartbeat toggle, label toggle,
 * which renderer is actually active (after an automatic fallback) and a screenshot hook.
 */
export interface SceneLocalState {
  heartbeat: boolean
  /** Software rasteriser detected: lower DPR, no MSAA, Lambert shading, ribs hidden initially. */
  lowPower: boolean
  labels: boolean
  activeRenderer: 'bodyparts3d' | 'procedural' | null
  fallbackReason: string | null
  screenshot: (() => string | null) | null
  /** World-space label anchors keyed by vessel target id (LAD/LCX/RCA). */
  anchors: Record<string, Vector3>
  /** Vessel meshes keyed by target id (used by the e2e click helper). */
  vesselMeshes: Record<string, Mesh>
  set: (patch: Partial<Omit<SceneLocalState, 'set'>>) => void
}

export const prefersReducedMotion = (): boolean =>
  typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

export const useSceneLocal = create<SceneLocalState>((set) => ({
  heartbeat: !prefersReducedMotion(),
  lowPower: false,
  labels: true,
  activeRenderer: null,
  fallbackReason: null,
  screenshot: null,
  anchors: {},
  vesselMeshes: {},
  set: (patch) => set(patch),
}))
