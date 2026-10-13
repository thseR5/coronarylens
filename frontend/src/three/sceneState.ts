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
  /** Frontal clipping plane through the heart: cuts myocardium/aorta (never vessels) to reveal the LV territories. */
  clip: { enabled: boolean; depth: number } // depth in heart units along -Z (anterior -> posterior), -1..1
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

/** Probe WebGL once: availability + whether it is a software rasteriser (SwiftShader, llvmpipe, WARP). */
export function probeWebGL(): { ok: boolean; software: boolean; renderer: string } {
  try {
    const c = document.createElement('canvas')
    const g = (c.getContext('webgl2') || c.getContext('webgl')) as WebGLRenderingContext | null
    if (!g) return { ok: false, software: false, renderer: '' }
    const ext = g.getExtension('WEBGL_debug_renderer_info')
    const renderer = String(ext ? g.getParameter(ext.UNMASKED_RENDERER_WEBGL) : g.getParameter(g.RENDERER))
    g.getExtension('WEBGL_lose_context')?.loseContext()
    const forced = new URLSearchParams(window.location.search).get('lowpower')
    const software = forced != null ? forced !== '0' : /swiftshader|llvmpipe|software|basic render|warp/i.test(renderer)
    return { ok: true, software, renderer }
  } catch {
    return { ok: false, software: false, renderer: '' }
  }
}

/** Probed once at module load (browser only), so low-power mode is known before the first render. */
export const WEBGL =
  typeof document !== 'undefined' ? probeWebGL() : { ok: true, software: false, renderer: '' }

export const useSceneLocal = create<SceneLocalState>((set) => ({
  heartbeat: !prefersReducedMotion(),
  lowPower: WEBGL.software,
  labels: true,
  clip: { enabled: false, depth: 0 },
  activeRenderer: null,
  fallbackReason: null,
  screenshot: null,
  anchors: {},
  vesselMeshes: {},
  set: (patch) => set(patch),
}))
