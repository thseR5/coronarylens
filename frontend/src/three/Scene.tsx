import { Component, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import * as THREE from 'three'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Html, useGLTF } from '@react-three/drei'
import { useStore } from '../store'
import { mockAnalysis } from '../api/mock'
import { AnatomyModel, type Hit } from './AnatomyModel'
import { CameraRig, DEFAULT_DISTANCE } from './CameraRig'
import Legend from './Legend'
import SceneControls from './SceneControls'
import { buildProceduralAnatomy } from './procedural'
import { prefersReducedMotion, useSceneLocal } from './sceneState'

declare global {
  interface Window {
    __coronaryLens?: {
      frames: number
      renderer: string | null
      /** CSS-pixel position of a visible point on a vessel (by id) or any node (by name), for e2e pointer tests. */
      project: (vesselOrNode: string) => { x: number; y: number } | null
      /** Current uniform scale factor of a vessel mesh (heartbeat check). */
      meshScale: (vessel: string) => number
    }
  }
}

/** Probe WebGL once: availability + whether it is a software rasteriser (SwiftShader, llvmpipe, WARP). */
function probeWebGL(): { ok: boolean; software: boolean; renderer: string } {
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

/** Real anatomy from anatomy.glb (meshopt-compressed; decoder bundled, no CDN). */
function GlbAnatomy({ onHit }: { onHit: (h: Hit | null) => void }) {
  const anatomy = useStore((s) => s.config!.anatomy)
  const gltf = useGLTF(anatomy.glb_path, false, true)
  const missingVessels = anatomy.structures.filter((s) => s.role === 'vessel' && !gltf.scene.getObjectByName(s.node))
  if (missingVessels.length) throw new Error(`anatomy.glb lacks vessel nodes: ${missingVessels.map((s) => s.node).join(', ')}`)
  useEffect(() => useSceneLocal.getState().set({ activeRenderer: 'bodyparts3d' }), [])
  return <AnatomyModel root={gltf.scene} onHit={onHit} />
}

function ProceduralAnatomy({ onHit }: { onHit: (h: Hit | null) => void }) {
  const structures = useStore((s) => s.config!.anatomy.structures)
  const root = useMemo(() => buildProceduralAnatomy(structures).root, [structures])
  useEffect(() => useSceneLocal.getState().set({ activeRenderer: 'procedural' }), [])
  return <AnatomyModel root={root} onHit={onHit} />
}

class GlbBoundary extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  componentDidCatch(err: unknown) {
    const msg = err instanceof Error ? err.message : String(err)
    console.warn('[CoronaryLens] anatomy.glb failed, using procedural heart:', msg)
    useSceneLocal.getState().set({ fallbackReason: msg })
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}

/** Keeps frames coming only while something animates (heartbeat, uncertain stripes); idle = no GPU work. */
function FrameDriver() {
  const heartbeat = useSceneLocal((s) => s.heartbeat)
  const anyUncertain = useStore((s) => Object.values(s.analysis?.vessels ?? {}).some((v) => v.status === 'uncertain'))
  const invalidate = useThree((s) => s.invalidate)
  const continuous = !prefersReducedMotion() && (heartbeat || anyUncertain)
  useEffect(() => {
    invalidate()
  }, [continuous, invalidate])
  useFrame((state) => {
    if (window.__coronaryLens) window.__coronaryLens.frames++
    if (continuous && !document.hidden) state.invalidate()
  })
  return null
}

/** Registers a PNG export that renders the current frame synchronously (no preserveDrawingBuffer cost). */
function ScreenshotHook() {
  const { gl, scene, camera } = useThree()
  useEffect(() => {
    window.__coronaryLens = {
      frames: 0,
      renderer: null,
      meshScale: (vessel) => useSceneLocal.getState().vesselMeshes[vessel]?.scale.x ?? NaN,
      project: (vessel) => {
        // first vessel vertex whose camera ray hits this vessel before anything else
        let mesh: THREE.Mesh | undefined = useSceneLocal.getState().vesselMeshes[vessel]
        if (!mesh) {
          scene.getObjectByName(vessel)?.traverse((o) => {
            if (!mesh && (o as THREE.Mesh).isMesh) mesh = o as THREE.Mesh
          })
        }
        if (!mesh) return null
        const pos = mesh.geometry.getAttribute('position')
        const ray = new THREE.Raycaster()
        const v = new THREE.Vector3()
        const r = gl.domElement.getBoundingClientRect()
        for (let i = 0; i < pos.count; i += 7) {
          v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld).project(camera)
          if (Math.abs(v.x) > 0.9 || Math.abs(v.y) > 0.9) continue
          ray.setFromCamera(new THREE.Vector2(v.x, v.y), camera)
          const hit = ray.intersectObjects(scene.children, true)[0]
          if (hit?.object === mesh) return { x: r.left + ((v.x + 1) / 2) * r.width, y: r.top + ((1 - v.y) / 2) * r.height }
        }
        return null
      },
    }
    try {
      const ctx = gl.getContext()
      const ext = ctx.getExtension('WEBGL_debug_renderer_info')
      window.__coronaryLens.renderer = ext ? String(ctx.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : null
    } catch {
      /* ignore */
    }
    useSceneLocal.getState().set({
      screenshot: () => {
        gl.render(scene, camera)
        return gl.domElement.toDataURL('image/png')
      },
    })
    return () => useSceneLocal.getState().set({ screenshot: null })
  }, [gl, scene, camera])
  return null
}

function DemoToggle() {
  const analysis = useStore((s) => s.analysis)
  const setAnalysis = useStore((s) => s.setAnalysis)
  const [waited, setWaited] = useState(false)
  useEffect(() => {
    const t = setTimeout(() => setWaited(true), 1000)
    return () => clearTimeout(t)
  }, [])
  const isDemo = analysis?.model_version === 'mock'
  if (!(isDemo || (waited && !analysis))) return null
  return (
    <button
      type="button"
      data-testid="demo-toggle"
      onClick={() => setAnalysis(isDemo ? null : mockAnalysis)}
      className="rounded border border-warn/60 bg-warn/15 px-2 py-0.5 text-[11px] text-warn focus:outline-none focus-visible:ring-2 focus-visible:ring-warn"
    >
      {isDemo ? 'DEMO DATA shown (mock, not a patient) · clear' : 'No analysis yet · load DEMO data (mock)'}
    </button>
  )
}

function Tooltip({ hit }: { hit: Hit | null }) {
  const structures = useStore((s) => s.config!.anatomy.structures)
  const analysis = useStore((s) => s.analysis)
  if (!hit) return null
  const spec = structures.find((s) => s.node === hit.node)
  const risk = spec?.target ? analysis?.vessels[spec.target] : undefined
  return (
    <div
      className="pointer-events-none fixed z-50 max-w-64 rounded border border-line bg-ink/95 px-2 py-1 text-[11px] text-text shadow"
      style={{ left: hit.x + 14, top: hit.y + 14 }}
      role="tooltip"
    >
      <div className="font-semibold">{hit.label}</div>
      {spec?.role === 'territory' && <div className="text-muted">supplied by {spec.target} (illustrative)</div>}
      {risk && (
        <div className="font-mono">
          P = {(risk.p * 100).toFixed(0)}% [{(risk.interval[0] * 100).toFixed(0)}–{(risk.interval[1] * 100).toFixed(0)}] · {risk.status}
        </div>
      )}
      {spec?.note && <div className="text-muted">{spec.note}</div>}
      {spec?.role === 'vessel' && <div className="text-muted">click to select</div>}
    </div>
  )
}

// Software rasterisers (no GPU): render at reduced resolution without MSAA so interaction stays fluid.
const LOW_POWER_DPR = 0.5

export default function CoronaryScene() {
  const anatomy = useStore((s) => s.config?.anatomy)
  const showTerritories = useStore((s) => s.view.showTerritories)
  const select = useStore((s) => s.select)
  const activeRenderer = useSceneLocal((s) => s.activeRenderer)
  const fallbackReason = useSceneLocal((s) => s.fallbackReason)
  const [hit, setHit] = useState<Hit | null>(null)
  const [probe] = useState(() => {
    const p = probeWebGL()
    useSceneLocal.getState().set({ lowPower: p.software })
    if (p.software) useStore.getState().setView({ showRibs: false })
    return p
  })
  const lowPower = probe.software
  const down = useRef<{ x: number; y: number } | null>(null)

  if (!anatomy) return null
  if (!probe.ok)
    return (
      <div className="flex h-full items-center justify-center p-6 text-center text-sm text-muted" data-testid="no-webgl">
        WebGL is not available in this browser, so the 3D heart cannot be shown. All risk numbers remain available in the
        dashboard.
      </div>
    )

  const procedural = <ProceduralAnatomy onHit={setHit} />
  // ?renderer=procedural forces the fallback (demo/testing); otherwise anatomy.yaml decides.
  const renderer = new URLSearchParams(window.location.search).get('renderer') ?? anatomy.renderer
  return (
    <div
      className="absolute inset-0"
      data-testid="scene"
      onPointerDown={(e) => (down.current = { x: e.clientX, y: e.clientY })}
    >
      <Canvas
        frameloop="demand"
        dpr={lowPower ? LOW_POWER_DPR : [1, 1.5]}
        flat
        shadows={false}
        gl={{ antialias: !lowPower, powerPreference: 'high-performance' }}
        camera={{ position: [0, 0.4, DEFAULT_DISTANCE], fov: 38, near: 0.05, far: 100 }}
        onCreated={({ gl: r }) => {
          r.setClearColor(new THREE.Color('#0b1118'), 1)
          r.localClippingEnabled = true // per-material clipping plane (see AnatomyModel)
        }}
        onPointerMissed={(e) => {
          const d = down.current
          if (d && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 5) select(null)
        }}
        aria-label="3D coronary anatomy"
      >
        <hemisphereLight args={['#ffffff', '#3a4250', 1.1]} />
        <directionalLight position={[3, 5, 6]} intensity={1.7} />
        <directionalLight position={[-4, -2, -5]} intensity={0.6} />
        <ScreenshotHook />
        <FrameDriver />
        <CameraRig />
        <Suspense
          fallback={
            <Html center>
              <div className="text-sm text-muted">Loading anatomy…</div>
            </Html>
          }
        >
          {renderer === 'procedural' ? (
            procedural
          ) : (
            <GlbBoundary fallback={procedural}>
              <GlbAnatomy onHit={setHit} />
            </GlbBoundary>
          )}
        </Suspense>
      </Canvas>

      <div className="pointer-events-none absolute inset-x-2 top-2 flex flex-col items-start gap-1">
        <SceneControls />
        <div className="pointer-events-auto">
          <DemoToggle />
        </div>
      </div>
      <div className="pointer-events-none absolute bottom-2 left-2 flex flex-col items-start gap-1">
        {showTerritories && <div className="max-w-72 text-[10px] text-muted">{anatomy.territory_note}</div>}
        <div className="text-[10px] text-muted" data-testid="renderer-info">
          {activeRenderer === 'procedural'
            ? `Procedural heart (illustrative)${fallbackReason ? ' · anatomy.glb unavailable' : ''}`
            : activeRenderer === 'bodyparts3d'
              ? 'Anatomy: BodyParts3D © DBCLS, CC BY 4.0'
              : ''}
          {lowPower && ' · software rendering: reduced quality'}
        </div>
      </div>
      <Legend className="pointer-events-auto absolute right-2 bottom-2" />
      <Tooltip hit={hit} />
    </div>
  )
}
