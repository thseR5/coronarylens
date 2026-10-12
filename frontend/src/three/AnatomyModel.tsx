import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { useFrame, useThree, type ThreeEvent } from '@react-three/fiber'
import { useStore } from '../store'
import type { TargetId } from '../types/api'
import type { StructureSpec } from '../types/config'
import { NEUTRAL_VESSEL, NOT_MODELLED, riskColor } from './colorScale'
import { makeContextMaterial, makeVesselMaterial, makeXrayMaterial, stripeUniforms, type LitMaterial } from './materials'
import { prefersReducedMotion, useSceneLocal } from './sceneState'
import { ahaSegmentSpecs, type SegmentSpec } from './aha'

const MYOCARDIUM = '#9b7f7a'
const CONTEXT_STYLE: Record<string, { color: string; opacity: number }> = {
  Myocardium: { color: MYOCARDIUM, opacity: 1 },
  Aorta: { color: '#b0756c', opacity: 1 },
  Lungs: { color: '#8fb0cf', opacity: 0.22 },
  RibCage: { color: '#e2d8c6', opacity: 0.2 },
  Sternum: { color: '#e2d8c6', opacity: 0.25 },
}
const TRANSITION_MS = 400
const TERRITORY_MIX = 0.55
// Nodes that beat with the myocardium (scaled about the heart centre = origin). Aorta pulses subtly.
const BEAT_AMPLITUDE: Record<string, number> = { vessel: 0.035, territory: 0.035, segment: 0.035, Myocardium: 0.035, LeftMain: 0.035, Aorta: 0.01 }

export interface Hit {
  node: string
  label: string
  x: number
  y: number
}

const isTerritory = (role: string) => role === 'territory' || role === 'segment'

interface NodeRec {
  spec: StructureSpec & { segment?: number }
  mesh: THREE.Mesh
  basePos: THREE.Vector3
  baseScale: THREE.Vector3
  solid: THREE.Material
  xray?: THREE.ShaderMaterial
  color: { from: THREE.Color; to: THREE.Color; cur: THREE.Color; t0: number }
  anchor?: THREE.Vector3
  hot: number
}

/** Pulse waveform in [0,1]: sharp systolic contraction then slower relaxation, with a small atrial kick. */
function pulse(phase: number): number {
  if (phase < 0.12) return 0.25 * Math.sin((Math.PI * phase) / 0.12) // atrial kick
  if (phase < 0.22) return 0
  if (phase < 0.5) return Math.sin((Math.PI * (phase - 0.22)) / 0.56) // contraction to peak at ~0.5
  if (phase < 0.75) return Math.cos((Math.PI * (phase - 0.5)) / 0.5) // relaxation
  return 0
}

function findMesh(root: THREE.Object3D, name: string): THREE.Mesh | null {
  const obj = root.getObjectByName(name)
  if (!obj) return null
  if ((obj as THREE.Mesh).isMesh) return obj as THREE.Mesh
  let found: THREE.Mesh | null = null
  obj.traverse((o) => {
    if (!found && (o as THREE.Mesh).isMesh) found = o as THREE.Mesh
  })
  return found
}

/**
 * Label anchor: among vessel vertices that face outward (in the outer third by distance from the heart centre),
 * the one closest to the vessel's centroid, i.e. a point on the main course rather than a branch tip.
 */
function computeAnchor(mesh: THREE.Mesh): { anchor: THREE.Vector3 } {
  mesh.updateWorldMatrix(true, false)
  const pos = mesh.geometry.getAttribute('position')
  const v = new THREE.Vector3()
  const c = new THREE.Vector3()
  const pts: THREE.Vector3[] = []
  const step = Math.max(1, Math.floor(pos.count / 2000))
  for (let i = 0; i < pos.count; i += step) {
    v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld)
    pts.push(v.clone())
    c.add(v)
  }
  c.divideScalar(pts.length)
  const radii = pts.map((p) => p.length()).sort((a, b) => a - b)
  const rMin = radii[Math.floor(radii.length * 0.66)]
  let best = pts[0]
  let bestD = Infinity
  for (const p of pts) {
    if (p.length() < rMin) continue
    const d = p.distanceToSquared(c)
    if (d < bestD) {
      bestD = d
      best = p
    }
  }
  const out = best.lengthSq() > 1e-6 ? best.clone().normalize() : new THREE.Vector3(0, 0, 1)
  return { anchor: best.clone().addScaledVector(out, 0.14) }
}

export function AnatomyModel({ root, onHit }: { root: THREE.Object3D; onHit: (h: Hit | null) => void }) {
  const anatomy = useStore((s) => s.config!.anatomy)
  // config structures + the 17 AHA segment meshes (vessel mapping shared with the 2D bullseye)
  const structures = useMemo<(StructureSpec | SegmentSpec)[]>(() => [...anatomy.structures, ...ahaSegmentSpecs(anatomy)], [anatomy])
  const hoveredSegment = useStore((s) => s.hoveredSegment)
  const hoverSegment = useStore((s) => s.hoverSegment)
  const analysis = useStore((s) => s.analysis)
  const hovered = useStore((s) => s.hovered)
  const selected = useStore((s) => s.selected)
  const view = useStore((s) => s.view)
  const pr = useStore((s) => s.features.pr)
  const select = useStore((s) => s.select)
  const heartbeat = useSceneLocal((s) => s.heartbeat)
  const labels = useSceneLocal((s) => s.labels)
  const invalidate = useThree((s) => s.invalidate)
  const reduced = useMemo(prefersReducedMotion, [])
  const lite = useSceneLocal((s) => s.lowPower)

  // ---- one-time setup: find nodes by name, assign materials, cache transforms + anchors
  const nodes = useMemo(() => {
    const recs = new Map<string, NodeRec>()
    root.updateMatrixWorld(true)
    for (const spec of structures) {
      const mesh = findMesh(root, spec.node)
      if (!mesh) continue
      let solid: THREE.Material
      let xray: THREE.ShaderMaterial | undefined
      if (spec.role === 'vessel') {
        solid = makeVesselMaterial(lite)
        ;(solid as LitMaterial).side = THREE.DoubleSide
      } else if (isTerritory(spec.role)) {
        solid = makeContextMaterial(MYOCARDIUM, 1, lite)
        xray = makeXrayMaterial('#e8b4a8')
      } else if (spec.node === 'LeftMain') {
        solid = makeContextMaterial(NOT_MODELLED, 1, lite)
      } else {
        const st = CONTEXT_STYLE[spec.node] ?? { color: '#888888', opacity: 1 }
        solid = makeContextMaterial(st.color, st.opacity, lite)
        if (spec.node === 'Myocardium') xray = makeXrayMaterial('#e8b4a8')
      }
      mesh.material = solid
      mesh.renderOrder = (solid as THREE.Material).transparent ? 2 : 0
      const interactive = spec.role !== 'context' || spec.node === 'LeftMain' || spec.node === 'Myocardium'
      if (!interactive) mesh.raycast = () => {}
      const initial = new THREE.Color(spec.role === 'vessel' ? NEUTRAL_VESSEL : isTerritory(spec.role) ? MYOCARDIUM : '#000000')
      if (spec.role === 'vessel') (solid as LitMaterial).color.copy(initial)
      const a = spec.role === 'vessel' || spec.node === 'LeftMain' ? computeAnchor(mesh) : undefined
      recs.set(spec.node, {
        spec,
        mesh,
        basePos: mesh.position.clone(),
        baseScale: mesh.scale.clone(),
        solid,
        xray,
        color: { from: initial.clone(), to: initial.clone(), cur: initial.clone(), t0: 0 },
        hot: 0,
        anchor: a?.anchor,
      })
    }
    return recs
  }, [root, structures, lite])

  useEffect(() => {
    const anchors: Record<string, THREE.Vector3> = {}
    const vesselMeshes: Record<string, THREE.Mesh> = {}
    for (const r of nodes.values())
      if (r.anchor && r.spec.target) {
        anchors[r.spec.target] = r.anchor
        vesselMeshes[r.spec.target] = r.mesh
      }
    useSceneLocal.getState().set({ anchors, vesselMeshes })
  }, [nodes])

  useEffect(
    () => () => {
      for (const r of nodes.values()) {
        r.solid.dispose()
        r.xray?.dispose()
      }
    },
    [nodes],
  )

  // ---- target colours whenever analysis / territory toggle changes (animated in useFrame)
  useEffect(() => {
    const now = performance.now()
    for (const r of nodes.values()) {
      const { spec } = r
      let to: THREE.Color | null = null
      const p = spec.target ? analysis?.vessels[spec.target]?.p : undefined
      if (spec.role === 'vessel') to = p == null ? new THREE.Color(NEUTRAL_VESSEL) : new THREE.Color().setRGB(...riskColor(p), THREE.SRGBColorSpace)
      if (isTerritory(spec.role)) {
        to = new THREE.Color(MYOCARDIUM)
        if (p != null && view.showTerritories)
          to.lerp(new THREE.Color().setRGB(...riskColor(p), THREE.SRGBColorSpace), TERRITORY_MIX)
        // alternate lightness by segment so borders between same-vessel segments stay visible (hue unchanged)
        if (spec.role === 'segment' && (spec as { segment?: number }).segment! % 2 === 0) to.multiplyScalar(0.86)
      }
      if (!to || to.equals(r.color.to)) continue
      r.color.from.copy(r.color.cur)
      r.color.to.copy(to)
      r.color.t0 = now
    }
    invalidate()
  }, [nodes, analysis, view.showTerritories, invalidate])

  // ---- visibility / x-ray / highlight
  const hasSegments = useMemo(() => [...nodes.values()].some((r) => r.spec.role === 'segment'), [nodes])
  useEffect(() => {
    for (const r of nodes.values()) {
      const n = r.spec.node
      if (n === 'RibCage' || n === 'Sternum') r.mesh.visible = view.showRibs
      if (n === 'Lungs') r.mesh.visible = view.showLungs
      if (n === 'Aorta') r.mesh.visible = view.showAorta
      // AHA segments (default) and nearest-artery (Voronoi) territories are alternative views of the same LV surface;
      // the procedural fallback has no segment meshes, so it always shows its territories
      if (r.spec.role === 'segment') r.mesh.visible = view.territoryMode === 'aha'
      if (r.spec.role === 'territory') r.mesh.visible = view.territoryMode === 'voronoi' || !hasSegments
      // Always (re)bind: under React StrictMode the memo above runs twice and the mesh may hold the
      // discarded run's material, so colour/stripe updates on `r.solid` would never be drawn.
      r.mesh.material = view.xray && r.xray ? r.xray : r.solid
      r.mesh.renderOrder = view.xray && r.xray ? 3 : (r.solid.transparent ? 2 : 0)
      const target = r.spec.target
      // 0 = normal, 1 = hovered, 2 = selected; applied in useFrame as an emissive boost in the vessel's own hue
      r.hot = target == null ? 0 : target === selected ? 2 : target === hovered ? 1 : 0
      if (r.spec.segment != null && r.spec.segment === hoveredSegment) r.hot = 3 // the exact hovered segment
    }
    invalidate()
  }, [nodes, view.showRibs, view.showLungs, view.showAorta, view.xray, view.territoryMode, hasSegments, hovered, hoveredSegment, selected, invalidate])

  // ---- uncertain-status stripes (target values; faded in useFrame)
  const stripeTarget = useRef(new Map<string, number>())
  useEffect(() => {
    for (const r of nodes.values()) {
      if (r.spec.role !== 'vessel' || !r.spec.target) continue
      stripeTarget.current.set(r.spec.node, analysis?.vessels[r.spec.target]?.status === 'uncertain' ? 1 : 0)
    }
    invalidate()
  }, [nodes, analysis, invalidate])

  const bpm = typeof pr === 'number' && pr >= 30 && pr <= 220 ? pr : 70

  useFrame((state, delta) => {
    const now = performance.now()
    let animating = false
    if (!reduced) stripeUniforms.uTime.value += Math.min(delta, 0.1)
    const beat = heartbeat && !reduced ? pulse((state.clock.elapsedTime * bpm) / 60 % 1) : 0
    for (const r of nodes.values()) {
      // colour transition
      const k = Math.min(1, (now - r.color.t0) / TRANSITION_MS)
      const e = k < 1 ? 1 - (1 - k) ** 3 : 1
      if (k < 1) animating = true
      r.color.cur.copy(r.color.from).lerp(r.color.to, e)
      if (r.spec.role === 'vessel' || isTerritory(r.spec.role)) {
        const m = r.solid as LitMaterial
        m.color.copy(r.color.cur)
        const boost = r.spec.role === 'vessel' ? [0, 0.35, 0.6, 0.6][r.hot] : [0, 0.12, 0.18, 0.5][r.hot]
        m.emissive.copy(r.color.cur).multiplyScalar(boost)
      }
      // stripes fade
      const u = r.solid.userData.stripes as { value: number } | undefined
      if (u) {
        const t = stripeTarget.current.get(r.spec.node) ?? 0
        if (Math.abs(u.value - t) > 1e-3) {
          u.value += Math.sign(t - u.value) * Math.min(Math.abs(t - u.value), delta / (TRANSITION_MS / 1000))
          animating = true
        }
      }
      // heartbeat: contraction scales the heart about the origin
      const amp = BEAT_AMPLITUDE[r.spec.role] ?? BEAT_AMPLITUDE[r.spec.node] ?? 0
      if (amp) {
        const s = r.spec.node === 'Aorta' ? 1 + amp * beat : 1 - amp * beat
        r.mesh.position.copy(r.basePos).multiplyScalar(s)
        r.mesh.scale.copy(r.baseScale).multiplyScalar(s)
      }
    }
    if (animating) invalidate()
  })

  // ---- pointer handling (only interactive meshes are raycast)
  const vesselOf = (node: string): string | null => {
    const r = nodes.get(node)
    return r?.spec.target ?? null
  }
  const onMove = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    const node = e.object.name
    const r = nodes.get(node)
    if (!r) return
    const v = vesselOf(node)
    const seg = r.spec.segment ?? null
    const st = useStore.getState()
    if (v !== st.hovered || seg !== st.hoveredSegment) hoverSegment(seg, v)
    onHit({ node, label: r.spec.label, x: e.nativeEvent.clientX, y: e.nativeEvent.clientY })
    document.body.style.cursor = v ? 'pointer' : 'default'
  }
  const onOut = () => {
    hoverSegment(null, null)
    onHit(null)
    document.body.style.cursor = 'default'
  }
  const onClick = (e: ThreeEvent<MouseEvent>) => {
    if (e.delta > 5) return // ignore orbit drags
    e.stopPropagation()
    const v = vesselOf(e.object.name)
    if (v) select(v as TargetId)
  }

  const labelled = [...nodes.values()].filter((r) => r.anchor)
  return (
    <group>
      <primitive object={root} onPointerMove={onMove} onPointerOut={onOut} onClick={onClick} />
      {labels &&
        labelled.map((r) => {
          const t = r.spec.target
          const risk = t ? analysis?.vessels[t] : undefined
          const hot = t != null && (t === hovered || t === selected)
          return (
            <Html key={r.spec.node} position={r.anchor!} center zIndexRange={[30, 0]} style={{ pointerEvents: 'none' }}>
              <div
                data-testid={`label-${r.spec.node}`}
                className={`whitespace-nowrap rounded border px-1.5 py-0.5 text-[11px] leading-tight shadow ${
                  hot ? 'border-accent bg-ink/95' : 'border-line bg-ink/80'
                } ${t ? 'text-text' : 'text-muted'}`}
              >
                <div className="font-semibold">{t ?? r.spec.label}</div>
                {!t && <div>not modelled</div>}
                {t && risk && (
                  <div className="font-mono">
                    {Math.round(risk.p * 100)}% <span className="text-muted">[{Math.round(risk.interval[0] * 100)}–{Math.round(risk.interval[1] * 100)}]</span>
                    {risk.status === 'uncertain' && <span className="ml-1 text-warn">uncertain</span>}
                  </div>
                )}
                {t && !risk && <div className="text-muted">no analysis</div>}
              </div>
            </Html>
          )
        })}
    </group>
  )
}
