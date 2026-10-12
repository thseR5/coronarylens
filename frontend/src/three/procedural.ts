import * as THREE from 'three'
import type { StructureSpec } from '../types/config'

/**
 * Procedural stand-in for anatomy.glb (renderer: procedural, or automatic fallback when the glb fails).
 * Same frame as the glb: +X patient left, +Y superior, +Z anterior, heart centred at the origin, ~2 units across.
 * Emits one mesh per anatomy.yaml node it knows how to draw; vessel paths follow the textbook grooves
 * (LAD: anterior interventricular groove to the apex; LCX: left AV groove; RCA: right AV groove to the
 * posterior interventricular groove). Illustrative geometry only.
 */

const R = { x: 0.78, y: 1.0, z: 0.72 } // ellipsoid radii in the heart's local frame (local -Y = apex)
const APEX_DIR = new THREE.Vector3(0.55, -0.7, 0.45).normalize()
const Q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, -1, 0), APEX_DIR)
const deg = THREE.MathUtils.degToRad

/** Point on the (outset) heart ellipsoid. h in [-1 (apex), 1 (base)], phi: 0 = anterior, +90 = patient left. */
function surf(h: number, phiDeg: number, outset = 1): THREE.Vector3 {
  const r = Math.sqrt(Math.max(0, 1 - h * h))
  const phi = deg(phiDeg)
  return new THREE.Vector3(R.x * r * Math.sin(phi) * outset, R.y * h * outset, R.z * r * Math.cos(phi) * outset)
    .applyQuaternion(Q)
}

function patch(h0: number, h1: number, phi0: number, phi1: number, outset: number, nu = 24, nv = 24) {
  const pos: number[] = []
  const idx: number[] = []
  for (let j = 0; j <= nv; j++) {
    for (let i = 0; i <= nu; i++) {
      const p = surf(h0 + ((h1 - h0) * j) / nv, phi0 + ((phi1 - phi0) * i) / nu, outset)
      pos.push(p.x, p.y, p.z)
    }
  }
  for (let j = 0; j < nv; j++) {
    for (let i = 0; i < nu; i++) {
      const a = j * (nu + 1) + i
      const b = a + nu + 1
      idx.push(a, a + 1, b, a + 1, b + 1, b)
    }
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setIndex(idx)
  g.computeVertexNormals()
  return g
}

function tube(points: THREE.Vector3[], radius: number, segments = 120) {
  return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), segments, radius, 10, false)
}

const AV_H = 0.62 // height of the atrioventricular groove
const range = (a: number, b: number, n: number) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))

function builders(): Record<string, () => THREE.BufferGeometry> {
  const out = 1.04
  const bifurcation = surf(AV_H, 40, out)
  return {
    Myocardium: () => {
      const g = new THREE.SphereGeometry(1, 64, 48)
      g.scale(R.x, R.y, R.z)
      g.applyQuaternion(Q)
      return g
    },
    LAD: () => tube([bifurcation, ...range(0.55, -0.97, 10).map((h) => surf(h, 18 - 10 * (0.55 - h), out))], 0.04),
    LCX: () => tube(range(40, 200, 10).map((p) => surf(AV_H - 0.05 * Math.sin(deg(p - 40)), p, out)), 0.035),
    RCA: () =>
      tube(
        [
          ...range(-25, -200, 12).map((p) => surf(AV_H, p, out)),
          ...range(AV_H - 0.12, -0.6, 6).map((h) => surf(h, -195 + 10 * (AV_H - h), out)),
        ],
        0.04,
      ),
    LeftMain: () => tube([new THREE.Vector3(0.05, 0.85, 0.2), bifurcation], 0.045, 16),
    LV_Anterior: () => patch(-0.92, 0.5, -5, 60, 1.012),
    LV_Lateral: () => patch(-0.92, 0.5, 60, 140, 1.012),
    LV_Inferior: () => patch(-0.92, 0.5, 140, 215, 1.012),
    Aorta: () =>
      tube(
        [
          new THREE.Vector3(0.0, 0.7, 0.05), new THREE.Vector3(-0.05, 1.35, 0.15), new THREE.Vector3(0.15, 1.75, 0.0),
          new THREE.Vector3(0.4, 1.6, -0.55), new THREE.Vector3(0.35, 0.8, -0.8), new THREE.Vector3(0.3, -1.6, -0.8),
        ],
        0.13,
        80,
      ),
    Lungs: () => {
      const l = new THREE.SphereGeometry(1, 24, 16)
      l.scale(0.9, 1.6, 1.0)
      const r = l.clone()
      l.translate(1.55, 0.6, -0.2)
      r.translate(-1.65, 0.6, -0.2)
      return mergeSimple([l, r])
    },
    RibCage: () =>
      mergeSimple(
        range(1.9, -1.1, 8).map((y) => {
          const t = new THREE.TorusGeometry(2.0 + 0.15 * Math.cos(y), 0.05, 6, 40, deg(300))
          t.rotateX(Math.PI / 2 + 0.3)
          t.rotateY(deg(120))
          t.scale(1.15, 1, 0.85)
          t.translate(0, y, -0.25)
          return t
        }),
      ),
    Sternum: () => {
      const b = new THREE.BoxGeometry(0.25, 2.0, 0.1)
      b.translate(0, 0.7, 1.5)
      return b
    },
  }
}

function mergeSimple(geoms: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const pos: number[] = []
  const nor: number[] = []
  const idx: number[] = []
  let offset = 0
  for (const g0 of geoms) {
    const g = g0
    const p = g.getAttribute('position')
    if (!g.getAttribute('normal')) g.computeVertexNormals()
    const n = g.getAttribute('normal')
    for (let i = 0; i < p.count; i++) {
      pos.push(p.getX(i), p.getY(i), p.getZ(i))
      nor.push(n.getX(i), n.getY(i), n.getZ(i))
    }
    const index = g.index
    if (index) for (let i = 0; i < index.count; i++) idx.push(index.getX(i) + offset)
    else for (let i = 0; i < p.count; i++) idx.push(i + offset)
    offset += p.count
  }
  const out = new THREE.BufferGeometry()
  out.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  out.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3))
  out.setIndex(idx)
  return out
}

/** Build a group with one named mesh per anatomy.yaml structure the procedural model can draw. */
export function buildProceduralAnatomy(structures: StructureSpec[]): { root: THREE.Group; missing: string[] } {
  const root = new THREE.Group()
  root.name = 'ProceduralAnatomy'
  const b = builders()
  const missing: string[] = []
  for (const s of structures) {
    const make = b[s.node]
    if (!make) {
      missing.push(s.node)
      continue
    }
    const mesh = new THREE.Mesh(make(), new THREE.MeshStandardMaterial())
    mesh.name = s.node
    root.add(mesh)
  }
  return { root, missing }
}

export const PROCEDURAL_NODES = Object.keys(builders())
