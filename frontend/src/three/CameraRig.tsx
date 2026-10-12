import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { useStore } from '../store'
import type { CameraPreset } from '../types/config'
import { prefersReducedMotion, useSceneLocal } from './sceneState'

export const DEFAULT_DISTANCE = 5.2
const FLY_MS = 900

/**
 * Viewing direction (unit vector from target to camera) for an angiographic projection, in the anatomy frame
 * (+X patient left, +Y superior, +Z anterior). LAO (+) swings the camera toward the patient's left about the
 * body long axis; cranial (+) tilts it toward the head.
 */
export function presetDirection(p: Pick<CameraPreset, 'rao_lao' | 'cran_caud'>): THREE.Vector3 {
  const a = THREE.MathUtils.degToRad(p.rao_lao)
  const c = THREE.MathUtils.degToRad(p.cran_caud)
  return new THREE.Vector3(Math.cos(c) * Math.sin(a), Math.sin(c), Math.cos(c) * Math.cos(a))
}

interface Flight {
  fromPos: THREE.Vector3
  toPos: THREE.Vector3
  fromTarget: THREE.Vector3
  toTarget: THREE.Vector3
  t0: number
  dur: number
}

/** OrbitControls + preset/selection fly-to. Vessel anchors come from the anatomy model (world space). */
export function CameraRig() {
  const anchors = useSceneLocal((s) => s.anchors)
  const controls = useRef<OrbitControlsImpl>(null)
  const camera = useThree((s) => s.camera)
  const invalidate = useThree((s) => s.invalidate)
  const flight = useRef<Flight | null>(null)
  const presets = useStore((s) => s.config!.anatomy.camera_presets)
  const cameraPreset = useStore((s) => s.view.cameraPreset)
  const autoRotate = useStore((s) => s.view.autoRotate)
  const setView = useStore((s) => s.setView)
  const selected = useStore((s) => s.selected)

  const flyTo = (toPos: THREE.Vector3, toTarget: THREE.Vector3) => {
    const c = controls.current
    if (!c) return
    flight.current = {
      fromPos: camera.position.clone(),
      toPos,
      fromTarget: c.target.clone(),
      toTarget,
      t0: performance.now(),
      dur: prefersReducedMotion() ? 0 : FLY_MS,
    }
    invalidate()
  }

  // camera presets: consume then clear
  useEffect(() => {
    if (!cameraPreset) return
    const p = presets.find((x) => x.id === cameraPreset)
    if (p) {
      const target = new THREE.Vector3(0, 0, 0)
      flyTo(presetDirection(p).multiplyScalar(DEFAULT_DISTANCE), target)
    }
    setView({ cameraPreset: null })
  }, [cameraPreset])

  // selection: fly toward the selected vessel; deselect: recentre on the heart keeping the view direction
  useEffect(() => {
    const c = controls.current
    if (!c) return
    const anchor = selected ? anchors[selected] : undefined
    if (anchor) {
      const viewDir = camera.position.clone().sub(c.target).normalize()
      const outward = anchor.clone().normalize()
      const dir = outward.multiplyScalar(0.65).add(viewDir.multiplyScalar(0.35)).normalize()
      const target = anchor.clone().multiplyScalar(0.6)
      flyTo(target.clone().add(dir.multiplyScalar(3.6)), target)
    } else if (!selected && c.target.lengthSq() > 1e-6) {
      const viewDir = camera.position.clone().sub(c.target).normalize()
      flyTo(viewDir.multiplyScalar(DEFAULT_DISTANCE), new THREE.Vector3())
    }
  }, [selected, anchors])

  useFrame(() => {
    const f = flight.current
    const c = controls.current
    if (!f || !c) return
    const k = f.dur <= 0 ? 1 : Math.min(1, (performance.now() - f.t0) / f.dur)
    const e = k < 0.5 ? 4 * k * k * k : 1 - (-2 * k + 2) ** 3 / 2
    camera.position.lerpVectors(f.fromPos, f.toPos, e)
    c.target.lerpVectors(f.fromTarget, f.toTarget, e)
    c.update()
    if (k >= 1) flight.current = null
    invalidate()
  })

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enablePan={false}
      minDistance={2.2}
      maxDistance={12}
      minPolarAngle={0.15}
      maxPolarAngle={Math.PI - 0.15}
      autoRotate={autoRotate}
      autoRotateSpeed={1.2}
      onStart={() => (flight.current = null)}
    />
  )
}
