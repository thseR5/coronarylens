import type { AnatomyConfig, StructureSpec } from '../types/config'
import { bullseyeSegments, segmentVesselMap } from './Bullseye'

export interface SegmentSpec extends StructureSpec {
  role: 'segment'
  segment: number
}

/**
 * The 17 AHA segment meshes (AHA_01..AHA_17, built by tools/build_anatomy.py), each assigned to its vessel by the
 * SAME mapping the 2D bullseye uses (anatomy.yaml `bullseye`), so the 3D view and the bullseye cannot disagree.
 */
export function ahaSegmentSpecs(anatomy: AnatomyConfig): SegmentSpec[] {
  const vessel = segmentVesselMap(anatomy.bullseye)
  return bullseyeSegments().map((s) => ({
    node: `${anatomy.aha_segments.node_prefix}${String(s.id).padStart(2, '0')}`,
    role: 'segment',
    target: vessel.get(s.id) ?? null,
    label: `AHA segment ${s.id} · ${s.name}`,
    fma: [],
    segment: s.id,
  }))
}
