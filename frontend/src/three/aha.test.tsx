import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { bundledConfig } from '../api/client'
import { useStore } from '../store'
import manifestRaw from '../../public/models/anatomy_manifest.json?raw'
import { ahaSegmentSpecs } from './aha'
import Bullseye, { segmentVesselMap } from './Bullseye'

const cfg = bundledConfig()
const manifest = JSON.parse(manifestRaw) as { nodes: Record<string, { role: string; segment?: number; target: string; triangles: number }> }

describe('3D AHA segments <-> 2D bullseye', () => {
  const specs = ahaSegmentSpecs(cfg.anatomy)
  const bullseye = segmentVesselMap(cfg.anatomy.bullseye)

  it('the 3D segment -> vessel mapping equals the bullseye mapping for all 17 segments', () => {
    expect(specs.map((s) => s.segment)).toEqual(Array.from({ length: 17 }, (_, i) => i + 1))
    for (const s of specs) expect(s.target, `segment ${s.segment}`).toBe(bullseye.get(s.segment))
  })

  it('uses the standard AHA assignment', () => {
    const by = (v: string) => specs.filter((s) => s.target === v).map((s) => s.segment)
    expect(by('LAD')).toEqual([1, 2, 7, 8, 13, 14, 17])
    expect(by('RCA')).toEqual([3, 4, 9, 10, 15])
    expect(by('LCX')).toEqual([5, 6, 11, 12, 16])
  })

  it('every segment exists in the built glb manifest with the same segment number and vessel', () => {
    for (const s of specs) {
      const n = manifest.nodes[s.node]
      expect(n, s.node).toBeDefined()
      expect(n.role).toBe('segment')
      expect(n.segment).toBe(s.segment)
      expect(n.target).toBe(s.target)
      expect(n.triangles).toBeGreaterThan(0)
    }
  })
})

describe('segment hover sync', () => {
  beforeEach(() => useStore.setState({ config: cfg, hovered: null, hoveredSegment: null, selected: null }))
  afterEach(cleanup)

  it('hovering a bullseye segment sets the shared segment + vessel (which the 3D view highlights)', () => {
    const { container } = render(<Bullseye />)
    fireEvent.mouseEnter(container.querySelector('[data-segment="11"]')!)
    expect(useStore.getState().hoveredSegment).toBe(11)
    expect(useStore.getState().hovered).toBe('LCX')
    fireEvent.mouseLeave(container.querySelector('[data-segment="11"]')!)
    expect(useStore.getState().hoveredSegment).toBeNull()
  })

  it('a segment hovered in 3D is highlighted in the bullseye', () => {
    useStore.getState().hoverSegment(4, 'RCA') // what AnatomyModel calls on pointer move over AHA_04
    const { container } = render(<Bullseye />)
    expect(container.querySelector('[data-segment="4"]')!.getAttribute('data-hot')).toBe('segment')
    expect(container.querySelector('[data-segment="3"]')!.getAttribute('data-hot')).toBe('vessel') // same vessel
    expect(container.querySelector('[data-segment="5"]')!.getAttribute('data-hot')).toBeNull()
  })
})
