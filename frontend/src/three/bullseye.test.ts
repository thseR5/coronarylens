import { describe, expect, it } from 'vitest'
import { bundledConfig } from '../api/client'
import { bullseyeSegments, segmentVesselMap } from './Bullseye'

describe('AHA 17-segment bullseye', () => {
  it('config assigns segments 1..17 exactly once', () => {
    const all = Object.values(bundledConfig().anatomy.bullseye).flat().sort((a, b) => a - b)
    expect(all).toEqual(Array.from({ length: 17 }, (_, i) => i + 1))
    expect(segmentVesselMap(bundledConfig().anatomy.bullseye).size).toBe(17)
  })

  it('geometry draws segments 1..17 exactly once and tiles each ring', () => {
    const segs = bullseyeSegments()
    expect(segs.map((s) => s.id).sort((a, b) => a - b)).toEqual(Array.from({ length: 17 }, (_, i) => i + 1))
    for (const ring of ['basal', 'mid', 'apical'] as const) {
      const r = segs.filter((s) => s.ring === ring)
      expect(r.reduce((a, s) => a + (s.a1 - s.a0), 0)).toBe(360)
    }
    // segment 1 (basal anterior) is centred at the top, segment 4 (basal inferior) at the bottom
    const mid = (id: number) => { const s = segs.find((x) => x.id === id)!; return (s.a0 + s.a1) / 2 }
    expect(mid(1)).toBe(90)
    expect(mid(4)).toBe(270)
  })

  it('rejects duplicate assignments', () => {
    expect(() => segmentVesselMap({ LAD: [1], RCA: [1] })).toThrow()
  })
})
