import { describe, expect, it } from 'vitest'
import { DOMAIN_START, VIRIDIS_STOPS, hexToRgb, relativeLuminance, riskColor, riskHex, viridis } from './colorScale'

describe('colour scale', () => {
  it('is strictly monotone in luminance over p in [0, 1]', () => {
    let prev = -1
    for (let i = 0; i <= 200; i++) {
      const l = relativeLuminance(riskColor(i / 200))
      expect(l).toBeGreaterThan(prev)
      prev = l
    }
  })

  it('has the right endpoints', () => {
    expect(riskHex(1)).toBe(VIRIDIS_STOPS[VIRIDIS_STOPS.length - 1])
    expect(viridis(0)).toEqual(hexToRgb(VIRIDIS_STOPS[0]))
    expect(riskColor(0)).toEqual(viridis(DOMAIN_START))
    expect(riskHex(-5)).toBe(riskHex(0))
    expect(riskHex(7)).toBe(riskHex(1))
    expect(riskHex(Number.NaN)).toBe(riskHex(0))
  })

  it('has usable contrast between ends (low end still visible, high end bright)', () => {
    const lo = relativeLuminance(riskColor(0))
    const hi = relativeLuminance(riskColor(1))
    expect((hi + 0.05) / (lo + 0.05)).toBeGreaterThan(5)
    expect(lo).toBeGreaterThan(0.04)
  })
})
