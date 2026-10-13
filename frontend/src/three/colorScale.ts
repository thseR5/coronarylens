/**
 * Probability -> colour. Viridis (van der Walt & Smith, matplotlib; perceptually uniform, colourblind-safe,
 * monotone in lightness), sampled at 11 stops and linearly interpolated in sRGB.
 * The darkest 15% of viridis is skipped (DOMAIN_START) so p = 0 vessels stay visible on the dark UI.
 */

export type RGB = [number, number, number]

/** Viridis at t = 0, 0.1, ..., 1 (same values as d3-scale-chromatic interpolateViridis). */
export const VIRIDIS_STOPS = [
  '#440154', '#482475', '#414487', '#355f8d', '#2a788e', '#21918c',
  '#22a884', '#44bf70', '#7ad151', '#bddf26', '#fde725',
] as const

export const DOMAIN_START = 0.15

/** Colour for structures before any analysis / without a target. */
export const NEUTRAL_VESSEL = '#c9d2db'
export const NOT_MODELLED = '#7c7c7c'

export function hexToRgb(hex: string): RGB {
  const n = parseInt(hex.slice(1), 16)
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
}

export function rgbToHex([r, g, b]: RGB): string {
  const c = (x: number) => Math.round(Math.min(1, Math.max(0, x)) * 255).toString(16).padStart(2, '0')
  return `#${c(r)}${c(g)}${c(b)}`
}

const STOPS_RGB = VIRIDIS_STOPS.map(hexToRgb)

/** Raw viridis at t in [0, 1]. */
export function viridis(t: number): RGB {
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (STOPS_RGB.length - 1)
  const i = Math.min(STOPS_RGB.length - 2, Math.floor(x))
  const f = x - i
  const a = STOPS_RGB[i]
  const b = STOPS_RGB[i + 1]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

/** Probability (0..1) -> sRGB colour on the trimmed viridis scale. */
export function riskColor(p: number): RGB {
  const q = Number.isFinite(p) ? Math.min(1, Math.max(0, p)) : 0
  return viridis(DOMAIN_START + (1 - DOMAIN_START) * q)
}

export const riskHex = (p: number) => rgbToHex(riskColor(p))

/** WCAG relative luminance of an sRGB colour. */
export function relativeLuminance([r, g, b]: RGB): number {
  const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
}

/** CSS gradient for legends (left = p 0, right = p 1). */
export function cssGradient(direction = 'to right', steps = 10): string {
  const parts = Array.from({ length: steps + 1 }, (_, i) => `${riskHex(i / steps)} ${(100 * i) / steps}%`)
  return `linear-gradient(${direction}, ${parts.join(', ')})`
}

/** Categorical colours for "which artery usually supplies this region" (territories and bullseye): magenta, orange and
 *  brown (ColorBrewer Dark2 hues), none of which occur in the viridis risk scale (purple-blue-green-yellow), so a
 *  territory colour is never read as risk. */
export const ARTERY_COLORS: Record<string, string> = { LAD: '#e7298a', LCX: '#d95f02', RCA: '#a6761d' }
