import type { KeyboardEvent } from 'react'
import { useStore } from '../store'
import type { TargetId } from '../types/api'
import { ARTERY_COLORS, NEUTRAL_VESSEL } from './colorScale'

/** AHA 17-segment model (Cerqueira et al., Circulation 2002). Anterior at top, septum left, lateral right. */
export interface Segment {
  id: number
  ring: 'basal' | 'mid' | 'apical' | 'apex'
  name: string
  a0: number // math degrees, CCW from +x (screen right)
  a1: number
  r0: number
  r1: number
}

const WALLS6 = ['anterior', 'anteroseptal', 'inferoseptal', 'inferior', 'inferolateral', 'anterolateral']
const WALLS4 = ['anterior', 'septal', 'inferior', 'lateral']
const RADII = { apex: [0, 22], apical: [22, 46], mid: [46, 70], basal: [70, 94] } as const

export function bullseyeSegments(): Segment[] {
  const segs: Segment[] = []
  ;(['basal', 'mid'] as const).forEach((ring, k) => {
    WALLS6.forEach((w, i) =>
      segs.push({ id: k * 6 + i + 1, ring, name: `${ring} ${w}`, a0: 60 + 60 * i, a1: 120 + 60 * i, r0: RADII[ring][0], r1: RADII[ring][1] }),
    )
  })
  WALLS4.forEach((w, i) =>
    segs.push({ id: 13 + i, ring: 'apical', name: `apical ${w}`, a0: 45 + 90 * i, a1: 135 + 90 * i, r0: RADII.apical[0], r1: RADII.apical[1] }),
  )
  segs.push({ id: 17, ring: 'apex', name: 'apex', a0: 0, a1: 360, r0: 0, r1: RADII.apex[1] })
  return segs
}

/** segment id -> vessel id from anatomy.yaml `bullseye` (throws on duplicates). */
export function segmentVesselMap(bullseye: Record<string, number[]>): Map<number, string> {
  const m = new Map<number, string>()
  for (const [vessel, ids] of Object.entries(bullseye))
    for (const id of ids) {
      if (m.has(id)) throw new Error(`segment ${id} assigned to both ${m.get(id)} and ${vessel}`)
      m.set(id, vessel)
    }
  return m
}

const pt = (r: number, deg: number) => {
  const a = (deg * Math.PI) / 180
  return `${(r * Math.cos(a)).toFixed(2)} ${(-r * Math.sin(a)).toFixed(2)}`
}

function sectorPath(s: Segment): string {
  if (s.a1 - s.a0 >= 360) return `M ${s.r1} 0 A ${s.r1} ${s.r1} 0 1 0 ${-s.r1} 0 A ${s.r1} ${s.r1} 0 1 0 ${s.r1} 0 Z`
  const large = s.a1 - s.a0 > 180 ? 1 : 0
  return `M ${pt(s.r1, s.a0)} A ${s.r1} ${s.r1} 0 ${large} 0 ${pt(s.r1, s.a1)} L ${pt(s.r0, s.a1)} A ${s.r0} ${s.r0} 0 ${large} 1 ${pt(s.r0, s.a0)} Z`
}

/** SVG bullseye coloured by the probability of the vessel that typically supplies each segment; synced with 3D. */
export default function Bullseye({ size = 220, className = '' }: { size?: number; className?: string }) {
  const bullseye = useStore((s) => s.config?.anatomy.bullseye)
  const analysis = useStore((s) => s.analysis)
  const hovered = useStore((s) => s.hovered)
  const selected = useStore((s) => s.selected)
  const hoveredSegment = useStore((s) => s.hoveredSegment)
  const hoverSegment = useStore((s) => s.hoverSegment)
  const select = useStore((s) => s.select)
  if (!bullseye) return null
  const map = segmentVesselMap(bullseye)
  const segs = bullseyeSegments()

  const onKey = (e: KeyboardEvent, vessel: string) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      select(selected === vessel ? null : (vessel as TargetId))
    }
  }

  return (
    <figure className={`inline-flex flex-col items-center gap-1 ${className}`} data-testid="bullseye">
      <svg viewBox="-100 -100 200 200" width={size} height={size} role="group" aria-label="AHA 17-segment bullseye">
        {segs.map((s) => {
          const vessel = map.get(s.id) ?? ''
          const p = analysis?.vessels[vessel]?.p
          const fill = ARTERY_COLORS[vessel] ?? NEUTRAL_VESSEL // which artery usually supplies it (not risk)
          const segHot = s.id === hoveredSegment // the exact segment hovered here or on the 3D heart
          const hot = segHot || vessel === hovered || vessel === selected
          const dim = (hovered || selected) && !hot
          const mid = s.ring === 'apex' ? 0 : (s.r0 + s.r1) / 2
          const am = (s.a0 + s.a1) / 2
          return (
            <g
              key={s.id}
              role="button"
              tabIndex={0}
              aria-label={`Segment ${s.id}, ${s.name}, ${vessel}${p != null ? ` ${Math.round(p * 100)}%` : ''}`}
              onMouseEnter={() => hoverSegment(s.id, vessel)}
              onMouseLeave={() => hoverSegment(null, null)}
              onFocus={() => hoverSegment(s.id, vessel)}
              onBlur={() => hoverSegment(null, null)}
              onClick={() => select(selected === vessel ? null : (vessel as TargetId))}
              onKeyDown={(e) => onKey(e, vessel)}
              className="cursor-pointer outline-none"
              data-segment={s.id}
              data-hot={segHot ? 'segment' : hot ? 'vessel' : undefined}
              data-vessel={vessel}
            >
              <path
                d={sectorPath(s)}
                fill={fill}
                fillOpacity={dim ? 0.45 : 1}
                stroke={segHot ? '#ffffff' : hot ? 'var(--color-accent)' : 'var(--color-ink)'}
                strokeWidth={segHot ? 4 : hot ? 2.5 : 1.2}
              />
              <text
                x={mid * Math.cos((am * Math.PI) / 180)}
                y={-mid * Math.sin((am * Math.PI) / 180)}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={10}
                fill="#0b1118"
                pointerEvents="none"
              >
                {s.id}
              </text>
            </g>
          )
        })}
      </svg>
      <figcaption className="text-center text-[10px] leading-tight text-muted">
        <span className="mb-0.5 flex justify-center gap-2">
          {Object.keys(bullseye).map((v) => (
            <span key={v} className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-sm" style={{ background: ARTERY_COLORS[v] }} />
              {v}
            </span>
          ))}
        </span>
        Colour = artery that usually supplies the segment (AHA), not risk · illustrative; coronary dominance varies · anterior
        top, septum left
      </figcaption>
    </figure>
  )
}
