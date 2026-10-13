import type { Status } from '../types/api'

export const pct = (p: number | null | undefined, digits = 0) =>
  p == null ? '–' : `${(p * 100).toFixed(digits)}%`

export const signedPct = (d: number, digits = 1) => `${d >= 0 ? '+' : '−'}${Math.abs(d * 100).toFixed(digits)} pts`

export const STATUS_LABEL: Record<Status, string> = {
  likely: 'Likely',
  uncertain: 'Uncertain',
  unlikely: 'Unlikely',
}

export const STATUS_HELP: Record<Status, string> = {
  likely: 'The 90% conformal prediction set contains only "stenosis", and it agrees with the threshold decision.',
  uncertain:
    'The 90% conformal set contains both outcomes (patients without stenosis in this artery also often score this high), ' +
    'or it disagrees with the threshold decision. Calibrated per artery, so a higher % can still be uncertain.',
  unlikely: 'The 90% conformal prediction set contains only "no stenosis", and it agrees with the threshold decision.',
}

export const STATUS_CLASS: Record<Status, string> = {
  likely: 'bg-rose-500/15 text-rose-300 ring-rose-400/40',
  uncertain: 'bg-warn/15 text-warn ring-warn/40 [background-image:repeating-linear-gradient(45deg,transparent_0_4px,rgba(246,195,91,.15)_4px_8px)]',
  unlikely: 'bg-emerald-500/15 text-emerald-300 ring-emerald-400/40',
}
