import type { Status, StatusReason } from '../types/api'

export const pct = (p: number | null | undefined, digits = 0) =>
  p == null ? '–' : `${(p * 100).toFixed(digits)}%`

export const signedPct = (d: number, digits = 1) => `${d >= 0 ? '+' : '−'}${Math.abs(d * 100).toFixed(digits)} pts`

// Input confidence (mirrors ml/input_confidence.py): share of the model's usual signal that comes from imputed inputs.
export const REDUCED_FROM = 0.1
export const UNCERTAIN_FROM = 0.25
export const LOW_FROM = 0.5
/** 0 = shown normally, 1 = fully greyed; grows linearly with the missing-signal share between 10% and 50%. */
export const confidenceFade = (share: number | undefined) =>
  share == null ? 0 : Math.min(1, Math.max(0, (share - REDUCED_FROM) / (LOW_FROM - REDUCED_FROM)))
/** CSS filter for low-confidence elements: desaturate and dim in proportion to the fade. */
export const fadeFilter = (fade: number) => (fade > 0 ? `saturate(${(1 - fade).toFixed(2)}) opacity(${(1 - 0.35 * fade).toFixed(2)})` : undefined)
export const LOW_CONFIDENCE_TEXT = 'Estimate based mostly on cohort averages'

export const STATUS_LABEL: Record<Status, string> = {
  likely: 'Likely',
  uncertain: 'Uncertain',
  unlikely: 'Unlikely',
}

export const STATUS_HELP: Record<Status, string> = {
  likely: 'The 90% conformal prediction set contains only "stenosis", and it agrees with the threshold decision.',
  uncertain:
    'The 90% conformal set contains both outcomes (patients without stenosis in this artery also often score this high), ' +
    'or it disagrees with the threshold decision, or the patient is out of distribution, or missing inputs carry ' +
    '25% or more of the model\'s usual signal. ' +
    'Calibrated per artery, so a higher % can still be uncertain.',
  unlikely: 'The 90% conformal prediction set contains only "no stenosis", and it agrees with the threshold decision.',
}

/** Inline reason shown under every status chip (mirrors ml/conformal.status_reason and the predictor overrides). */
export function statusReasonText(status: Status, reason: StatusReason): string {
  switch (reason) {
    case 'single_outcome':
      return status === 'likely' ? 'only "stenosis" plausible at 90% confidence' : 'only "no stenosis" plausible at 90% confidence'
    case 'both_outcomes':
      return 'both outcomes plausible at 90% confidence'
    case 'empty_set':
      return 'neither outcome fits well at 90% confidence'
    case 'threshold_disagrees':
      return '90% set and threshold point different ways'
    case 'out_of_distribution':
      return 'patient unlike the training data'
    case 'missing_inputs':
      return 'too much of the input is missing'
  }
}

export const STATUS_LEGEND =
  'Status comes from a per-artery 90% conformal set, calibrated separately for each artery, so a higher probability can still be "uncertain".'

export const STATUS_CLASS: Record<Status, string> = {
  likely: 'bg-rose-500/15 text-rose-300 ring-rose-400/40',
  uncertain: 'bg-warn/15 text-warn ring-warn/40 [background-image:repeating-linear-gradient(45deg,transparent_0_4px,rgba(246,195,91,.15)_4px_8px)]',
  unlikely: 'bg-emerald-500/15 text-emerald-300 ring-emerald-400/40',
}
