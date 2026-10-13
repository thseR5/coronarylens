import { useStore } from '../store'
import { confidenceFade, fadeFilter, LOW_CONFIDENCE_TEXT, pct, signedPct } from '../lib/risk'
import { riskColor } from '../lib/color'
import { useWhatIf } from './whatIfStore'
import StatusChip from './StatusChip'

function Gauge({ p, lo, hi, threshold }: { p: number; lo: number; hi: number; threshold: number }) {
  // Semicircle gauge; arc length proportional to probability. Interval drawn as a thin outer band.
  const R = 54
  const C = Math.PI * R
  const at = (v: number) => {
    const a = Math.PI * (1 - v)
    return [60 + R * Math.cos(a), 62 - R * Math.sin(a)]
  }
  const [tx, ty] = at(threshold)
  return (
    <svg viewBox="0 0 120 70" className="w-36" role="img" aria-label={`CAD probability ${pct(p)}`}>
      <path d="M6 62 A54 54 0 0 1 114 62" fill="none" stroke="var(--color-line)" strokeWidth="9" strokeLinecap="round" />
      <path
        d="M6 62 A54 54 0 0 1 114 62"
        fill="none"
        stroke={riskColor(p)}
        strokeWidth="9"
        strokeLinecap="round"
        strokeDasharray={`${C * p} ${C}`}
        style={{ transition: 'stroke-dasharray 400ms ease, stroke 400ms ease' }}
      />
      <path
        d="M-2 62 A62 62 0 0 1 122 62"
        fill="none"
        stroke="var(--color-text)"
        strokeOpacity="0.5"
        strokeWidth="2"
        pathLength={1}
        strokeDasharray={`0 ${lo} ${Math.max(hi - lo, 0.004)} 1`}
      />
      <circle cx={tx} cy={ty} r="2.5" fill="var(--color-text)" />
      <text x="60" y="58" textAnchor="middle" className="fill-text font-mono" fontSize="20">
        {pct(p)}
      </text>
    </svg>
  )
}


/** Three significant digits, e.g. 18.365473 → 18.4 (ranges come from raw training data). */
const round3 = (x: number) => Number(x.toPrecision(3))
export default function RiskSummary() {
  const a = useStore((s) => s.analysis)
  const nFeatures = useStore((s) => s.config?.features.features.length ?? 54)
  const mode = useStore((s) => s.mode)
  const setMode = useStore((s) => s.setMode)
  const baseline = useWhatIf((s) => s.baseline)
  if (!a) return <div className="card h-40 animate-pulse" aria-busy="true" />

  const dist = a.multivessel.n_vessels_dist
  const conf = a.input_confidence
  const fade = confidenceFade(conf.missing_signal_share)
  const delta = baseline ? a.cad.p - baseline.analysis.cad.p : null

  return (
    <div className="card flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h2 className="h-section !mb-0">Overall CAD</h2>
        <div role="radiogroup" aria-label="Operating threshold" className="flex rounded-md bg-ink/50 p-0.5 text-[11px] ring-1 ring-line">
          {(['f1', 'screening'] as const).map((m) => (
            <button
              key={m}
              role="radio"
              aria-checked={mode === m}
              type="button"
              onClick={() => setMode(m)}
              className={`rounded px-2 py-0.5 ${mode === m ? 'bg-accent/20 text-text' : 'text-muted'}`}
              title={m === 'f1' ? 'Threshold that maximises F1 on out-of-fold predictions' : 'Threshold with sensitivity ≥ 0.90'}
            >
              {m === 'f1' ? 'Balanced' : 'Screening'}
            </button>
          ))}
        </div>
      </div>
      <div className="flex items-center gap-4">
        <div
          className="relative shrink-0"
          style={{ filter: fadeFilter(fade) }}
          data-testid="cad-gauge"
          data-confidence={conf.level}
          data-fade={fade.toFixed(2)}
        >
          <Gauge p={a.cad.p} lo={a.cad.interval[0]} hi={a.cad.interval[1]} threshold={a.cad.threshold} />
          {conf.level === 'low' && (
            <div
              aria-hidden
              className="pointer-events-none absolute inset-0 rounded-md [background-image:repeating-linear-gradient(45deg,transparent_0_5px,rgba(148,163,184,.28)_5px_9px)]"
            />
          )}
        </div>
        <div className="flex flex-col gap-1.5 text-sm">
          <div className="flex items-center gap-2">
            <span className={`font-semibold ${a.cad.positive ? 'text-rose-300' : 'text-emerald-300'}`}>
              {a.cad.positive ? 'CAD predicted' : 'CAD not predicted'}
            </span>
            <StatusChip status={a.cad.status} reason={a.cad.reason} />
          </div>
          <span className="text-xs text-muted">
            90% interval {pct(a.cad.interval[0])}–{pct(a.cad.interval[1])} · threshold {pct(a.cad.threshold)}
          </span>
          {conf.level === 'low' && (
            <span className="text-xs font-semibold text-warn" data-testid="low-confidence-label">
              {LOW_CONFIDENCE_TEXT}
            </span>
          )}
          {delta != null && (
            <span className={`font-mono text-xs ${delta > 0 ? 'text-rose-300' : 'text-emerald-300'}`}>
              {signedPct(delta)} vs pinned baseline
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="rounded-md bg-ink/40 p-2">
          <div className="text-muted">Multi-vessel (≥2)</div>
          <div className="font-mono text-lg">{pct(a.multivessel.p_ge2)}</div>
        </div>
        <div className="rounded-md bg-ink/40 p-2">
          <div className="text-muted">Three-vessel</div>
          <div className="font-mono text-lg">{pct(a.multivessel.p_3vd)}</div>
        </div>
      </div>

      <div>
        <div className="mb-1 text-[11px] text-muted">Number of diseased vessels</div>
        <div className="flex h-5 overflow-hidden rounded ring-1 ring-line" role="img" aria-label="Distribution of number of diseased vessels">
          {(['0', '1', '2', '3'] as const).map((k, i) => (
            <div
              key={k}
              title={`${k} vessels: ${pct(dist[k], 1)}`}
              style={{ width: `${dist[k] * 100}%`, background: riskColor(i / 3), transition: 'width 400ms ease' }}
              className="flex items-center justify-center text-[10px] font-semibold text-ink"
            >
              {dist[k] > 0.08 ? k : ''}
            </div>
          ))}
        </div>
      </div>

      {conf.level !== 'full' && (
        <div role="status" className="rounded-md bg-warn/10 px-2 py-1.5 text-xs text-warn ring-1 ring-warn/40" data-testid="imputed-warning">
          <span className="font-semibold">
            {conf.level === 'low' ? `Low confidence: ${LOW_CONFIDENCE_TEXT.toLowerCase()}.` : 'Reduced confidence.'}
          </span>{' '}
          {a.imputed.length} of {nFeatures} inputs not entered; they carry {pct(conf.missing_signal_share)} of the model's usual
          signal and use cohort averages{conf.forces_uncertain ? ', so every status is shown as uncertain' : ''}. Colours fade as
          more of the signal is missing. Load a preset or enter the patient's findings.
        </div>
      )}
      {conf.level === 'full' && a.imputed.length > 0 && (
        <div className="text-[11px] text-muted" data-testid="imputed-note">
          {a.imputed.length} minor input{a.imputed.length > 1 ? 's' : ''} not entered ({pct(conf.missing_signal_share, 1)} of the
          model's usual signal): cohort averages used.
        </div>
      )}
      {!a.applicability.in_domain && (
        <div role="status" className="rounded-md bg-warn/10 px-2 py-1.5 text-xs text-warn ring-1 ring-warn/40">
          Outside training distribution: interpret with extra caution.
        </div>
      )}
      {a.out_of_range.length > 0 && (
        <div className="text-[11px] text-warn">
          Outside the training data range: {a.out_of_range.map((o) => `${o.feature} = ${round3(o.value)} (training ${round3(o.range[0])}–${round3(o.range[1])})`).join(', ')}
        </div>
      )}
    </div>
  )
}
