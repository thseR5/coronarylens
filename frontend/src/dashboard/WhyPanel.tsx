import { useState } from 'react'
import { useStore } from '../store'
import { pct } from '../lib/risk'
import type { Contribution, TargetId } from '../types/api'
import { binaryText, featureMeta, TARGET_LABEL } from './labels'

const TOP_N = 8
const fmtValue = (v: Contribution['value'], type?: string, f?: { value_labels?: Record<string, string> }) =>
  v == null
    ? ''
    : type === 'binary'
      ? binaryText(f, v)
      : typeof v === 'number'
        ? Number.isInteger(v)
          ? String(v)
          : v.toFixed(1)
        : v

export default function WhyPanel() {
  const config = useStore((s) => s.config)
  const explanation = useStore((s) => s.explanation)
  const loading = useStore((s) => s.loading.explain)
  const selected = useStore((s) => s.selected)
  const select = useStore((s) => s.select)
  const [byGroup, setByGroup] = useState(false)
  const [patientView, setPatientView] = useState(false)

  const target: TargetId = selected ?? 'CAD'
  const ex = explanation?.targets[target]
  const targets: TargetId[] = ['CAD', 'LAD', 'LCX', 'RCA', 'P_GE2']

  let rows: { id: string; label: string; value: string; shap: number }[] = []
  let diffuse = false // the summed "other features" bar is larger than any single feature
  if (ex) {
    if (byGroup) {
      rows = Object.entries(ex.group_totals)
        .map(([g, s]) => ({ id: g, label: featureMeta(config, g).label, value: '', shap: s }))
        .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
    } else {
      const sorted = [...ex.contributions].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
      rows = sorted.slice(0, TOP_N).map((c) => {
        const m = featureMeta(config, c.feature)
        const unit = m.unit && c.value != null && m.type !== 'binary' ? ` ${m.unit}` : ''
        return { id: c.feature, label: m.label, value: `${fmtValue(c.value, m.type, m)}${unit}`, shap: c.shap }
      })
      const rest = sorted.slice(TOP_N).reduce((a, c) => a + c.shap, 0)
      if (sorted.length > TOP_N) rows.push({ id: 'rest', label: `${sorted.length - TOP_N} other features`, value: '', shap: rest })
      diffuse = sorted.length > TOP_N && Math.abs(rest) > Math.abs(sorted[0].shap)
    }
  }

  // Waterfall geometry on the probability axis.
  const steps: { start: number; end: number }[] = []
  let cur = ex?.base_value ?? 0
  for (const r of rows) {
    steps.push({ start: cur, end: cur + r.shap })
    cur += r.shap
  }
  const all = [ex?.base_value ?? 0, ex?.output ?? 0, ...steps.flatMap((s) => [s.start, s.end])]
  const lo = Math.max(0, Math.min(...all) - 0.05)
  const hi = Math.min(1, Math.max(...all) + 0.05)
  const x = (p: number) => `${((p - lo) / (hi - lo || 1)) * 100}%`

  return (
    <div className="card flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <h2 className="h-section !mb-0">Why this estimate?</h2>
        <div className="flex gap-1 text-[11px]">
          <button type="button" className={`btn !py-0.5 ${byGroup ? '!ring-accent' : ''}`} aria-pressed={byGroup} onClick={() => setByGroup(!byGroup)}>
            By group
          </button>
          <button type="button" className={`btn !py-0.5 ${patientView ? '!ring-accent' : ''}`} aria-pressed={patientView} onClick={() => setPatientView(!patientView)}>
            Patient view
          </button>
        </div>
      </div>

      <div role="tablist" aria-label="Explained output" className="flex flex-wrap gap-1">
        {targets.map((t) => (
          <button
            key={t}
            role="tab"
            type="button"
            aria-selected={t === target}
            onClick={() => select(t === 'CAD' ? null : t)}
            className={`rounded px-2 py-0.5 text-[11px] ring-1 ${t === target ? 'bg-accent/20 ring-accent/60' : 'text-muted ring-line'}`}
          >
            {t === 'P_GE2' ? '≥2 vessels' : t}
          </button>
        ))}
      </div>

      {!ex ? (
        <div className="flex flex-col gap-1.5" aria-busy={loading}>
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-5 animate-pulse rounded bg-line/40" />
          ))}
        </div>
      ) : (
        <>
          <div className={`text-xs text-muted transition-opacity ${loading ? 'opacity-60' : ''}`} aria-busy={loading}>
            {loading && <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-accent align-middle" />}
            {TARGET_LABEL[target]}: from the cohort average <span className="font-mono text-text">{pct(ex.base_value)}</span> to this
            patient's <span className="font-mono text-text">{pct(ex.output)}</span>
          </div>
          <div className={`flex flex-col gap-1 transition-opacity ${loading ? 'opacity-60' : ''}`} role="list" aria-label="Feature contributions">
            {rows.map((r, i) => {
              const s = steps[i]
              const left = Math.min(s.start, s.end)
              const up = r.shap >= 0
              return (
                <div key={r.id} role="listitem" className="grid grid-cols-[minmax(0,9.5rem)_1fr_3.5rem] items-center gap-2 text-xs">
                  <span className="truncate" title={`${r.label} ${r.value}`}>
                    {r.label} <span className="text-muted">{r.value}</span>
                  </span>
                  <div className="relative h-4">
                    <div className="absolute inset-y-0 w-px bg-line" style={{ left: x(ex.base_value) }} />
                    <div
                      className={`absolute inset-y-0.5 rounded-sm ${up ? 'bg-rose-400/80' : 'bg-sky-400/80'}`}
                      style={{ left: x(left), width: `max(2px, calc(${x(left + Math.abs(r.shap))} - ${x(left)}))`, transition: 'all 300ms ease' }}
                    />
                  </div>
                  <span className={`text-right font-mono ${up ? 'text-rose-300' : 'text-sky-300'}`}>
                    {up ? '+' : '−'}
                    {Math.abs(r.shap * 100).toFixed(1)}
                  </span>
                </div>
              )
            })}
          </div>
          {diffuse && (
            <p className="rounded bg-warn/10 px-2 py-1 text-[11px] text-warn ring-1 ring-warn/30" data-testid="diffuse-note">
              Signal for this artery is spread thinly across many features. No single factor dominates.
            </p>
          )}
          <ul className="list-disc space-y-1 pl-4 text-xs text-text/90">
            {(patientView ? ex.narrative_patient : ex.narrative).map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
          {target !== 'CAD' && target !== 'P_GE2' && (
            <p className="rounded bg-ink/40 px-2 py-1 text-[10px] text-muted">
              Vessel estimates mostly reflect overall CAD risk. Among patients who have CAD, the model separates LAD from the other
              arteries only modestly and LCX/RCA no better than chance (within-CAD AUC LAD 0.69, LCX 0.52, RCA 0.57), so similar
              drivers appear for every vessel.
            </p>
          )}
          <p className="text-[10px] text-muted">
            Contributions are SHAP values ({explanation?.method}) in percentage points on the same output that is displayed.
            They describe model behaviour, not causation. Red raises the estimate, blue lowers it.
          </p>
        </>
      )}
    </div>
  )
}
