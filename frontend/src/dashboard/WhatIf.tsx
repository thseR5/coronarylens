import { useState } from 'react'
import { api } from '../api/client'
import { useStore } from '../store'
import { pct } from '../lib/risk'
import type { CounterfactualResponse, TargetId } from '../types/api'
import { featureMeta, TARGET_LABEL } from './labels'
import { useWhatIf } from './whatIfStore'

export default function WhatIf() {
  const config = useStore((s) => s.config)
  const features = useStore((s) => s.features)
  const analysis = useStore((s) => s.analysis)
  const selected = useStore((s) => s.selected)
  const setFeatures = useStore((s) => s.setFeatures)
  const { baseline, pin, clear } = useWhatIf()
  const [cf, setCf] = useState<CounterfactualResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const target: TargetId = selected ?? 'CAD'

  const changed = baseline
    ? Object.keys(features).filter((k) => String(features[k]) !== String(baseline.features[k]))
    : []

  const suggest = async () => {
    setBusy(true)
    setErr(null)
    try {
      setCf(await api.counterfactual(features, target))
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card flex flex-col gap-2">
      <h2 className="h-section !mb-0">What-if</h2>
      <div className="flex flex-wrap gap-2">
        <button type="button" className="btn" disabled={!analysis} onClick={() => analysis && pin(features, analysis)}>
          {baseline ? 'Re-pin baseline' : 'Pin current as baseline'}
        </button>
        {baseline && (
          <>
            <button type="button" className="btn" onClick={() => setFeatures(baseline.features)}>
              Restore baseline
            </button>
            <button type="button" className="btn" onClick={clear}>
              Unpin
            </button>
          </>
        )}
      </div>
      {baseline && (
        <p className="text-xs text-muted">
          Edit any input: deltas appear on every probability and the 3D vessels recolor live.
          {changed.length > 0 && (
            <> Changed: {changed.map((k) => `${featureMeta(config, k).label} ${baseline.features[k]}→${features[k]}`).join(', ')}.</>
          )}
        </p>
      )}

      <div className="mt-1 flex items-center justify-between gap-2">
        <span className="text-xs text-muted">Smallest changes to modifiable factors that lower {TARGET_LABEL[target]}</span>
        <button type="button" className="btn shrink-0" disabled={busy || !analysis} onClick={suggest}>
          {busy ? 'Searching…' : 'Suggest'}
        </button>
      </div>
      {err && <p className="text-xs text-warn">{err}</p>}
      {cf && (
        <ul className="flex flex-col gap-1.5">
          {cf.counterfactuals.length === 0 && <li className="text-xs text-muted">No plausible change within bounds found.</li>}
          {cf.counterfactuals.map((c, i) => (
            <li key={i} className="flex items-center justify-between gap-2 rounded-md bg-ink/40 px-2 py-1.5 text-xs">
              <span>
                {c.changes.map((ch) => `${featureMeta(config, ch.feature).label} ${ch.from_value}→${ch.to_value}`).join(' · ')}
              </span>
              <span className="shrink-0 font-mono">
                {pct(c.p_before)}→<span className="text-emerald-300">{pct(c.p_after)}</span>
              </span>
              <button
                type="button"
                className="btn !py-0.5"
                onClick={() => {
                  if (analysis && !baseline) pin(features, analysis)
                  setFeatures({ ...features, ...Object.fromEntries(c.changes.map((ch) => [ch.feature, ch.to_value])) })
                }}
              >
                Apply
              </button>
            </li>
          ))}
          <li className="text-[10px] text-muted">{cf.note}</li>
        </ul>
      )}
    </div>
  )
}
