import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import { useStore } from '../store'
import { binaryText } from './labels'
import type { FeatureValue, Features, Preset } from '../types/api'
import type { FeatureSpec } from '../types/config'

export const defaultFeatures = (specs: FeatureSpec[]): Features =>
  Object.fromEntries(specs.map((f) => [f.id, f.default]))

const NO_IMPUTED: string[] = []

const outOfRef = (f: FeatureSpec, v: FeatureValue) =>
  f.type === 'numeric' && f.ref != null && typeof v === 'number' && (v < f.ref[0] || v > f.ref[1])

const step = (f: FeatureSpec) => {
  if (!f.range) return 1
  const span = f.range[1] - f.range[0]
  return span <= 20 ? 0.1 : span <= 300 ? 1 : 10
}

function randomPatient(specs: FeatureSpec[]): Features {
  const rnd = Math.random
  return Object.fromEntries(
    specs.map((f) => {
      if (f.type === 'binary') return [f.id, rnd() < 0.3 ? 1 : 0]
      if (f.type === 'categorical' || f.type === 'ordinal') {
        const lv = f.levels ?? []
        return [f.id, rnd() < 0.6 ? f.default : lv[Math.floor(rnd() * lv.length)]]
      }
      const d = Number(f.default)
      const [lo, hi] = f.ref ?? [d * 0.85, d * 1.15]
      const v = lo + rnd() * (hi - lo) * 1.3 - (hi - lo) * 0.15
      return [f.id, Number(v.toFixed(step(f) < 1 ? 1 : 0))]
    }),
  )
}

function FieldInput({ f, value, onChange }: { f: FeatureSpec; value: FeatureValue; onChange: (v: FeatureValue) => void }) {
  const flagged = outOfRef(f, value)
  const id = `feat-${f.id}`
  const label = (
    <label htmlFor={id} className="flex items-baseline justify-between gap-2 text-xs">
      <span className="text-text">{f.label}</span>
      <span className={flagged ? 'text-warn' : 'text-muted'}>
        {f.ref ? `ref ${f.ref[0]}–${f.ref[1]}` : ''} {f.unit}
      </span>
    </label>
  )

  if (f.type === 'binary') {
    const on = Number(value) === 1
    return (
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={on}
        title={f.description}
        onClick={() => onChange(on ? 0 : 1)}
        className={`flex items-center justify-between rounded-md px-2.5 py-1.5 text-left text-xs ring-1 transition ${
          f.value_labels // two named values (e.g. Female / Male): neither is an "on" state
            ? 'bg-panel-2 text-text ring-line hover:ring-accent/50'
            : on
              ? 'bg-accent/15 text-text ring-accent/50'
              : 'bg-panel-2 text-muted ring-line hover:text-text'
        }`}
      >
        <span>{f.label}</span>
        <span className="font-mono">{binaryText(f, value)}</span>
      </button>
    )
  }

  if (f.type === 'categorical' || f.type === 'ordinal') {
    return (
      <div className="col-span-2 flex flex-col gap-1" title={f.description}>
        {label}
        <select
          id={id}
          value={String(value ?? '')}
          onChange={(e) => {
            const raw = e.target.value
            onChange(f.levels?.some((l) => typeof l === 'number') ? Number(raw) : raw)
          }}
          className="rounded-md bg-panel-2 px-2 py-1.5 text-sm ring-1 ring-line focus:ring-accent focus:outline-none"
        >
          {(f.levels ?? []).map((l) => (
            <option key={String(l)} value={String(l)}>
              {String(l)}
            </option>
          ))}
        </select>
      </div>
    )
  }

  const [lo, hi] = f.range ?? [0, 100]
  const num = typeof value === 'number' ? value : Number(value ?? f.default)
  return (
    <div className="col-span-2 flex flex-col gap-1" title={f.description}>
      {label}
      <div className="flex items-center gap-2">
        <input
          aria-label={`${f.label} slider`}
          type="range"
          min={lo}
          max={hi}
          step={step(f)}
          value={num}
          onChange={(e) => onChange(Number(e.target.value))}
          className="flex-1 accent-[var(--color-accent)]"
        />
        <input
          id={id}
          type="number"
          min={lo}
          max={hi}
          step={step(f)}
          value={num}
          onChange={(e) => e.target.value !== '' && onChange(Number(e.target.value))}
          className={`w-20 rounded-md bg-panel-2 px-2 py-1 text-right font-mono text-sm ring-1 focus:outline-none ${
            flagged ? 'text-warn ring-warn/60' : 'ring-line focus:ring-accent'
          }`}
        />
      </div>
    </div>
  )
}

export default function PatientInput() {
  const config = useStore((s) => s.config)
  const features = useStore((s) => s.features)
  const setFeatures = useStore((s) => s.setFeatures)
  const setFeature = useStore((s) => s.setFeature)
  const imputed = useStore((s) => s.analysis?.imputed ?? NO_IMPUTED)
  const [full, setFull] = useState(false)
  const [presets, setPresets] = useState<Preset[]>([])
  const presetId = useStore((s) => s.presetId) ?? ''
  const setPresetId = useStore((s) => s.setPresetId)
  const [open, setOpen] = useState<Record<string, boolean>>({})

  const specs = useMemo(() => config?.features.features ?? [], [config])
  const groups = config?.features.groups ?? []

  useEffect(() => {
    api
      .presets()
      .then((ps) => {
        setPresets(ps)
        // land on a real patient (Low risk), not on mode-imputed defaults (which describe a typical angina patient)
        // LAD-only: a real test patient with no out-of-range inputs
        const first = ps.find((p) => p.id === 'lad-only') ?? ps[0]
        const st = useStore.getState()
        if (first && specs.length) {
          setPresetId(first.id)
          st.setFeatures({ ...defaultFeatures(specs), ...first.features })
        }
      })
      .catch(() => setPresets([]))
  }, [specs])


  const shown = full ? specs : specs.filter((f) => f.quick)
  const preset = presets.find((p) => p.id === presetId)
  const modified = !!preset && Object.entries(preset.features).some(([k, v]) => String(features[k]) !== String(v))

  const loadPreset = (id: string) => {
    setPresetId(id)
    const p = presets.find((x) => x.id === id)
    if (p) setFeatures({ ...defaultFeatures(specs), ...p.features })
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <select
          aria-label="Load preset patient"
          value={presetId}
          onChange={(e) => loadPreset(e.target.value)}
          className="min-w-0 flex-1 rounded-md bg-panel-2 px-2 py-1.5 text-sm ring-1 ring-line"
        >
          <option value="">{presets.length ? 'Preset patients…' : 'No presets (backend offline)'}</option>
          {presets.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="btn"
          title="Clear all inputs: missing values are imputed with typical cohort values and the estimate is shown as uncertain"
          onClick={() => {
            setPresetId(null)
            setFeatures({}) // nothing entered: the server imputes and reports every field
          }}
        >
          Reset
        </button>
        <button type="button" className="btn" onClick={() => { setPresetId(null); setFeatures(randomPatient(specs)) }}>
          Randomize
        </button>
      </div>
      {preset && !modified && (
        <p className="text-xs text-muted">
          {preset.description}
          {preset.true_pattern && (
            <> · true angiography pattern (LAD/LCX/RCA): <span className="font-mono text-text">{preset.true_pattern}</span></>
          )}
        </p>
      )}
      {preset && modified && (
        <p className="text-xs text-warn" data-testid="preset-modified">
          {preset.label} (modified): the true angiography result no longer applies to these inputs.
        </p>
      )}

      <div role="tablist" aria-label="Input mode" className="flex rounded-md bg-panel-2 p-0.5 text-xs ring-1 ring-line">
        {[
          ['Quick', false],
          ['Full', true],
        ].map(([label, val]) => (
          <button
            key={String(label)}
            role="tab"
            aria-selected={full === val}
            type="button"
            onClick={() => setFull(val as boolean)}
            className={`flex-1 rounded px-2 py-1 ${full === val ? 'bg-accent/20 text-text' : 'text-muted'}`}
          >
            {label as string} {val ? `(${specs.length})` : `(${specs.filter((f) => f.quick).length})`}
          </button>
        ))}
      </div>

      {groups.map((g) => {
        const fs = shown.filter((f) => f.group === g.id)
        if (!fs.length) return null
        const isOpen = !full || open[g.id] !== false
        return (
          <fieldset key={g.id} className="rounded-lg border border-line">
            <legend className="px-1">
              <button
                type="button"
                aria-expanded={isOpen}
                onClick={() => setOpen((o) => ({ ...o, [g.id]: !isOpen }))}
                className="text-[11px] font-semibold uppercase tracking-wider text-muted hover:text-text"
              >
                {isOpen ? '▾' : '▸'} {g.label}
              </button>
            </legend>
            {isOpen && (
              <div className="grid grid-cols-2 gap-2 p-2">
                {fs.map((f) => (
                  <FieldInput key={f.id} f={f} value={features[f.id] ?? f.default} onChange={(v) => setFeature(f.id, v)} />
                ))}
              </div>
            )}
          </fieldset>
        )
      })}

      {!full && (
        <p className="text-[11px] text-muted">
          Quick mode shows the most influential inputs; the other {specs.length - shown.length} keep their current values
          (defaults are typical values for this cohort).
          {imputed.length > 0 && <> Imputed by the model: {imputed.join(', ')}.</>}
        </p>
      )}
    </div>
  )
}
