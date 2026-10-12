import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import { riskColor } from '../lib/color'
import { useStore } from '../store'
import type { CohortMapResponse } from '../types/api'

const W = 400
const H = 230
const PAD = 14

/** 2-D map of the training cohort (PCA); the current patient is the white star. Click a dot to load that patient. */
export default function CohortMap() {
  const features = useStore((s) => s.features)
  const analysis = useStore((s) => s.analysis)
  const setFeatures = useStore((s) => s.setFeatures)
  const [data, setData] = useState<CohortMapResponse | null>(null)
  const [hover, setHover] = useState<number | null>(null)
  const [loaded, setLoaded] = useState<{ index: number; pattern: string } | null>(null)

  useEffect(() => {
    if (!analysis) return
    let live = true
    api.cohortMap(features).then((d) => live && setData(d)).catch(() => live && setData(null))
    return () => {
      live = false
    }
  }, [analysis, features])

  const scale = useMemo(() => {
    if (!data) return null
    const xs = [...data.points.map((p) => p.x), data.patient.x]
    const ys = [...data.points.map((p) => p.y), data.patient.y]
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
    return {
      x: (v: number) => PAD + ((v - x0) / (x1 - x0 || 1)) * (W - 2 * PAD),
      y: (v: number) => H - PAD - ((v - y0) / (y1 - y0 || 1)) * (H - 2 * PAD),
    }
  }, [data])

  if (!data || !scale) return null
  const hp = hover != null ? data.points.find((p) => p.index === hover) : null
  const ev = data.explained_variance.reduce((a, b) => a + b, 0)

  const load = async (index: number) => {
    const p = await api.trainingPatient(index)
    setFeatures({ ...useStore.getState().features, ...p.features })
    setLoaded({ index, pattern: p.pattern })
  }

  return (
    <div className="card">
      <h2 className="h-section">Cohort map · where this patient sits among training patients</h2>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Cohort map of training patients">
        {data.points.map((p) => (
          <circle
            key={p.index}
            cx={scale.x(p.x)}
            cy={scale.y(p.y)}
            r={hover === p.index ? 5 : 3.2}
            fill={riskColor(p.n_vessels / 3)}
            fillOpacity={0.85}
            stroke={hover === p.index ? '#fff' : 'none'}
            className="cursor-pointer"
            onMouseEnter={() => setHover(p.index)}
            onMouseLeave={() => setHover(null)}
            onClick={() => load(p.index)}
            data-testid={`cohort-${p.index}`}
          >
            <title>{`Patient #${p.index}: ${p.n_vessels} diseased vessel(s), pattern ${p.pattern}`}</title>
          </circle>
        ))}
        <path
          d="M0,-8 L2.2,-2.6 8,-2.6 3.3,1 5,7 0,3.4 -5,7 -3.3,1 -8,-2.6 -2.2,-2.6 Z"
          transform={`translate(${scale.x(data.patient.x)},${scale.y(data.patient.y)})`}
          fill="#fff"
          stroke="#0b1118"
          strokeWidth={1}
          aria-label="Current patient"
        />
      </svg>
      <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-muted">
        {[0, 1, 2, 3].map((n) => (
          <span key={n} className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: riskColor(n / 3) }} />
            {n} vessel{n === 1 ? '' : 's'}
          </span>
        ))}
        <span>☆ this patient</span>
        <span className="ml-auto">
          {hp ? `#${hp.index}: true pattern ${hp.pattern} (click to load)` : `PCA, ${Math.round(ev * 100)}% of variance`}
        </span>
      </div>
      {loaded && (
        <p className="mt-1 text-[11px] text-text/90">
          Loaded training patient #{loaded.index}; true angiography pattern (LAD/LCX/RCA):{' '}
          <span className="font-mono">{loaded.pattern}</span>
        </p>
      )}
      <p className="mt-1 text-[10px] text-muted">
        Colours are the true number of diseased vessels at angiography (training split only; the locked test set is never shown).
      </p>
    </div>
  )
}
