import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { useStore } from '../store'
import type { SimilarResponse } from '../types/api'

function PatternDots({ pattern, vessels }: { pattern: string; vessels: string[] }) {
  return (
    <span className="flex gap-1">
      {vessels.map((v, i) => (
        <span
          key={v}
          title={`${v}: ${pattern[i] === '1' ? 'stenotic' : 'normal'}`}
          className={`rounded px-1 font-mono text-[10px] ring-1 ${pattern[i] === '1' ? 'bg-rose-500/20 text-rose-200 ring-rose-400/50' : 'text-muted ring-line'}`}
        >
          {v}
        </span>
      ))}
    </span>
  )
}

export default function SimilarPatients() {
  const config = useStore((s) => s.config)
  const features = useStore((s) => s.features)
  const analysis = useStore((s) => s.analysis)
  const [data, setData] = useState<SimilarResponse | null>(null)
  const vessels = config?.targets.vessels.map((v) => v.id) ?? []

  useEffect(() => {
    if (!analysis) return
    let live = true
    api.similar(features, 5).then((d) => live && setData(d)).catch(() => live && setData(null))
    return () => {
      live = false
    }
  }, [analysis, features])

  return (
    <div className="card">
      <h2 className="h-section">Most similar training patients · actual angiography</h2>
      {!data ? (
        <p className="text-xs text-muted">Unavailable.</p>
      ) : (
        <ul className="flex flex-col gap-1 text-xs">
          {data.neighbors.map((n) => (
            <li key={n.index} className="flex items-center justify-between gap-2 rounded bg-ink/40 px-2 py-1">
              <span className="text-muted">
                #{n.index} · {String(n.features.age ?? '?')} y · {Number(n.features.sex_male) === 1 ? 'M' : 'F'}
              </span>
              <PatternDots pattern={n.pattern} vessels={vessels} />
              <span className={n.cad ? 'text-rose-300' : 'text-emerald-300'}>{n.cad ? 'CAD' : 'Normal'}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
