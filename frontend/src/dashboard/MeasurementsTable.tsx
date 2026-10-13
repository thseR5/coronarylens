import { useMemo, useState } from 'react'
import { useStore } from '../store'
import type { TargetId } from '../types/api'
import { TARGET_LABEL } from './labels'

type SortKey = 'impact' | 'name' | 'group'

export default function MeasurementsTable() {
  const config = useStore((s) => s.config)
  const features = useStore((s) => s.features)
  const explanation = useStore((s) => s.explanation)
  const selected = useStore((s) => s.selected)
  const [sort, setSort] = useState<SortKey>('impact')
  const target: TargetId = selected ?? 'CAD'
  const ex = explanation?.targets[target]

  const rows = useMemo(() => {
    const specs = config?.features.features ?? []
    const shapOf = new Map(ex?.contributions.map((c) => [c.feature, c.shap]))
    const list = specs.map((f) => {
      const v = features[f.id]
      const num = typeof v === 'number' ? v : null
      const flag = f.ref && num != null ? (num < f.ref[0] ? 'low' : num > f.ref[1] ? 'high' : null) : null
      return { f, v, flag, shap: shapOf.get(f.id) ?? 0 }
    })
    if (sort === 'impact') list.sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
    if (sort === 'name') list.sort((a, b) => a.f.label.localeCompare(b.f.label))
    if (sort === 'group') list.sort((a, b) => a.f.group.localeCompare(b.f.group))
    return list
  }, [config, features, ex, sort])

  const maxAbs = Math.max(0.01, ...rows.map((r) => Math.abs(r.shap)))

  return (
    <div className="card">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="h-section !mb-0">Measurements · contribution to {TARGET_LABEL[target]}</h2>
        <select aria-label="Sort measurements" value={sort} onChange={(e) => setSort(e.target.value as SortKey)} className="rounded bg-panel-2 px-1.5 py-0.5 text-[11px] ring-1 ring-line">
          <option value="impact">by impact</option>
          <option value="name">by name</option>
          <option value="group">by group</option>
        </select>
      </div>
      <div className="max-h-80 overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-panel text-[10px] uppercase tracking-wide text-muted">
            <tr>
              <th className="py-1 text-left font-medium">Measurement</th>
              <th className="text-right font-medium">Value</th>
              <th className="pl-2 text-left font-medium">Ref</th>
              <th className="w-36 text-left font-medium">Contribution (pts)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ f, v, flag, shap }) => (
              <tr key={f.id} className="border-t border-line/60">
                <td className="py-1 pr-1">{f.label}</td>
                <td className={`text-right font-mono ${flag ? 'text-warn' : ''}`}>
                  {f.type === 'binary' ? (Number(v) === 1 ? 'Yes' : 'No') : String(v ?? '–')}
                  {f.type === 'numeric' && <span className="ml-0.5 text-muted">{f.unit}</span>}
                  {flag && <span className="ml-1" aria-label={flag === 'high' ? 'above reference' : 'below reference'}>{flag === 'high' ? '▲' : '▼'}</span>}
                </td>
                <td className="pl-2 text-muted">{f.ref ? `${f.ref[0]}–${f.ref[1]}` : ''}</td>
                <td className="flex items-center gap-1">
                  <span className={`w-9 shrink-0 text-right font-mono text-[10px] ${shap >= 0 ? 'text-rose-300' : 'text-sky-300'}`}>
                    {shap >= 0 ? '+' : '−'}
                    {Math.abs(shap * 100).toFixed(1)}
                  </span>
                  <div className="relative h-3 flex-1" title={`${shap >= 0 ? '+' : ''}${(shap * 100).toFixed(2)} pts`}>
                    <div className="absolute inset-y-0 left-1/2 w-px bg-line" />
                    <div
                      className={`absolute inset-y-0.5 rounded-sm ${shap >= 0 ? 'bg-rose-400/80' : 'bg-sky-400/80'}`}
                      style={{
                        left: shap >= 0 ? '50%' : `${50 - (Math.abs(shap) / maxAbs) * 50}%`,
                        width: `${(Math.abs(shap) / maxAbs) * 50}%`,
                        transition: 'all 300ms ease',
                      }}
                    />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
