import { useStore } from '../store'
import { pct, signedPct } from '../lib/risk'
import { riskColor } from '../lib/color'
import type { TargetId } from '../types/api'
import { useWhatIf } from './whatIfStore'
import StatusChip from './StatusChip'

export default function VesselCards() {
  const config = useStore((s) => s.config)
  const a = useStore((s) => s.analysis)
  const selected = useStore((s) => s.selected)
  const hovered = useStore((s) => s.hovered)
  const select = useStore((s) => s.select)
  const hover = useStore((s) => s.hover)
  const baseline = useWhatIf((s) => s.baseline)
  const vessels = config?.targets.vessels ?? []
  const cad = a?.cad.p ?? null

  return (
    <div className="flex flex-col gap-1">
    <div className="flex items-center gap-1.5 text-[10px] text-warn" data-testid="localisation-badge">
      <span aria-hidden>⚠</span> Artery ranking is not reliable: the model estimates risk per artery, not where the lesion is.
    </div>
    <div className="grid grid-cols-3 gap-2">
      {vessels.map((v) => {
        const r = a?.vessels[v.id]
        const active = selected === v.id
        const delta = r && baseline ? r.p - baseline.analysis.vessels[v.id].p : null
        return (
          <button
            key={v.id}
            type="button"
            aria-pressed={active}
            onClick={() => select(active ? null : (v.id as TargetId))}
            onMouseEnter={() => hover(v.id)}
            onMouseLeave={() => hover(null)}
            onFocus={() => hover(v.id)}
            onBlur={() => hover(null)}
            title={v.label}
            className={`flex flex-col gap-1 rounded-lg bg-panel-2/60 p-2.5 text-left ring-1 transition ${
              active ? 'ring-accent' : hovered === v.id ? 'ring-accent/50' : 'ring-line'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="font-semibold">{v.id}</span>
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: r ? riskColor(r.p) : 'var(--color-line)' }} />
            </div>
            <span className="font-mono text-xl">{pct(r?.p)}</span>
            {r ? (
              <>
                <span className="text-[10px] text-muted">
                  {pct(r.interval[0])}–{pct(r.interval[1])} · thr {pct(r.threshold)}
                </span>
                <span className={`text-[11px] font-medium ${r.positive ? 'text-rose-300' : 'text-emerald-300'}`}>
                  {r.positive ? 'Above threshold' : 'Below threshold'}
                </span>
                <StatusChip status={r.status} />
                {cad != null && cad > 0 && (
                  <span className="text-[10px] text-muted" title="Share of the CAD probability attributed to this artery">
                    if CAD: {pct(Math.min(1, r.p / cad))}
                  </span>
                )}
                {delta != null && (
                  <span className={`font-mono text-[10px] ${delta > 0 ? 'text-rose-300' : 'text-emerald-300'}`}>{signedPct(delta)}</span>
                )}
              </>
            ) : (
              <span className="h-8 animate-pulse rounded bg-line/50" />
            )}
          </button>
        )
      })}
    </div>
    <p className="text-[10px] leading-snug text-muted">
      Artery estimates mostly follow overall CAD risk. The model cannot reliably tell which artery is affected (among patients
      with CAD: LAD modestly, LCX/RCA barely better than chance). Use them as risk levels, not lesion locations.
    </p>
    </div>
  )
}
