import { useEffect, useState } from 'react'
import { api } from '../api/client'

// Filled in once ml/artifacts/metrics.json has its final shape (Phase 1 output).
export default function ModelPerformance() {
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null)
  const [err, setErr] = useState(false)
  useEffect(() => {
    api.metrics().then(setMetrics).catch(() => setErr(true))
  }, [])
  if (err) return <p className="text-xs text-muted">Metrics unavailable (backend offline).</p>
  if (!metrics) return <div className="card h-40 animate-pulse" />
  return <pre className="overflow-x-auto text-[10px] text-muted">{JSON.stringify(metrics, null, 1).slice(0, 4000)}</pre>
}
