import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import { useStore } from '../store'

const DEBOUNCE_MS = 150

/**
 * Re-runs analyze (fast) and then explain (slower) whenever the inputs or mode change.
 * Earlier in-flight requests are aborted so the UI never shows a stale answer.
 */
export function useLiveAnalysis() {
  const features = useStore((s) => s.features)
  const mode = useStore((s) => s.mode)
  const ready = useStore((s) => s.config !== null && Object.keys(s.features).length > 0)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const ctrl = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!ready) return
    const { setAnalysis, setExplanation, setLoading, setError } = useStore.getState()
    const timer = setTimeout(async () => {
      ctrl.current?.abort()
      const c = new AbortController()
      ctrl.current = c
      const t0 = performance.now()
      setLoading('analyze', true)
      try {
        const a = await api.analyze(features, mode, c.signal)
        setAnalysis(a)
        setLatencyMs(Math.round(performance.now() - t0))
        setError(null)
        setLoading('analyze', false)
        setLoading('explain', true)
        setExplanation(await api.explain(features, c.signal))
      } catch (e) {
        if ((e as Error).name !== 'AbortError') setError((e as Error).message)
      } finally {
        if (ctrl.current === c) {
          setLoading('analyze', false)
          setLoading('explain', false)
        }
      }
    }, DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [features, mode, ready])

  return latencyMs
}
