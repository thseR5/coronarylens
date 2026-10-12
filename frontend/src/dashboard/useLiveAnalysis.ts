import { startTransition, useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import { useStore } from '../store'

const DEBOUNCE_MS = 100

/** Resolves after the browser has painted the current React commit (two animation frames). */
const afterPaint = () => new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())))

// Performance marks (read by e2e/latency.spec.ts):
//   cl:input:<n>      the inputs changed (store update)
//   cl:analyze:<n>    probabilities + vessel colours for that input are on screen
//   cl:explain:<n>    the SHAP panel for that input is on screen
let inputSeq = 0
useStore.subscribe((s, prev) => {
  if (s.features !== prev.features || s.mode !== prev.mode) performance.mark(`cl:input:${++inputSeq}`)
})

/**
 * On every input or mode change: /analyze first (fast path: probabilities + 3D colours, applied as soon as it
 * returns), then /explain (slow path: SHAP). A new change aborts both in-flight requests, so the UI never shows a
 * stale answer, and the server drops this tab's queued explain jobs. The SHAP update is a React transition so it
 * never blocks typing or dragging.
 */
export function useLiveAnalysis() {
  const features = useStore((s) => s.features)
  const mode = useStore((s) => s.mode)
  const ready = useStore((s) => s.config !== null && Object.keys(s.features).length > 0)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const ctrl = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!ready) return
    const seq = inputSeq
    const { setAnalysis, setExplanation, setLoading, setError } = useStore.getState()
    ctrl.current?.abort() // a new change cancels in-flight analyze/explain immediately, not after the debounce
    const timer = setTimeout(async () => {
      const c = new AbortController()
      ctrl.current = c
      const t0 = performance.now()
      setLoading('analyze', true)
      try {
        const a = await api.analyze(features, mode, c.signal)
        setAnalysis(a)
        setError(null)
        setLoading('analyze', false)
        setLoading('explain', true)
        // request the explanation as soon as analyze returns (the server already started computing it), while
        // the new probabilities and colours paint
        const explainP = api.explain(features, c.signal)
        explainP.catch(() => {}) // awaited below; avoids an unhandled rejection if we return early
        await afterPaint()
        if (c.signal.aborted) return
        performance.mark(`cl:analyze:${seq}`)
        setLatencyMs(Math.round(performance.now() - t0))
        const e = await explainP
        if (c.signal.aborted) return
        performance.mark(`cl:explainResp:${seq}`)
        startTransition(() => setExplanation(e))
        await afterPaint()
        if (!c.signal.aborted) performance.mark(`cl:explain:${seq}`)
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
