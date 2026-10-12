import { mkdirSync, writeFileSync } from 'node:fs'
import { expect, test } from '@playwright/test'

const N = Number(process.env.CL_LATENCY_N ?? 40)
const pct = (xs: number[], q: number) => {
  const s = [...xs].sort((a, b) => a - b)
  return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1) + 0.5))]
}

test('live update latency (production build): analyze path and explain path', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('img', { name: /CAD probability/ })).toBeVisible({ timeout: 60_000 })
  await expect(page.getByRole('list', { name: 'Feature contributions' })).toBeVisible({ timeout: 30_000 })

  const bp = page.locator('#feat-bp')
  const analyze: number[] = []
  const explain: number[] = []
  const explainResp: number[] = [] // explain response received (before the SHAP panel renders)
  for (let i = 0; i < N + 3; i++) {
    const seq = await page.evaluate(() => performance.getEntriesByType('mark').filter((m) => m.name.startsWith('cl:input:')).length)
    // unique value each time so the server-side cache never hits
    await bp.fill(String(95 + ((i * 37) % 95)))
    const res = await page.waitForFunction(
      (n) => {
        const marks = performance.getEntriesByType('mark')
        const input = marks.filter((m) => m.name.startsWith('cl:input:'))[n]
        if (!input) return null
        const id = input.name.split(':')[2]
        const a = marks.find((m) => m.name === `cl:analyze:${id}`)
        const e = marks.find((m) => m.name === `cl:explain:${id}`)
        const r = marks.find((m) => m.name === `cl:explainResp:${id}`)
        return a && e && r
          ? { a: a.startTime - input.startTime, e: e.startTime - input.startTime, r: r.startTime - input.startTime }
          : null
      },
      seq,
      { timeout: 20_000, polling: 'raf' },
    )
    const { a, e, r } = (await res.jsonValue()) as { a: number; e: number; r: number }
    if (i >= 3) {
      analyze.push(a) // first iterations warm up the JIT / connections
      explain.push(e)
      explainResp.push(r)
    }
    await page.waitForTimeout(250) // idle gap: measure one edit at a time
  }
  const out = {
    build: 'vite production build + preview',
    n: analyze.length,
    debounce_ms: 100,
    analyze_path_ms: { p50: Math.round(pct(analyze, 0.5)), p95: Math.round(pct(analyze, 0.95)) },
    explain_path_ms: { p50: Math.round(pct(explain, 0.5)), p95: Math.round(pct(explain, 0.95)) },
    explain_response_ms: { p50: Math.round(pct(explainResp, 0.5)), p95: Math.round(pct(explainResp, 0.95)) },
    note: 'input change -> painted probabilities + vessel colour targets (analyze) / painted SHAP panel (explain); includes the debounce',
  }
  console.log(JSON.stringify(out))
  const tag = process.env.CL_REDUCED_MOTION ? '-reduced-motion' : ''
  // single run -> gitignored file; docs/latency.json holds the curated multi-run record
  mkdirSync('../test-results', { recursive: true })
  writeFileSync(`../test-results/latency-last-run${tag}.json`, JSON.stringify(out, null, 1) + '\n')
  expect(out.analyze_path_ms.p95).toBeLessThanOrEqual(300)
  expect(out.explain_path_ms.p95).toBeLessThanOrEqual(700)
})
