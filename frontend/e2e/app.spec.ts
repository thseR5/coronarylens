import { expect, test } from '@playwright/test'

const pctText = (p: number) => `${Math.round(p * 100)}%`

test('full stack: preset → API → dashboard + 3D agree; slider updates live', async ({ page, request }) => {
  const errors: string[] = []
  const reactErrors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  page.on('console', (m) => {
    if (m.type() === 'error' && /unmount|Cannot update a component|Maximum update depth/i.test(m.text())) reactErrors.push(m.text())
  })
  await page.goto('/')
  // lands on a real preset patient, not on mode-imputed defaults
  await expect(page.getByLabel('Load preset patient')).toHaveValue('lad-only', { timeout: 15_000 })
  // Reset sends no inputs: the server imputes them and every status becomes 'uncertain' with a visible warning
  await page.getByRole('button', { name: 'Reset' }).click()
  await expect(page.getByTestId('imputed-warning')).toBeVisible({ timeout: 10_000 })
  await expect(page.getByTestId('localisation-badge')).toBeVisible()
  await expect(page.getByTestId('localisation-badge')).toContainText(
    'Vessel-level estimates are less certain than overall CAD (nested-CV AUC: CAD 0.93 · LAD 0.87 · LCX 0.71 · RCA 0.74)',
  )

  // Disclaimer is visible and non-dismissible.
  await expect(page.getByRole('alert').first()).toContainText('Not a substitute for formal diagnostic imaging')
  await expect(page.locator('canvas')).toBeVisible({ timeout: 30_000 })

  // Load the 3-vessel preset and compare the dashboard with a direct API call on the same features.
  const presets = await (await request.get('http://127.0.0.1:8000/api/presets')).json()
  const p3 = presets.find((p: { true_pattern: string }) => p.true_pattern === '111') ?? presets[0]
  await page.getByLabel('Load preset patient').selectOption(p3.id)
  const direct = await (
    await request.post('http://127.0.0.1:8000/api/analyze', { data: { features: p3.features, mode: 'f1' } })
  ).json()
  for (const v of ['LAD', 'LCX', 'RCA']) {
    await expect(page.getByRole('button', { name: new RegExp(`^${v}`) }).first()).toContainText(pctText(direct.vessels[v].p))
  }
  expect(direct.cad.p).toBeGreaterThanOrEqual(Math.max(...Object.values(direct.vessels).map((v: any) => v.p)))
  await expect(page.getByRole('list', { name: 'Feature contributions' })).toBeVisible({ timeout: 10_000 })
  await page.waitForTimeout(1200) // let the 400 ms colour transition finish
  await page.screenshot({ path: '../docs/img/app-3vd-preset.png' })

  // AHA segments are the default territories; hovering a 3D segment highlights the same bullseye segment.
  const ahaBtn = page.getByTestId('toggle-territory-mode')
  await expect(ahaBtn).toHaveText('AHA 17-segment')
  // pause the heartbeat so the mesh does not move between projecting a point and hovering it
  const beat = page.getByTestId('toggle-heartbeat')
  const beating = (await beat.getAttribute('aria-pressed')) === 'true'
  if (beating) await beat.click()
  await page.waitForTimeout(300)
  let hit: { seg: number; x: number; y: number } | null = null
  for (const seg of [7, 1, 13, 12, 6, 2, 8, 17]) {
    const p = await page.evaluate((n) => window.__coronaryLens?.project(n) ?? null, `AHA_${String(seg).padStart(2, '0')}`)
    if (p) {
      hit = { seg, ...p }
      break
    }
  }
  expect(hit, 'some AHA segment is visible from the default view').not.toBeNull()
  await page.mouse.move(hit!.x, hit!.y)
  await expect(page.locator(`[data-testid="bullseye"] [data-segment="${hit!.seg}"]`)).toHaveAttribute('data-hot', 'segment')
  await page.screenshot({ path: '../docs/img/app-aha-segments.png' })
  await page.mouse.move(5, 5)
  if (beating) await beat.click()
  await ahaBtn.click()
  await expect(ahaBtn).toHaveText('nearest-artery estimate (14/17 agree with AHA)')
  await page.waitForTimeout(600)
  await page.screenshot({ path: '../docs/img/app-voronoi-territories.png' })
  await ahaBtn.click()

  // Vessel card click switches the explanation target.
  await page.getByRole('button', { name: /^RCA/ }).first().click()
  await expect(page.getByRole('tab', { name: 'RCA', selected: true })).toBeVisible()
  await page.screenshot({ path: '../docs/img/app-rca-selected.png' })

  // Live update: change EF and wait until the app marks the new analysis as painted (probabilities + colours).
  // (The rounded CAD gauge can stay at the same whole percentage for a saturated patient, so it is not the signal.)
  const ef = page.locator('#feat-ef_tte')
  const nBefore = await page.evaluate(() => performance.getEntriesByType('mark').filter((m) => m.name.startsWith('cl:analyze:')).length)
  const t0 = Date.now()
  await ef.fill('20')
  await page.waitForFunction(
    (n) => performance.getEntriesByType('mark').filter((m) => m.name.startsWith('cl:analyze:')).length > n,
    nBefore,
    { timeout: 5_000 },
  )
  const ms = Date.now() - t0
  await expect(page.getByTestId('preset-modified')).toBeVisible() // true pattern no longer shown as applying
  // Smoke check only: this runs on the Vite DEV server (unbundled modules, HMR). Latency gates live in
  // e2e/latency.spec.ts on the production build (colours <= 300 ms, SHAP <= 700 ms, p95).
  console.log(`slider→gauge update (dev server, smoke only): ${ms} ms`)
  if (ms > 1000) console.warn(`WARNING: dev-mode update took ${ms} ms; check the production latency test`)
  expect(ms).toBeLessThan(5000)

  // Cohort map: training patients only; clicking a dot loads that patient (true pattern shown).
  const map = page.getByRole('img', { name: 'Cohort map of training patients' })
  await map.scrollIntoViewIfNeeded()
  await expect(map).toBeVisible()
  const dot = map.locator('circle').first()
  await dot.click()
  await expect(page.getByText(/Loaded training patient #\d+; true angiography pattern/)).toBeVisible()
  await page.getByRole('complementary', { name: 'Risk dashboard' }).screenshot({ path: '../docs/img/app-cohort-map.png' })

  // Clipping plane reveals the LV territories; vessels stay visible.
  await page.getByTestId('toggle-clip').click()
  await page.getByTestId('clip-depth').fill('0.1')
  await page.waitForTimeout(500)
  await page.locator('section[aria-label="3D heart"]').screenshot({ path: '../docs/img/app-clip-plane.png' })
  await page.getByTestId('toggle-clip').click()

  // Printable report (Report / PDF button) carries the disclaimer.
  const rep = await request.post('http://127.0.0.1:8000/api/report', { data: { features: p3.features } })
  expect(await rep.text()).toContain('Not a substitute for formal diagnostic imaging')
  await expect(page.getByRole('button', { name: 'Report / PDF' })).toBeEnabled()

  // Model performance tab renders real metrics.
  await page.getByRole('tab', { name: 'Model performance' }).click()
  await expect(page.getByText(/^Headline · nested CV/)).toBeVisible()
  await expect(page.getByText(/Secondary · locked test set \(n = 61\)/)).toBeVisible()
  await expect(page.getByText('Leakage ablation', { exact: false })).toBeVisible()
  await page.screenshot({ path: '../docs/img/app-model-tab.png', fullPage: false })
  await page.getByRole('complementary', { name: 'Risk dashboard' }).screenshot({ path: '../docs/img/app-model-tab-panel.png' })

  expect(errors).toEqual([])
  expect(reactErrors).toEqual([])
})
