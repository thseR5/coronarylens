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

  // Sex is shown as Male / Female (model input stays sex_male 1/0)
  await page.getByLabel('Load preset patient').selectOption('lad-only')
  const sex = page.locator('#feat-sex_male')
  await expect(sex).toContainText(/Male|Female/)
  const before = (await sex.innerText()).includes('Female') ? 'Female' : 'Male'
  await sex.click()
  await expect(sex).toContainText(before === 'Male' ? 'Female' : 'Male')
  await sex.click()
  await expect(sex).toContainText(before)

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
  // coherence chart used as the video's opening shot: values printed on the bars
  await expect(page.getByTestId('coherence-chart')).toContainText('32.1%')
  await expect(page.getByTestId('coherence-chart')).toContainText('5.4%')
  await expect(page.getByTestId('coherence-chart')).toContainText('Served procedure')
  // the localisation finding with corrected CIs, read from metrics.json
  await expect(page.getByTestId('within-cad-table')).toContainText('0.69 [0.59–0.78]')
  await expect(page.getByTestId('within-cad-table')).toContainText('0.52 [0.42–0.61]')
  await expect(page.getByTestId('within-cad-table')).toContainText('0.57 [0.47–0.66]')
  await page.screenshot({ path: '../docs/img/app-model-tab.png', fullPage: false })
  await page.getByRole('complementary', { name: 'Risk dashboard' }).screenshot({ path: '../docs/img/app-model-tab-panel.png' })

  expect(errors).toEqual([])
  expect(reactErrors).toEqual([])
})

test('vessel labels never sit under the legend (1280×800, 1440×900, 1920×1080; all camera presets)', async ({ page }) => {
  for (const [width, height] of [
    [1280, 800],
    [1440, 900],
    [1920, 1080],
  ]) {
    await page.setViewportSize({ width, height })
    await page.goto('/')
    await expect(page.getByLabel('Load preset patient')).toHaveValue('lad-only', { timeout: 15_000 })
    for (const preset of ['three-vessel', 'rca-only']) {
      await page.getByLabel('Load preset patient').selectOption(preset)
      await expect(page.getByTestId('label-RCA')).toContainText('%', { timeout: 15_000 })
      for (const cam of ['AP', 'LAO 30°', 'RAO 30°', 'LAO 30° cranial 20°', 'RAO 30° caudal 20°']) {
        await page.getByRole('button', { name: cam, exact: true }).click()
        await page.waitForTimeout(900) // camera transition
        const overlapping = await page.evaluate(() => {
          const box = (el: Element) => el.getBoundingClientRect()
          const legend = box(document.querySelector('[data-testid=legend]')!)
          return [...document.querySelectorAll('[data-testid^=label-]')]
            .filter((el) => {
              const b = box(el)
              return b.x < legend.right && b.right > legend.x && b.y < legend.bottom && b.bottom > legend.y
            })
            .map((el) => (el as HTMLElement).dataset.testid)
        })
        expect(overlapping, `${width}×${height} ${preset} ${cam}`).toEqual([])
      }
    }
  }
})

test('honesty cues: inline status reasons, diffuse-signal note, low-confidence blank patient', async ({ page, request }) => {
  await page.goto('/')
  await expect(page.getByLabel('Load preset patient')).toHaveValue('lad-only', { timeout: 15_000 })
  // full input: shown normally, every chip carries its reason inline, one legend line explains per-artery calibration
  await expect(page.getByTestId('cad-gauge')).toHaveAttribute('data-confidence', 'full')
  await expect(page.getByTestId('vessel-cards')).toHaveAttribute('data-fade', '0.00')
  await expect(page.getByTestId('status-reason')).toHaveCount(4) // CAD + 3 vessels
  await expect(page.getByTestId('vessel-cards')).toContainText('both outcomes plausible at 90% confidence')
  await expect(page.getByTestId('status-legend')).toContainText('so a higher probability can still be "uncertain"')

  // diffuse note appears exactly when the summed "other features" bar beats every single feature (checked via API)
  const presets = await (await request.get('http://127.0.0.1:8000/api/presets')).json()
  const lad = presets.find((p: { id: string }) => p.id === 'lad-only')
  const ex = await (await request.post('http://127.0.0.1:8000/api/explain', { data: { features: lad.features } })).json()
  let anyDiffuse = false
  for (const v of ['LCX', 'RCA', 'LAD']) {
    const c = [...ex.targets[v].contributions].sort((a: { shap: number }, b: { shap: number }) => Math.abs(b.shap) - Math.abs(a.shap))
    const rest = c.slice(8).reduce((s: number, x: { shap: number }) => s + x.shap, 0)
    const diffuse = Math.abs(rest) > Math.abs(c[0].shap)
    anyDiffuse ||= diffuse
    await page.getByRole('tab', { name: v, exact: true }).click()
    await expect(page.getByRole('listitem').filter({ hasText: 'other features' })).toBeVisible()
    await expect(page.getByTestId('diffuse-note')).toHaveCount(diffuse ? 1 : 0)
    if (diffuse)
      await expect(page.getByTestId('diffuse-note')).toHaveText(
        'Signal for this artery is spread thinly across many features. No single factor dominates.',
      )
  }
  expect(anyDiffuse).toBe(true) // LCX in this preset (measured)

  // blank patient: greyed + hatched gauge, faded vessels, explicit text, every status uncertain with its reason
  await page.getByRole('button', { name: 'Reset' }).click()
  await expect(page.getByTestId('cad-gauge')).toHaveAttribute('data-confidence', 'low', { timeout: 10_000 })
  await expect(page.getByTestId('low-confidence-label')).toHaveText('Estimate based mostly on cohort averages')
  await expect(page.getByTestId('vessel-cards')).toHaveAttribute('data-fade', '1.00')
  await expect(page.getByTestId('imputed-warning')).toContainText('100% of the model')
  await expect(page.getByTestId('vessel-cards')).toContainText('too much of the input is missing')
  await page.screenshot({ path: '../docs/img/app-low-confidence.png' })
})

// Walks the exact click path of the demo-video script and checks every number the narration reads out.
for (const [width, height] of [
  [1920, 1080],
  [1440, 900],
]) {
  test(`demo script click path works at ${width}×${height}`, async ({ page, context }) => {
    await page.setViewportSize({ width, height })
    await page.goto('/')
    const preset = page.getByLabel('Load preset patient')
    await expect(preset).toHaveValue('lad-only', { timeout: 15_000 })
    const tab = (name: string) => page.getByRole('tab', { name, exact: true })

    // 0:00 hook
    await tab('Model performance').click()
    const coh = page.getByTestId('coherence-chart')
    await coh.scrollIntoViewIfNeeded()
    for (const t of ['32.1%', '5.4%', 'Served procedure']) await expect(coh).toContainText(t)

    // 0:20 + 0:40 anatomy
    await tab('Patient analysis').click()
    await expect(page.getByRole('alert').first()).toContainText('Decision support and educational use only')
    for (const cam of ['AP', 'LAO 30°', 'RAO 30° caudal 20°']) await page.getByRole('button', { name: cam, exact: true }).click()
    await page.getByRole('button', { name: 'Ribs', exact: true }).click()
    await page.getByRole('tab', { name: /^Full \(54\)/ }).click()
    await expect(page.locator('#feat-pr')).toHaveValue('100')
    await page.getByRole('tab', { name: /^Quick \(13\)/ }).click()
    await page.getByRole('button', { name: 'Ribs', exact: true }).click()

    // 1:40 three-vessel
    await preset.selectOption('three-vessel')
    await expect(page.getByTestId('cad-gauge')).toContainText('97%', { timeout: 10_000 })
    const cards = page.getByTestId('vessel-cards')
    for (const t of ['78%', '57%', '53%']) await expect(cards).toContainText(t)
    await page.locator('[data-testid="bullseye"] [data-segment="7"]').hover()
    await page.getByRole('button', { name: /^LAD/ }).first().click()
    await expect(tab('LAD')).toHaveAttribute('aria-selected', 'true')
    await expect(page.getByText(/^Measurements · contribution to/)).toBeVisible()

    // 2:50 what-if, counterfactual, similar patients
    await preset.selectOption('uncertain')
    await expect(page.getByTestId('cad-gauge')).toContainText('78%', { timeout: 10_000 })
    await tab('CAD').click()
    await page.getByRole('button', { name: 'Pin current as baseline' }).click()
    await expect(page.locator('#feat-age')).toHaveValue('70')
    await page.locator('#feat-age').fill('45')
    await expect(page.getByText(/vs pinned baseline/)).toBeVisible({ timeout: 10_000 })
    await page.getByRole('button', { name: 'Restore baseline' }).click()
    await expect(page.locator('#feat-age')).toHaveValue('70')
    await page.getByRole('button', { name: 'Suggest' }).click()
    const cf = page.getByText(/Triglycerides/).first()
    await expect(cf).toBeVisible({ timeout: 20_000 })
    await expect(page.getByText(/78%.*74%|78% → 74%/).first()).toBeVisible()
    await expect(page.getByText('Most similar training patients · actual angiography')).toBeVisible()

    // 3:50 uncertainty, OOD, blank patient
    await expect(cards).toContainText('both outcomes plausible at 90% confidence')
    await preset.selectOption('synthetic-ood')
    await expect(page.getByText('Outside training distribution: interpret with extra caution.')).toBeVisible({ timeout: 10_000 })
    await expect(cards).toContainText('patient unlike the training data')
    await page.getByRole('button', { name: 'Reset' }).click()
    await expect(page.getByTestId('low-confidence-label')).toHaveText('Estimate based mostly on cohort averages', { timeout: 10_000 })

    // 4:40 validation + 5:30 finding
    await tab('Model performance').click()
    for (const t of [/^Headline · nested CV/, /^Model search · 748 procedures/, /^Served procedure vs independent ensemble/, /^Leakage ablation/, /^Finding · which artery\?/])
      await expect(page.getByText(t)).toBeVisible()
    await expect(page.getByTestId('within-cad-table')).toContainText('0.52 [0.42–0.61]')

    // 5:50 montage
    await tab('Patient analysis').click()
    await preset.selectOption('three-vessel')
    await expect(page.getByTestId('cad-gauge')).toContainText('97%', { timeout: 10_000 })
    const dot = page.locator('[data-testid^="cohort-"]').first()
    await dot.scrollIntoViewIfNeeded()
    await dot.click()
    await expect(preset).toHaveValue('')
    await page.getByTestId('toggle-clip').click()
    await page.getByTestId('toggle-clip').click()
    await page.getByRole('button', { name: 'Patient view' }).click()
    await page.evaluate(() => (window.print = () => {}))
    const [report] = await Promise.all([context.waitForEvent('page'), page.getByRole('button', { name: 'Report / PDF' }).click()])
    await expect(report.locator('body')).toContainText('Decision support and educational use only', { timeout: 15_000 })
    await report.close()

    // 6:40 limitations
    await tab('About & limits').click()
    await expect(page.getByText('Limitations', { exact: true })).toBeVisible()
  })
}
