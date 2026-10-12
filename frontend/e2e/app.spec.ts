import { expect, test } from '@playwright/test'

const pctText = (p: number) => `${Math.round(p * 100)}%`

test('full stack: preset → API → dashboard + 3D agree; slider updates live', async ({ page, request }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto('/')

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

  // Vessel card click switches the explanation target.
  await page.getByRole('button', { name: /^RCA/ }).first().click()
  await expect(page.getByRole('tab', { name: 'RCA', selected: true })).toBeVisible()
  await page.screenshot({ path: '../docs/img/app-rca-selected.png' })

  // Live update latency: change EF and wait for the CAD gauge to change.
  const gauge = page.getByRole('img', { name: /CAD probability/ })
  const before = await gauge.getAttribute('aria-label')
  const ef = page.locator('#feat-ef_tte')
  const t0 = Date.now()
  await ef.fill('20')
  await expect(gauge).not.toHaveAttribute('aria-label', before!, { timeout: 5_000 })
  const ms = Date.now() - t0
  console.log(`slider→gauge update: ${ms} ms (includes 150 ms debounce)`)
  expect(ms).toBeLessThan(1000)

  // Model performance tab renders real metrics.
  await page.getByRole('tab', { name: 'Model performance' }).click()
  await expect(page.getByText('Locked test set · served model')).toBeVisible()
  await expect(page.getByText('Leakage ablation', { exact: false })).toBeVisible()
  await page.screenshot({ path: '../docs/img/app-model-tab.png', fullPage: false })
  await page.getByRole('complementary', { name: 'Risk dashboard' }).screenshot({ path: '../docs/img/app-model-tab-panel.png' })

  expect(errors).toEqual([])
})
