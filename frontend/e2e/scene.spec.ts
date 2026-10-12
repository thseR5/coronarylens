import { expect, test, type Page } from '@playwright/test'
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const IMG = resolve(HERE, '../../docs/img')
const OUT = resolve(HERE, '../test-results/e2e-3d')
// The 3D harness (scene + legend + bullseye, no dashboard). Set CL_E2E_PATH=/ to run against the full app.
const PATH = process.env.CL_E2E_PATH ?? '/e2e/harness.html'

async function measureFps(page: Page, ms: number) {
  return page.evaluate(async (dur) => {
    const w = window as unknown as { __coronaryLens: { frames: number } }
    const f0 = w.__coronaryLens.frames
    let raf = 0
    const t0 = performance.now()
    await new Promise<void>((done) => {
      const tick = () => {
        raf++
        if (performance.now() - t0 < dur) requestAnimationFrame(tick)
        else done()
      }
      requestAnimationFrame(tick)
    })
    const secs = (performance.now() - t0) / 1000
    return { rafFps: raf / secs, renderedFps: (w.__coronaryLens.frames - f0) / secs, seconds: secs }
  }, ms)
}

test('3D scene: loads glb, demo data, presets, selection, FPS', async ({ page }, info) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto(PATH)
  const scene = page.getByTestId('scene')
  await expect(scene.locator('canvas')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByTestId('renderer-info')).toContainText('BodyParts3D', { timeout: 30_000 })
  await expect(page.getByTestId('label-LAD')).toBeVisible()

  // backend absent -> demo toggle appears after 1 s
  await page.getByTestId('demo-toggle').click()
  await expect(page.getByTestId('label-LAD')).toContainText('%')
  await expect(page.getByTestId('label-LCX')).toContainText('uncertain')
  await expect(page.getByTestId('label-LeftMain')).toContainText('not modelled')
  await page.waitForTimeout(1200)

  const renderer = await page.evaluate(() => (window as unknown as { __coronaryLens: { renderer: string } }).__coronaryLens.renderer)
  const gpu = info.project.name === 'gpu'
  mkdirSync(IMG, { recursive: true })
  if (gpu) await page.screenshot({ path: `${IMG}/scene-default.png` })

  await page.getByTestId('preset-lao30').click()
  await page.waitForTimeout(1500)
  if (gpu) await scene.screenshot({ path: `${IMG}/scene-lao30.png` })

  // FPS over 5 s of auto-rotate (heartbeat + uncertain stripes also running)
  await page.getByTestId('toggle-rotate').click()
  await page.waitForTimeout(500)
  const fps = await measureFps(page, 5000)
  await page.getByTestId('toggle-rotate').click()

  // click the LAD in 3D -> select + fly-to
  await page.getByTestId('preset-ap').click()
  await page.getByTestId('toggle-heartbeat').click() // pause the beat so the vessel does not move under the cursor
  await page.waitForTimeout(1500)
  const pt = await page.evaluate(() => (window as unknown as { __coronaryLens: { project: (v: string) => { x: number; y: number } | null } }).__coronaryLens.project('LAD'))
  expect(pt).not.toBeNull()
  await page.mouse.move(pt!.x, pt!.y)
  await page.waitForTimeout(200)
  // 3D hover -> bullseye: LAD segments highlighted
  await expect(page.locator('[data-segment="1"] path')).toHaveAttribute('stroke', 'var(--color-accent)')
  await expect(page.locator('[data-segment="3"] path')).not.toHaveAttribute('stroke', 'var(--color-accent)')
  await page.mouse.click(pt!.x, pt!.y)
  await page.waitForTimeout(1500)
  await page.mouse.move(5, 5) // leave the vessel: the label must stay highlighted because it is selected
  await expect(page.getByTestId('label-LAD')).toHaveClass(/border-accent/)
  if (gpu) await scene.screenshot({ path: `${IMG}/scene-selected-lad.png` })

  await page.getByTestId('toggle-heartbeat').click()
  // heartbeat: the LAD mesh scale changes over time (scene is beating while idle)
  const scales = await page.evaluate(async () => {
    const w = window as unknown as { __coronaryLens: { meshScale: (v: string) => number } }
    const out: number[] = []
    for (let i = 0; i < 12; i++) {
      out.push(w.__coronaryLens.meshScale('LAD'))
      await new Promise((r) => setTimeout(r, 70))
    }
    return out
  })
  expect(Math.max(...scales) - Math.min(...scales)).toBeGreaterThan(0)

  // bullseye hover -> 3D: hovering an RCA segment highlights the RCA label
  await page.mouse.move(5, 5)
  await page.locator('[data-segment="4"]').hover()
  await expect(page.getByTestId('label-RCA')).toHaveClass(/border-accent/)
  await page.mouse.move(5, 5)

  // x-ray + territories view for the docs
  await page.getByTestId('toggle-xray').click()
  await page.waitForTimeout(600)
  if (gpu) await scene.screenshot({ path: `${IMG}/scene-xray.png` })

  const result = { project: info.project.name, renderer, ...fps, errors }
  mkdirSync(OUT, { recursive: true })
  writeFileSync(`${OUT}/fps-${info.project.name}.json`, JSON.stringify(result, null, 1))
  console.log('FPS', JSON.stringify(result))
  expect(errors).toEqual([])
})

test('full app mounts the 3D scene', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto('/')
  await expect(page.getByTestId('scene').locator('canvas')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByTestId('renderer-info')).toContainText('BodyParts3D', { timeout: 30_000 })
  expect(errors).toEqual([])
})

test('procedural fallback renders the same nodes', async ({ page }, info) => {
  await page.goto(`${PATH}?renderer=procedural`)
  await expect(page.getByTestId('renderer-info')).toContainText('Procedural', { timeout: 30_000 })
  await page.getByTestId('demo-toggle').click()
  for (const v of ['LAD', 'LCX', 'RCA', 'LeftMain']) await expect(page.getByTestId(`label-${v}`)).toBeVisible()
  await page.waitForTimeout(1200)
  if (info.project.name === 'gpu') await page.getByTestId('scene').screenshot({ path: `${IMG}/scene-procedural.png` })
})
