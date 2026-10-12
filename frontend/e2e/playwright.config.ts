import { defineConfig } from '@playwright/test'

// 3D checks. Run from frontend/:  npx playwright test -c e2e/playwright.config.ts [--project gpu|software]
// Serves a production build (app + e2e/harness.html) with `vite preview`; no backend needed
// (bundled config + demo data).
const PORT = 4179

export default defineConfig({
  testDir: '.',
  testIgnore: 'app.spec.ts', // full-stack test has its own config (app.config.ts)
  timeout: 120_000,
  workers: 1,
  reporter: [['list']],
  outputDir: '../test-results/e2e-3d',
  use: { baseURL: `http://127.0.0.1:${PORT}`, viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 },
  webServer: {
    command: `npx vite build -c e2e/vite.harness.config.ts --logLevel error && npx vite preview -c e2e/vite.harness.config.ts --port ${PORT} --strictPort --host 127.0.0.1`,
    cwd: '..',
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: true,
    timeout: 180_000,
  },
  projects: [
    // channel 'chromium' = full Chromium in new headless mode, which uses the real GPU (ANGLE Metal on macOS);
    // the default headless shell silently falls back to SwiftShader.
    { name: 'gpu', use: { browserName: 'chromium', channel: 'chromium' } },
    { name: 'software', use: { browserName: 'chromium', channel: 'chromium', launchOptions: { args: ['--disable-gpu'] } } },
  ],
})
