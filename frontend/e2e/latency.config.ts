import { defineConfig } from '@playwright/test'

// Live-update latency on a PRODUCTION build (vite build + preview) against the real API.
// Run from frontend/:  npx playwright test -c e2e/latency.config.ts   (run on an idle machine)
export default defineConfig({
  testDir: '.',
  testMatch: 'latency.spec.ts',
  timeout: 300_000,
  workers: 1,
  reporter: [['list']],
  outputDir: '../test-results/e2e-latency',
  use: {
    baseURL: 'http://127.0.0.1:4180',
    viewport: { width: 1600, height: 1000 },
    browserName: 'chromium',
    channel: 'chromium',
    // CL_REDUCED_MOTION=1 measures without the continuous heartbeat/stripe animation (diagnostics only)
    reducedMotion: process.env.CL_REDUCED_MOTION ? 'reduce' : 'no-preference',
  },
  webServer: [
    {
      command: 'uv run uvicorn backend.app.main:app --port 8000 --host 127.0.0.1',
      cwd: '../..',
      url: 'http://127.0.0.1:8000/api/health',
      reuseExistingServer: true,
      timeout: 120_000,
    },
    {
      command: 'npx vite build --logLevel error && npx vite preview --port 4180 --strictPort --host 127.0.0.1',
      cwd: '..',
      url: 'http://127.0.0.1:4180',
      reuseExistingServer: false,
      timeout: 180_000,
    },
  ],
})
