import { defineConfig } from '@playwright/test'

// Full-stack smoke test: real FastAPI backend + Vite dev server (with /api proxy).
// Run from frontend/:  npx playwright test -c e2e/app.config.ts
export default defineConfig({
  testDir: '.',
  testMatch: 'app.spec.ts',
  timeout: 120_000,
  workers: 1,
  reporter: [['list']],
  outputDir: '../test-results/e2e-app',
  use: { baseURL: 'http://127.0.0.1:5173', viewport: { width: 1600, height: 1000 }, browserName: 'chromium', channel: 'chromium' },
  webServer: [
    {
      command: 'uv run uvicorn backend.app.main:app --port 8000 --host 127.0.0.1',
      cwd: '../..',
      url: 'http://127.0.0.1:8000/api/health',
      reuseExistingServer: true,
      timeout: 120_000,
    },
    {
      command: 'npx vite --port 5173 --strictPort --host 127.0.0.1',
      cwd: '..',
      url: 'http://127.0.0.1:5173',
      reuseExistingServer: true,
      timeout: 60_000,
    },
  ],
})
