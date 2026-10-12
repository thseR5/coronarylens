// Builds the app AND the 3D harness (e2e/harness.html) so FPS is measured on a production bundle.
// Run from frontend/: npx vite build -c e2e/vite.harness.config.ts
import { mergeConfig } from 'vite'
import base from '../vite.config'

export default mergeConfig(base, {
  build: {
    outDir: 'test-results/harness-dist',
    emptyOutDir: true,
    rollupOptions: { input: { app: 'index.html', harness: 'e2e/harness.html' } },
  },
})
