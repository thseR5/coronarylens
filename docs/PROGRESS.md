# Progress

No deadline (user, 2026-10-12): optimize for the best result.

| Phase | Status | Evidence (2026-10-12) |
|---|---|---|
| 0 Setup, config, contract | done | commit 26685d5 |
| 1 ML (Gate A) | done | 54 pytest pass; nested CV + test CIs in `ml/artifacts/metrics.json`; two trainings byte-identical |
| 2 Backend (Gate B) | done | contract, coherence, leakage, latency tests pass; analyze p95 ≈ 23 ms, explain p95 ≈ 253–328 ms |
| 3 3D (Gate C) | done | anatomy.glb 0.41 MB / 85,922 tris, no missing FMA IDs; vitest 13/13; GPU 60 FPS; `--disable-gpu` ≈ 30 FPS in low-power mode (noisy under load) |
| 4 Dashboard (Gate D) | done | production build, idle machine: colours p95 158–159 ms, SHAP p95 242–271 ms (docs/latency.json) |
| 5 Tier 2 | mostly in (conformal, AD, territories, bullseye, presets, heartbeat, counterfactuals, similar, DCA, ablation) | Model performance tab + TRIPOD checklist pending |
| 6 Docs + judge review | pending | |

## Round 2 (user plan, 2026-10-12)

| Part | Status | Evidence |
|---|---|---|
| 1 Fixes | done | coherence table incl. naive LR (32.1%) + regularized-LR footnote (24.8%); AHA 17-segment territories default (test: 3D == bullseye); Voronoi toggle (14/17); airways layer; latency gate passed (docs/latency.json); nested CV headline; 61 pytest + 21 vitest + 3D/full-stack e2e green |
| 2 Performance sprint | next | harness in ml/sprint/ |
| 3 Tier 2/3 completion | pending | |
| 4 Docs, packaging, review | pending | |
