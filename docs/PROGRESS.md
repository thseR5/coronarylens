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
| 4 Docs, packaging, review | done; final round steps 1–5 done, features refrozen (see DECISIONS) | user-only: licence name, publish repo, video, Devpost, HF Space (`docs/SUBMISSION_CHECKLIST.md`) |

### Live-update latency (Part 1 gate) — production build, idle machine, 2026-10-12

40 unique edits per run, cold server cache, real FastAPI backend; times include the 100 ms debounce. Source: `docs/latency.json`.

| Run | Load (1 min) before | Colours + probabilities p50 / p95 | SHAP panel p50 / p95 |
|---|---|---|---|
| 1 | 3.95 | 157 / 159 ms | 253 / 271 ms |
| 2 | 21.91 (previous run winding down) | 157 / 159 ms | 240 / 242 ms |
| 3 | 14.36 (previous run winding down) | 156 / 158 ms | 240 / 242 ms |
| Target (p95) | | ≤ 300 ms | ≤ 700 ms |

### Part 2 result (2026-10-13)

- 748 procedures screened (training split only), 10 finalists (pre-registered rule) in 10×5 nested CV with selection inside every fold.
- Adopted (pre-registered table). Corrected (Nadeau–Bengio) paired test: pattern log-loss −0.067, p 0.002 (46/50 folds) significant; Brier, CAD and LAD gains small, not statistically significant; LCX/RCA no improvement → ceiling held. Vs independent ensemble on identical folds: equal AUC (all p > 0.2), log-loss p 0.0003, Brier p 0.049; contradictions 0% vs 5.4%.
- Served on the full training split: catboost + catboost_d3 + svm | avg | vector.
- Locked test set, second look (n = 61): CAD 0.898, LAD 0.829, LCX 0.761, RCA 0.661 (first look 0.920 / 0.842 / 0.692 / 0.685); all differences inside the wide 95% CIs; nested CV remains the primary estimate.
- Latency (production build, load ≈ 3.5): colours p95 175 ms, SHAP p95 240–241 ms. Explain in-process p50 47 ms.
