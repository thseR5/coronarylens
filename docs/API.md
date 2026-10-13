# API v1 (additive changes only since freeze)

Types: `backend/app/schemas.py` (source) ⇄ `frontend/src/types/api.ts` (mirror).

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/api/health` | – | `HealthResponse` |
| GET | `/api/config` | – | `{features, targets, anatomy, models}`: the four YAMLs as JSON |
| GET | `/api/presets` | – | `Preset[]` (6-8 test-split patients) |
| GET | `/api/metrics` | – | contents of `ml/artifacts/metrics.json` |
| GET | `/api/global-explanations` | – | contents of `ml/artifacts/global_explanations.json` |
| POST | `/api/analyze` | `AnalyzeRequest` | `AnalyzeResponse` |
| POST | `/api/explain` | `ExplainRequest` | `ExplainResponse` |
| POST | `/api/counterfactual` | `CounterfactualRequest` | `CounterfactualResponse` |
| POST | `/api/similar` | `SimilarRequest` | `SimilarResponse` |
| POST | `/api/cohort-map` | `ExplainRequest` | `CohortMapResponse` (2-D PCA of the training split + projected patient) |
| GET | `/api/training-patient/{index}` | – | `TrainingPatient` (training split only; 404 for test rows) |
| POST | `/api/report` | `ExplainRequest` | printable HTML with disclaimer (UI: Report / PDF → browser Print) |

Invariants (tested): pattern probabilities sum to 1; `cad.p ≥ max(vessels[*].p)`; `cad.p = 1 − patterns["000"]`;
no forbidden column (`LAD, LCX, RCA, Cath`) is accepted as a feature (422).
Latency (production build, laptop CPU, slider change → painted, incl. 100 ms debounce): probabilities and colours p95 175 ms, SHAP panel p95 241 ms (targets 300 / 700 ms; `docs/latency.json`). Server-side test budget (`backend/tests`): analyze p95 < 150 ms, explain p95 < 400 ms. Status: `likely`/`unlikely` only when the 90% conformal set is one outcome and agrees with the threshold decision, else `uncertain`.
