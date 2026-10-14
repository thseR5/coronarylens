# API v1 (additive changes only since freeze)

Types: `backend/app/schemas.py` (source) ⇄ `frontend/src/types/api.ts` (mirror).

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/api/health` | – | `HealthResponse` |
| GET | `/api/config` | – | `{features, targets, anatomy, models}`: the four YAMLs as JSON |
| GET | `/api/presets` | – | `Preset[]` (6 test-split patients + 1 clearly labelled synthetic out-of-domain example) |
| GET | `/api/metrics` | – | contents of `ml/artifacts/metrics.json` |
| GET | `/api/global-explanations` | – | contents of `ml/artifacts/global_explanations.json` |
| POST | `/api/analyze` | `AnalyzeRequest` | `AnalyzeResponse` |
| POST | `/api/explain` | `ExplainRequest` | `ExplainResponse` |
| POST | `/api/counterfactual` | `CounterfactualRequest` | `CounterfactualResponse` |
| POST | `/api/similar` | `SimilarRequest` | `SimilarResponse` |
| POST | `/api/cohort-map` | `ExplainRequest` | `CohortMapResponse` (2-D PCA of the training split + projected patient) |
| GET | `/api/training-patient/{index}` | – | `TrainingPatient` (training split only; 404 for test rows) |
| POST | `/api/report` | `ExplainRequest` | printable HTML with disclaimer (UI: Report / PDF → browser Print) |
| GET | `/api/report?features=<JSON>&print=1` | query | same report by URL; `print=1` opens the Print dialog (UI: Report / PDF opens this in a new tab, which also works inside a Hugging Face Space frame) |

Invariants (tested): pattern probabilities sum to 1; `cad.p ≥ max(vessels[*].p)`; `cad.p = 1 − patterns["000"]`;
no forbidden column (`LAD, LCX, RCA, Cath`) is accepted as a feature (422).
Latency (production build, laptop CPU, slider change → painted, incl. 100 ms debounce): probabilities and colours p95 226 ms, SHAP panel p95 360 ms (targets 300 / 700 ms; `docs/latency.json`). Server-side regression guard (`backend/tests`, median so it is robust on shared CI runners): analyze p50 < 300 ms, explain p50 < 800 ms. Status: `likely`/`unlikely` only when the 90% conformal set is one outcome and agrees with the threshold decision, else `uncertain`; always `uncertain` when the patient is out of distribution or when imputed inputs carry ≥ 25% of the model's usual signal (`input_confidence.missing_signal_share`, each input weighted by its global mean |SHAP| for CAD; `level` is `full` < 0.10 ≤ `reduced` < 0.50 ≤ `low`). Every risk carries `reason`: `single_outcome`, `both_outcomes`, `empty_set`, `threshold_disagrees`, `out_of_distribution` or `missing_inputs`.
