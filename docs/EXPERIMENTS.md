# Experiments log

Every configuration evaluated, including failures. Rules (user, 2026-10-12):
- The locked 61-patient test set is never used to choose, tune or reject anything.
- Model selection happens inside nested CV; the headline is the outer score of the whole selection procedure.
- A change is adopted only if it improves outer-CV log-loss or AUC by > 1 SE, or matches it with a simpler model.
- Screening runs (repeated CV on the training split only) are never headline numbers; finalists go through full nested CV.

## Phase 1 baseline (2026-10-12, nested CV 5×5 outer / 3 inner on the 242-patient training split)

Source: `ml/artifacts/metrics.json` (`cv.variants`). ROC-AUC mean ± SD over 25 outer folds; mean pattern log-loss; coherence
violations = share of patients with P(CAD) < max vessel probability.

| # | Configuration | CAD | LAD | LCX | RCA | Pattern log-loss | Contradictions |
|---|---|---|---|---|---|---|---|
| 1 | joint ensemble (LR+RF+ET), no calibration | 0.924 | 0.848 | 0.704 | 0.737 | 0.500 | 0% |
| 2 | joint ensemble, temperature scaling | 0.924 | 0.848 | 0.703 | 0.737 | 0.502 | 0% |
| 3 | **joint ensemble, vector scaling (served)** | 0.923±0.044 | 0.850±0.057 | 0.711±0.053 | 0.740±0.051 | 0.489 | 0% |
| 4–6 | joint + engineered features (none / temperature / vector) | 0.927 (vector) | 0.853 | 0.710 | 0.744 | 0.488 (vector) | 0% |
| 7–8 | LightGBM joint (none / temperature) | 0.923 (temp) | 0.848 | 0.727 | 0.736 | 0.517 | 0% |
| 9 | independent binary ensembles (LR+RF+ET per target) | 0.924 | 0.854 | 0.707 | 0.742 | 0.501 | 5.9% ± 3.3% |
| 10 | classifier chains (elastic-net LR) | 0.914 | 0.834 | 0.655 | 0.724 | 0.544 | 23.4% |

| 11 | naive LR ×4 (L2, C = 1, lbfgs, StandardScaler) | 0.897 | 0.800 | 0.626 | 0.706 | 2.432 (8-pattern) | 32.1% ± 5.6% |
| 12 | regularized LR ×4 (elastic-net C = 0.3, l1 0.5) — same folds, `ml/benchmark_extra.py` | 0.914 | 0.833 | 0.654 | 0.725 | 1.842 (8-pattern) | 24.8% ± 6.4% |

Note: the "Pattern log-loss" column for rows 1–10 is the mean of per-target log-losses (`cv.headline`); rows 11–12 give the
8-pattern log-loss. For comparison the 8-pattern log-loss of rows 3 / 9 / 10 is 1.578 / 1.663 / 1.843.

Engineered features: paired Δ log-loss −0.0005 ± 0.0046, Δ AUC +0.0024 ± 0.0058 → within noise, not adopted.
Configurations so far: 12.

## Part 2 sprint (2026-10-13)

Candidate status before screening:
- (a) TabPFN v2: **considered, not used**: license-gated weights (`tabpfn` 9.2.0 requires accepting the Prior Labs license and an
  API token, `TabPFNLicenseError` in a non-interactive run) and CPU inference cost incompatible with live SHAP latency (the served
  explanation evaluates the model on ~5,000 coalition rows per request within a 700 ms budget). Package removed (torch ≈ 1 GB).
- (b) Monotonic constraints: LightGBM accepts `monotone_constraints` for multiclass, but on the 8-pattern model a constraint
  forces every pattern's raw score up together, which after the softmax does not mean "more diseased vessels". Not a meaningful
  test, so not run as an 8-class learner; the clinically directed prior is tested via `bayes_lr` (d) instead. CatBoost does not
  support monotone constraints for MultiClass.

### Screening A1 (training split only, 5×5 repeated CV, inner 3-fold OOF; NOT headline numbers)

278 procedures: 9 learners × 4 calibrations, all 2–3 learner averages / learned weights (vector calibration), vessel-count blend. Learner cache 237.3 s on 3 single-threaded workers. AUC = mean±SD over 25 folds; pattern log-loss = mean±SE; Brier = mean over CAD/LAD/LCX/RCA.

| Procedure | CAD | LAD | LCX | RCA | Pattern log-loss (±SE) | Mean Brier |
|---|---|---|---|---|---|---|
| **baseline (served recipe)** lr_en+rf+et \| avg \| vector | 0.933±0.037 | 0.855±0.047 | 0.717±0.054 | 0.743±0.059 | 1.5724±0.0146 | 0.162±0.014 |
| (j) learned weights on baseline | 0.932±0.037 | 0.859±0.047 | 0.728±0.053 | 0.741±0.058 | 1.5681±0.0147 | 0.162±0.015 |
| (h) vessel-count blend λ = 0.25 | 0.933±0.036 | 0.855±0.048 | 0.715±0.052 | 0.742±0.061 | 1.5753±0.0138 | 0.162±0.014 |
| (h) vessel-count blend λ = 0.5 | 0.932±0.038 | 0.852±0.050 | 0.711±0.051 | 0.741±0.061 | 1.5806±0.0131 | 0.162±0.013 |

Best calibration per learner:

| Procedure | CAD | LAD | LCX | RCA | Pattern log-loss (±SE) | Mean Brier |
|---|---|---|---|---|---|---|
| lr_l2 \| avg \| dirichlet | 0.924±0.038 | 0.847±0.056 | 0.705±0.042 | 0.732±0.070 | 1.5276±0.0174 | 0.164±0.018 |
| lr_l2_strong \| avg \| dirichlet | 0.931±0.037 | 0.855±0.049 | 0.713±0.041 | 0.744±0.070 | 1.5287±0.0154 | 0.159±0.016 |
| catboost \| avg \| dirichlet | 0.936±0.034 | 0.875±0.039 | 0.735±0.060 | 0.741±0.058 | 1.5292±0.0177 | 0.157±0.017 |
| catboost_d3 \| avg \| dirichlet | 0.935±0.037 | 0.873±0.037 | 0.741±0.054 | 0.740±0.051 | 1.5294±0.0169 | 0.158±0.017 |
| ebm \| avg \| dirichlet | 0.933±0.036 | 0.867±0.048 | 0.708±0.060 | 0.728±0.070 | 1.5304±0.0150 | 0.160±0.015 |
| lr_en \| avg \| dirichlet | 0.922±0.039 | 0.848±0.054 | 0.721±0.045 | 0.737±0.062 | 1.5309±0.0185 | 0.163±0.018 |
| rf \| avg \| dirichlet | 0.932±0.038 | 0.869±0.037 | 0.734±0.062 | 0.728±0.060 | 1.5383±0.0175 | 0.160±0.018 |
| lgbm \| avg \| dirichlet | 0.928±0.038 | 0.860±0.052 | 0.708±0.062 | 0.727±0.068 | 1.5452±0.0195 | 0.163±0.018 |
| et \| avg \| dirichlet | 0.921±0.039 | 0.848±0.047 | 0.708±0.059 | 0.721±0.065 | 1.5632±0.0175 | 0.166±0.018 |

Top 12 overall by pattern log-loss:

| Procedure | CAD | LAD | LCX | RCA | Pattern log-loss (±SE) | Mean Brier |
|---|---|---|---|---|---|---|
| lr_l2 \| avg \| dirichlet | 0.924±0.038 | 0.847±0.056 | 0.705±0.042 | 0.732±0.070 | 1.5276±0.0174 | 0.164±0.018 |
| lr_l2_strong \| avg \| dirichlet | 0.931±0.037 | 0.855±0.049 | 0.713±0.041 | 0.744±0.070 | 1.5287±0.0154 | 0.159±0.016 |
| catboost \| avg \| dirichlet | 0.936±0.034 | 0.875±0.039 | 0.735±0.060 | 0.741±0.058 | 1.5292±0.0177 | 0.157±0.017 |
| catboost_d3 \| avg \| dirichlet | 0.935±0.037 | 0.873±0.037 | 0.741±0.054 | 0.740±0.051 | 1.5294±0.0169 | 0.158±0.017 |
| ebm \| avg \| dirichlet | 0.933±0.036 | 0.867±0.048 | 0.708±0.060 | 0.728±0.070 | 1.5304±0.0150 | 0.160±0.015 |
| lr_en \| avg \| dirichlet | 0.922±0.039 | 0.848±0.054 | 0.721±0.045 | 0.737±0.062 | 1.5309±0.0185 | 0.163±0.018 |
| catboost+catboost_d3 \| avg \| vector | 0.937±0.035 | 0.877±0.037 | 0.750±0.057 | 0.748±0.052 | 1.5330±0.0145 | 0.158±0.016 |
| lgbm+catboost_d3 \| learned \| vector | 0.936±0.036 | 0.873±0.039 | 0.748±0.061 | 0.748±0.051 | 1.5334±0.0158 | 0.158±0.016 |
| catboost_d3 \| avg \| vector | 0.935±0.037 | 0.875±0.037 | 0.747±0.056 | 0.745±0.049 | 1.5334±0.0136 | 0.159±0.015 |
| rf+catboost_d3 \| learned \| vector | 0.935±0.037 | 0.875±0.037 | 0.748±0.057 | 0.746±0.050 | 1.5338±0.0135 | 0.159±0.015 |
| lgbm+catboost_d3+ebm \| learned \| vector | 0.938±0.035 | 0.875±0.040 | 0.746±0.060 | 0.748±0.053 | 1.5342±0.0157 | 0.158±0.016 |
| rf+lgbm+catboost_d3 \| learned \| vector | 0.937±0.036 | 0.873±0.039 | 0.749±0.061 | 0.749±0.053 | 1.5344±0.0157 | 0.158±0.016 |

A1 reading: (i) Dirichlet calibration gives the lowest log-loss for most learners (≈1.53 vs baseline 1.572, ~3 SE) but can lower AUC; (b) CatBoost is the strongest single learner on AUC (LAD 0.875, LCX 0.747 with vector scaling); (j) learned weights and (h) the vessel-count blend give no gain over the baseline (within / worse than 1 SE). With 278 procedures screened, winner's-curse risk is high: only nested CV decides.
Configurations so far: 12 + 278 = 290.

### Finalist-selection rule (written 2026-10-13 before A2 results were read)

From all procedures screened in A1 ∪ A2 (training split only):
1. the 8 procedures with the lowest screening pattern log-loss;
2. plus the procedure with the highest mean of LCX and RCA screening AUC (the sprint's focus), if not already included;
3. plus the procedure with the highest mean CAD/LAD/LCX/RCA screening AUC, if not already included;
4. plus the current served recipe (lr_en + rf + et | avg | vector) as the baseline.
All finalists enter the 10×5 nested run together; inside each outer fold the procedure with the lowest cross-fitted inner-OOF
pattern log-loss is selected and scored on the outer-test part. The headline is that outer score. Gains are reported as paired
outer-fold differences vs the served recipe (mean ± SD, SE, folds improved). Adoption: improvement > 1 SE in outer pattern
log-loss or mean AUC, or equal performance with a simpler procedure.

### Reading the nested result (pre-registered 2026-10-13, before nested results exist)

| Outcome | Interpretation | Action |
|---|---|---|
| Selected procedure beats the served recipe by > 1 SE (paired outer folds) and most folds improve | real gain | adopt; rerun all tests; update artifacts, dashboard and docs |
| Small mean gain but < 1 SE, or only about half the folds improve | noise | keep the served model; report "ceiling held after N procedures" |
| Implausible jump: LCX or RCA AUC > ~0.80, or CAD AUC > ~0.96 | almost certainly a bug or leakage | stop and investigate before adopting anything |
