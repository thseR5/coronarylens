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
Configurations so far: 12 + 278 = 290 (A1).

### Finalist-selection rule (written into this log 2026-10-13 while A2 was running, before its results were read; it was committed together with the A2 results, so the ordering is not verifiable from git history)

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

### Screening A2 (training split only; same folds as A1; NOT headline numbers)

498 procedures: learners lr_l2_strong, catboost, catboost_d3, ebm, bayes_lr (clinical prior), bayes_lr_flat (same model, no prior), sel_joint (in-fold L1 selection), stack (per-target in-fold feature sets → OOF logits → joint LR), svm; each alone with vector / Dirichlet calibration; all 2–3 learner averages and learned weights with vector and Dirichlet calibration. Wall time 404.2 s (3 single-threaded workers).

New candidates on their own:

| Procedure | CAD | LAD | LCX | RCA | Pattern log-loss (±SE) | Mean Brier |
|---|---|---|---|---|---|---|
| bayes_lr \| avg \| dirichlet | 0.923±0.037 | 0.846±0.053 | 0.708±0.046 | 0.725±0.067 | 1.5366±0.0176 | 0.166±0.017 |
| bayes_lr_flat \| avg \| dirichlet | 0.916±0.041 | 0.839±0.055 | 0.703±0.047 | 0.722±0.070 | 1.5555±0.0193 | 0.168±0.019 |
| bayes_lr \| avg \| vector | 0.924±0.036 | 0.840±0.054 | 0.699±0.054 | 0.733±0.063 | 1.6044±0.0148 | 0.167±0.013 |
| bayes_lr_flat \| avg \| vector | 0.916±0.041 | 0.837±0.056 | 0.695±0.053 | 0.728±0.066 | 1.6188±0.0158 | 0.170±0.014 |
| sel_joint \| avg \| dirichlet | 0.936±0.031 | 0.868±0.049 | 0.728±0.055 | 0.734±0.044 | 1.5287±0.0167 | 0.158±0.015 |
| sel_joint \| avg \| vector | 0.936±0.031 | 0.866±0.047 | 0.712±0.059 | 0.746±0.045 | 1.5587±0.0131 | 0.160±0.012 |
| stack \| avg \| dirichlet | 0.910±0.046 | 0.835±0.059 | 0.700±0.049 | 0.712±0.069 | 1.5645±0.0210 | 0.171±0.020 |
| stack \| avg \| vector | 0.912±0.043 | 0.831±0.056 | 0.698±0.051 | 0.723±0.076 | 1.6278±0.0167 | 0.172±0.015 |
| svm \| avg \| vector | 0.914±0.043 | 0.843±0.052 | 0.706±0.054 | 0.723±0.059 | 1.5611±0.0140 | 0.168±0.015 |
| svm \| avg \| dirichlet | 0.913±0.044 | 0.844±0.051 | 0.704±0.051 | 0.727±0.063 | 1.5662±0.0167 | 0.167±0.016 |

Top 10 by pattern log-loss:

| Procedure | CAD | LAD | LCX | RCA | Pattern log-loss (±SE) | Mean Brier |
|---|---|---|---|---|---|---|
| catboost_d3+sel_joint+svm \| avg \| dirichlet | 0.936±0.036 | 0.874±0.047 | 0.727±0.060 | 0.754±0.048 | 1.5049±0.0153 | 0.155±0.015 |
| ebm+bayes_lr+sel_joint \| avg \| dirichlet | 0.940±0.032 | 0.869±0.053 | 0.715±0.056 | 0.739±0.061 | 1.5051±0.0151 | 0.156±0.015 |
| catboost_d3+bayes_lr+sel_joint \| avg \| dirichlet | 0.941±0.033 | 0.870±0.049 | 0.715±0.051 | 0.744±0.056 | 1.5054±0.0154 | 0.155±0.015 |
| catboost_d3+sel_joint+svm \| avg \| vector | 0.936±0.036 | 0.874±0.046 | 0.727±0.059 | 0.753±0.044 | 1.5058±0.0121 | 0.156±0.013 |
| catboost+bayes_lr+sel_joint \| avg \| dirichlet | 0.940±0.034 | 0.871±0.050 | 0.715±0.054 | 0.743±0.055 | 1.5059±0.0157 | 0.155±0.015 |
| catboost+sel_joint+svm \| avg \| dirichlet | 0.937±0.036 | 0.874±0.047 | 0.731±0.061 | 0.752±0.049 | 1.5059±0.0159 | 0.154±0.015 |
| catboost+sel_joint+svm \| avg \| vector | 0.937±0.035 | 0.875±0.047 | 0.728±0.059 | 0.754±0.045 | 1.5060±0.0128 | 0.155±0.014 |
| catboost+svm \| avg \| vector | 0.933±0.038 | 0.875±0.045 | 0.752±0.064 | 0.754±0.059 | 1.5078±0.0152 | 0.157±0.016 |
| catboost+catboost_d3+svm \| avg \| vector | 0.936±0.036 | 0.878±0.039 | 0.751±0.061 | 0.754±0.055 | 1.5079±0.0147 | 0.156±0.016 |
| lr_l2_strong+bayes_lr+sel_joint \| avg \| dirichlet | 0.937±0.032 | 0.862±0.054 | 0.716±0.047 | 0.742±0.060 | 1.5081±0.0155 | 0.157±0.015 |

A2 reading: (d) the clinical prior helps the Bayesian LR on its own (1.537 vs 1.556 with Dirichlet, ≈1 SE; CAD AUC 0.923 vs 0.916); (e) in-fold selection is a strong single learner (1.529); (f+g) stacking with per-target feature sets is worse than the baseline (1.565) and is dropped; (k) SVM is weak alone but adds diversity in combinations. Best combinations reach ≈1.505 (baseline 1.572).

**Total screened: 748 unique procedures (A1 278 + A2 498, overlapping names counted once).**

### Finalists (pre-registered rule applied mechanically by `ml/sprint/finalists.py`)

| Finalist | Reason | Screening log-loss | CAD | LAD | LCX | RCA |
|---|---|---|---|---|---|---|
| catboost_d3+sel_joint+svm \| avg \| dirichlet | top-8 log-loss | 1.5049 | 0.936 | 0.874 | 0.727 | 0.754 |
| ebm+bayes_lr+sel_joint \| avg \| dirichlet | top-8 log-loss | 1.5051 | 0.940 | 0.869 | 0.715 | 0.739 |
| catboost_d3+bayes_lr+sel_joint \| avg \| dirichlet | top-8 log-loss | 1.5054 | 0.941 | 0.870 | 0.715 | 0.744 |
| catboost_d3+sel_joint+svm \| avg \| vector | top-8 log-loss | 1.5058 | 0.936 | 0.874 | 0.727 | 0.753 |
| catboost+bayes_lr+sel_joint \| avg \| dirichlet | top-8 log-loss | 1.5059 | 0.940 | 0.871 | 0.715 | 0.743 |
| catboost+sel_joint+svm \| avg \| dirichlet | top-8 log-loss | 1.5059 | 0.937 | 0.874 | 0.731 | 0.752 |
| catboost+sel_joint+svm \| avg \| vector | top-8 log-loss | 1.5060 | 0.937 | 0.875 | 0.728 | 0.754 |
| catboost+svm \| avg \| vector | top-8 log-loss | 1.5078 | 0.933 | 0.875 | 0.752 | 0.754 |
| catboost+catboost_d3+svm \| avg \| vector | best LCX+RCA AUC | 1.5079 | 0.936 | 0.878 | 0.751 | 0.754 |
| lr_en+rf+et \| avg \| vector | baseline (served recipe) | 1.5724 | 0.933 | 0.855 | 0.717 | 0.743 |

Note: in the nested run the baseline's forests use 300 trees per forest (the sprint learner spec), the served model 500; the comparison is like-for-like inside the sprint.

### Nested result B1 (headline): 10×5 outer folds, selection inside every fold

Candidates: 10 (9 finalists + previous served recipe). Wall time 1824 s on 3 single-threaded workers. A second run with stored outer predictions reproduced every metric and every per-fold choice exactly.

| Metric | Selected procedure (mean ± SD) | Previous served recipe | Paired diff (mean ± SD) | SE | Folds improved |
|---|---|---|---|---|---|
| Pattern log-loss | 1.5084 ± 0.0909 | 1.5754 ± 0.0779 | -0.0670 ± 0.0392 | 0.0055 | 46/50 (lower is better) |
| Mean Brier | 0.1579 ± 0.0137 | 0.1617 ± 0.0121 | -0.0039 ± 0.0059 | 0.0008 | 40/50 (lower is better) |
| CAD ROC-AUC | 0.9343 ± 0.0363 | 0.9271 ± 0.0406 | +0.0073 ± 0.0109 | 0.0015 | 39/50 (higher is better) |
| LAD ROC-AUC | 0.8657 ± 0.0429 | 0.8508 ± 0.0475 | +0.0149 ± 0.0193 | 0.0027 | 35/50 (higher is better) |
| LCX ROC-AUC | 0.7081 ± 0.0580 | 0.7139 ± 0.0515 | -0.0058 ± 0.0460 | 0.0065 | 23/50 (higher is better) |
| RCA ROC-AUC | 0.7373 ± 0.0556 | 0.7441 ± 0.0546 | -0.0068 ± 0.0416 | 0.0059 | 23/50 (higher is better) |
| ≥2 vessels ROC-AUC | 0.7657 ± 0.0515 | 0.7496 ± 0.0540 | +0.0161 ± 0.0444 | 0.0063 | 34/50 (higher is better) |
| 3-vessel ROC-AUC | 0.7553 ± 0.0625 | 0.7417 ± 0.0730 | +0.0136 ± 0.0562 | 0.0079 | 34/50 (higher is better) |

Inner choice per outer fold: ebm+bayes_lr+sel_joint|avg|dirichlet ×28, catboost+bayes_lr+sel_joint|avg|dirichlet ×10, catboost_d3+bayes_lr+sel_joint|avg|dirichlet ×4, catboost+catboost_d3+svm|avg|vector ×3, catboost_d3+sel_joint+svm|avg|vector ×2, catboost+svm|avg|vector ×2, catboost+sel_joint+svm|avg|vector ×1.

**Decision (pre-registered table; plain-SE wording, superseded by the corrected statistics below, where only log-loss is significant):** real gain in calibration/risk estimation (pattern log-loss −0.067, 12 SE, 46/50 folds; mean Brier −0.0039, 40/50) and in CAD and LAD discrimination (+0.007 and +0.015 AUC, both > 1 SE, 39/50 and 35/50 folds) → **adopt**. LCX and RCA: differences within noise (23/50 folds each) → **the LCX/RCA ceiling held after 748 screened procedures**. No implausible jump (CAD 0.934, LCX 0.708, RCA 0.737 ≪ alarm thresholds).

**Stopping rule:** every candidate (a)–(k) has been tried; the last three candidate families screened (stacking (f+g), SVM alone (k), vessel-count blend (h)) brought no gain > 1 SE on their own → stop.

**Total configurations:** 12 (Phase 1 benchmarks) + 748 screened procedures + 10 finalists in nested CV. Selection happened inside nested CV, so the reported performance accounts for that search.

### Corrected statistics (2026-10-13, user request): Nadeau–Bengio corrected resampled t-test

Repeated-CV folds share training data, so the plain SE of a mean over folds is too small. Corrected variance = (1/J + n_test/n_train)·s² with J = 50 outer folds, n_train ≈ 193.6, n_test ≈ 48.4 (factor 0.270 instead of 0.020, i.e. SE × 3.7); t with 49 df; two-sided p. **These replace the plain-SE statements above; only the corrected figures are used in the dashboard, report and Q&A.**

**Served procedure vs previous served model (same 10×5 folds):**

| Metric | Paired diff (mean ± SD) | Corrected SE | t (df 49) | p | Folds improved | Verdict |
|---|---|---|---|---|---|---|
| Pattern log-loss | -0.0670 ± 0.0392 | 0.0204 | -3.29 | 0.002 | 46/50 | significant improvement |
| Mean Brier | -0.0039 ± 0.0059 | 0.0031 | -1.25 | 0.217 | 40/50 | small, not statistically significant |
| CAD AUC | +0.0073 ± 0.0109 | 0.0057 | +1.28 | 0.207 | 39/50 | small, not statistically significant |
| LAD AUC | +0.0149 ± 0.0193 | 0.0100 | +1.49 | 0.142 | 35/50 | small, not statistically significant |
| LCX AUC | -0.0058 ± 0.0460 | 0.0239 | -0.24 | 0.809 | 23/50 | no improvement (not significant) |
| RCA AUC | -0.0068 ± 0.0416 | 0.0216 | -0.31 | 0.755 | 23/50 | no improvement (not significant) |
| ≥2 vessels AUC | +0.0161 ± 0.0444 | 0.0231 | +0.70 | 0.489 | 34/50 | small, not statistically significant |
| 3-vessel AUC | +0.0136 ± 0.0562 | 0.0292 | +0.47 | 0.642 | 34/50 | small, not statistically significant |

Reading: only the risk estimate (pattern log-loss) improves significantly. Brier, CAD/LAD and ≥2/3-vessel AUC gains are small and not statistically significant; LCX/RCA show no improvement. Adoption still holds under the pre-registered rule (log-loss **or** AUC improvement): log-loss t = −3.29, p = 0.002.

**Served procedure vs independent binary ensemble (identical 10×5 folds, `reports/sprint/indep_B1.json`):**

| Metric | Paired diff (mean ± SD) | Corrected SE | t (df 49) | p | Folds improved | Verdict |
|---|---|---|---|---|---|---|
| Pattern log-loss | -0.1533 ± 0.0751 | 0.0390 | -3.93 | <0.001 | 49/50 | significant improvement |
| Mean Brier | -0.0084 ± 0.0080 | 0.0042 | -2.02 | 0.049 | 43/50 | significant improvement |
| CAD AUC | +0.0083 ± 0.0131 | 0.0068 | +1.23 | 0.224 | 34/50 | small, not statistically significant |
| LAD AUC | +0.0109 ± 0.0237 | 0.0123 | +0.88 | 0.381 | 31/50 | small, not statistically significant |
| LCX AUC | +0.0022 ± 0.0510 | 0.0265 | +0.08 | 0.934 | 27/50 | small, not statistically significant |
| RCA AUC | -0.0045 ± 0.0515 | 0.0268 | -0.17 | 0.869 | 24/50 | no improvement (not significant) |

Coherence on these folds: independent ensemble 5.4% of patients with P(CAD) < max vessel; served procedure 0.0% (measured on all 50 outer folds; 0 by construction). "Equal AUC, better log-loss" therefore remains accurate for the served model.

**Headline (absolute, served procedure, 10×5 nested CV, mean [95% corrected CI]):**

| Target | ROC-AUC | PR-AUC | Brier | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| CAD | 0.934 [0.896, 0.973] | 0.972 [0.954, 0.991] | 0.089 [0.062, 0.116] | 0.880 [0.840, 0.920] | 0.909 [0.865, 0.953] | 0.926 [0.883, 0.968] | 0.916 [0.889, 0.944] |
| LAD | 0.866 [0.820, 0.911] | 0.894 [0.850, 0.938] | 0.144 [0.121, 0.167] | 0.805 [0.761, 0.848] | 0.789 [0.741, 0.837] | 0.913 [0.865, 0.961] | 0.845 [0.811, 0.879] |
| LCX | 0.708 [0.647, 0.769] | 0.578 [0.507, 0.649] | 0.204 [0.186, 0.223] | 0.630 [0.583, 0.676] | 0.515 [0.477, 0.554] | 0.848 [0.733, 0.963] | 0.638 [0.587, 0.690] |
| RCA | 0.737 [0.679, 0.796] | 0.583 [0.494, 0.673] | 0.194 [0.179, 0.209] | 0.654 [0.603, 0.705] | 0.527 [0.474, 0.580] | 0.848 [0.740, 0.956] | 0.647 [0.591, 0.703] |
| ≥2 vessels | 0.766 [0.711, 0.820] | | | | | | |
| 3 vessels | 0.755 [0.689, 0.821] | | | | | | |

Pattern log-loss 1.5084 [1.4126, 1.6043]; mean Brier 0.1579.

### What-if plausibility check (`ml/whatif_check.py`)

11 key features × 5 presets swept across the observed training range with the served model. Flags: 6; none for P(CAD) (overall risk always moves in the clinically expected direction); no single-step jump > 0.15. Flags are small vessel-level reversals (0.025–0.048): Region RWMA → LCX/RCA (regional finding; the expected sign in features.yaml is for overall CAD), LDL → LAD (−0.031; LDL has a weak, nearly flat effect in this cohort), EF → LCX (+0.028). Visible but sub-threshold: a step in P(LAD) of 0.03–0.11 depending on the preset as EF crosses ~52–55% (tree split at the reference-range boundary). Curves: `reports/whatif/whatif_<preset>.png` and the performance tab.

Monotone-constrained variant: not triggered (no material problem) and not available as specified: CatBoost does not support monotone constraints with the MultiClass loss used for the 8-pattern model; logged rather than substituted.
