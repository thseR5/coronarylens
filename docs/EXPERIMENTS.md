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
- (a) TabPFN v2: **skipped**. `tabpfn` 9.2.0 installs, but downloading weights requires accepting the Prior Labs license and
  an API token (`TabPFNLicenseError`, non-interactive). Accepting a license is the user's decision; it can be added later with
  `TABPFN_TOKEN`. Package removed to keep the environment lean (torch ≈ 1 GB).
- (b) Monotonic constraints: LightGBM accepts `monotone_constraints` for multiclass, but on the 8-pattern model a constraint
  forces every pattern's raw score up together, which after the softmax does not mean "more diseased vessels". Not a meaningful
  test, so not run as an 8-class learner; the clinically directed prior is tested via `bayes_lr` (d) instead. CatBoost does not
  support monotone constraints for MultiClass.
