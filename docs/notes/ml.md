# ML + backend notes (sub-agent A)

Decision support and educational use only. Not a substitute for formal diagnostic imaging (e.g., coronary
angiography) or clinical judgment.

## Decisions (one line each, with rationale)

- Model matrix = feature ids from features.yaml; `bbb` one-hot (`bbb_LBBB`, `bbb_RBBB`, N = baseline), ordinals as codes 0..k (vhd N/mild/Moderate/Severe → 0-3; function_class / region_rwma raw value → index). One code path (`ml/ingest.py:encode_value`) serves training and API input, so encodings cannot drift.
- Leakage guard checks model columns, feature ids AND source columns (normalized: case/space/punctuation-insensitive) against LAD/LCX/RCA/Cath/cad at ingest, training and inference; API returns 422 for forbidden keys.
- Engineered features (eGFR CKD-EPI 2021 race-free, TG/HDL, LDL/HDL, NLR, risk-factor count, ECG-abnormality count) implemented as a stateless transformer inside the pipeline; kept only if the nested-CV selection rule picks a `joint_fe_*` variant (see Evidence).
- Nested CV uses 200 trees per forest (served/final model uses the 500 from models.yaml) to keep total train time under ~20 min on the M2; same 200 in inner and outer loops so calibrators/thresholds match the model they were fitted for.
- Calibration (none / temperature / vector) is fitted on inner OOF 8-class probabilities, applied before marginalization (coherence preserved); chosen by the selection rule.
- Selection rule (coherent joint variants only): eligible = within 1 SE of best mean log-loss AND within 1 SE of best mean ROC-AUC (means over CAD/LAD/LCX/RCA per outer fold); simplest eligible wins (order: none < temperature < vector, no-FE < FE). Independent/chains/LightGBM are benchmarks, not servable (no coherence guarantee or not the registered served model).
- Thresholds (max-F1, screening sens ≥ 0.90) from 5-fold OOF predictions of the final model on the 80% split; never from test.
- Classifier chains: 6 random orders over (CAD, LAD, LCX, RCA); each link sees X + 3-fold OOF probabilities of earlier links at training and predicted probabilities at inference (never labels). Base learner = elastic-net LR only (forest bases made chains ~4× slower than everything else combined).
- Non-joint benchmarks get P(≥2)/P(3VD)/pattern log-loss from the product (conditional-independence) distribution of their vessel probabilities.
- Served SHAP method `owen-partition-sampled`: Owen values for the features.yaml group hierarchy estimated with antithetic hierarchy-respecting permutations (exactly additive per permutation). Reason: `shap.PartitionExplainer` measured 0.7-3.3 s/patient here (budget 400 ms); our vectorized sampler with 4 permutations × 50 k-means centroids matched a 200-permutation reference with CAD rank correlation ≥ 0.994 and max abs error ≤ 0.013. Cross-checked against `shap.PermutationExplainer` in tests.
- Fast inference: `ml/fastforest.py` calls each tree's Cython `apply` directly on pre-normalized, class-aligned leaf tables (sklearn's per-tree input validation dominated single-row latency: ~0.1 s for 500 trees → ~4 ms); identical output to sklearn (tested ≤ 1e-12).
- Bootstrap intervals: 50 refits on bootstrap resamples of the 80% with 50 trees per forest (not 500) to keep artifacts small and latency low; all bootstrap trees flattened into one node table for one vectorized traversal; the served calibrator is applied to every member. Interval = 5th-95th percentile, widened if needed to contain the served point estimate (fraction of test patients needing this is reported in metrics.json `bootstrap_intervals`).
- Conformal: Mondrian (class-conditional) with OOF scores from the 5-fold refit on the training split (cross-conformal style); empty prediction sets are reported as "uncertain" (conservative, does not reduce coverage of the {0,1}/singleton sets).
- Applicability domain: Mahalanobis distance with Ledoit-Wolf covariance on standardized features, threshold = 99th percentile of out-of-fold training scores (in-sample would be optimistic).
- Similar patients: `index` in `SimilarResponse` is the 0-based row index in the original xlsx (all such rows are in the training split), so neighbours are traceable to the source.
- Counterfactuals: modifiable features only; numeric candidates = ref-range boundary and the midpoint towards it, within plausibility range; binary → the lower-risk value; weight is coupled to BMI (height fixed) and counts as one change with BMI; exhaustive search over ≤ 3 units, best result per number of changes, only if P decreases. Note "Model sensitivity, not treatment advice."
- Presets chosen by a fixed rule (median predicted P(CAD) among test patients with the requested true pattern; 10th percentile for low risk; most conformal-uncertain targets for "uncertain"), not by correctness; the out-of-domain preset is a clearly labelled synthetic perturbation of the low-risk patient with `true_pattern: null`.
- `python -m ml.train --quick` writes to the OS temp dir (not the repo); it exists for smoke/determinism tests only.
- metrics.json/thresholds/conformal/presets/global_explanations contain no timestamps or timings (timings → reports/train_log.txt) so reruns are byte-identical.

## Evidence

All numbers below are copied from `ml/artifacts/metrics.json` / `reports/*.md` produced by `uv run python -m ml.train`
(full settings) on 2026-10-12. Full tables: `reports/model_report.md`, `reports/data_audit.md`, `reports/leakage_ablation.md`.

- Audit: every data fact from the project brief recomputed from the xlsx and matches (13/13). Proxy scan (in-fold univariate AUC > 0.95): nothing flagged for CAD/LAD/LCX/RCA.
- Nested CV (5×5 outer, 3 inner; 242 training patients), mean ± SD ROC-AUC, served variant `joint_vector`: CAD 0.923 ± 0.044, LAD 0.850 ± 0.057, LCX 0.711 ± 0.053, RCA 0.740 ± 0.051, ≥2 vessels 0.745 ± 0.051, 3VD 0.745 ± 0.066; mean log-loss 0.489 (SE 0.007), Brier 0.162.
- Benchmarks under the same folds: independent ensembles mean log-loss 0.501, mean AUC 0.807 (joint_vector 0.806), coherence violations 5.9% ± 3.3% of patients; classifier chains log-loss 0.544, coherence violations 23.4%; LightGBM+temperature log-loss 0.517. All joint variants: 0 violations (by construction).
- Calibration: vector scaling beats none/temperature (log-loss 0.489 vs 0.500/0.502); temperature alone fitted T≈1, i.e. no global over/under-confidence.
- Engineered features: paired Δ log-loss −0.0005, Δ AUC +0.0024 (joint_fe_vector vs joint_vector), within noise → not kept (simpler model within 1 SE).
- Locked test split (n=61, evaluated once; 2,000 bootstrap 95% CIs), served model ROC-AUC: CAD 0.920 [0.832, 0.981], LAD 0.842 [0.734, 0.930], LCX 0.692 [0.559, 0.817], RCA 0.685 [0.544, 0.814], ≥2 0.690 [0.551, 0.819], 3VD 0.729 [0.582, 0.868]. Independent benchmark: CAD 0.917, LAD 0.820, LCX 0.716, RCA 0.713. DeLong served − independent: all |Δ| ≤ 0.027, all p ≥ 0.22 (no significant difference in discrimination; the joint model's gain is coherence and pattern log-loss 1.611 vs 1.698).
- Coherence on test: served 0/61, independent 0/61 (the independent model's violations show up in CV OOF: 5.9%).
- Conformal (90% target) test coverage: CAD 0.869, LAD 0.885, LCX 0.918, RCA 0.902; mean set size 1.10 / 1.33 / 1.62 / 1.56 (LCX/RCA are often "uncertain", which is honest given AUC ~0.7).
- Test calibration slope / ECE: CAD 1.12 / 0.064, LAD 1.25 / 0.084, LCX 1.23 / 0.075, RCA 0.96 / 0.119.
- Subgroups (CV OOF AUC): performance drops for age ≥ 65 (CAD 0.825, LCX 0.516, RCA 0.605); report this as a limitation.
- Learning curve (CAD AUC 0.903 → 0.925 from 39 → 194 training patients; LCX/RCA still rising): more data would help vessel-level prediction.
- Explanations: additivity max error 5.6e-16 (global run on 242 patients); deletion curve: removing top-|Owen| features moves P(CAD) by 0.12 after 1 feature vs 0.01 for random order. Agreement with test-set permutation importance: CAD ρ = 0.59 (top-10 overlap 8), vessels ρ ≤ 0.22 (permutation importance on 61 patients is noisy for weak targets).
- Leakage ablation (RF, 10×3 CV, all 303): CAD AUC 0.968 (SMOTE + selection before CV) vs 0.925 (correct, in-fold); LCX 0.842 vs 0.718; RCA 0.851 vs 0.716. SMOTE before CV alone inflates most (CAD 0.978).
- Determinism: two full `python -m ml.train` runs produced byte-identical files for all 14 artifacts (metrics.json, joblib models, presets, …); `ml/tests/test_determinism.py` re-checks with quick settings.
- Train time: 395 s and 327 s (two runs; nested CV 243 s). Artifacts: 13 MB (`bootstrap_ensemble.joblib` 11 MB, `joint_model.joblib` 2.3 MB).
- Latency (pytest `backend/tests/test_api.py::test_latency`, uncached, after warm-up, M2, background load average 11-35 from iCloud sync + other agent): analyze p95 22.7-22.8 ms, explain p95 253-328 ms, HTTP analyze p95 38-70 ms.
