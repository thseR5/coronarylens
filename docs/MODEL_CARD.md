# Model card: CoronaryLens served model

> Decision support and educational use only. Not a substitute for formal diagnostic imaging (e.g., coronary angiography) or clinical judgment. Not a medical device.

## Model details

- **Type:** joint multiclass model over the 8 LAD/LCX/RCA stenosis patterns. Served outputs are marginals of one calibrated distribution: P(LAD), P(LCX), P(RCA), P(CAD) = 1 − P(no stenosis), P(≥ 2 vessels), P(3-vessel disease).
- **Served procedure:** a selection procedure validated by nested CV (`ml/procedure.py`, candidates in `config/models.yaml → served_procedure`). On the full training split it selected **CatBoost (depth 4) + CatBoost (depth 3) + RBF SVM, averaged, vector-scaling calibration**.
- **Version:** `procedure-sprint_procedure-73934324-ae7f75d9` (see `ml/artifacts/model_meta.json`; data SHA-256 starts `73934324`).
- **Libraries:** Python 3.12, scikit-learn 1.9.1, CatBoost 1.2.10 (pinned in `uv.lock`).
- **Authors / contact:** the CoronaryLens hackathon team.

## Intended use

- Education and research demonstration of how clinical, ECG, laboratory and echocardiographic findings relate to angiographic CAD and vessel-level stenosis in one published cohort.
- **Out of scope:** diagnosis, triage or treatment decisions; populations unlike the training cohort; replacing angiography or CT coronary angiography.

## Inputs and outputs

- **Inputs:** 54 features (demographics, risk factors, symptoms and examination, ECG, labs, echo) defined in `config/features.yaml`; missing inputs are imputed with training medians/modes and listed in the response.
- **Never inputs:** LAD, LCX, RCA, Cath (enforced in training, evaluation and inference).
- **Outputs:** probabilities with 90% intervals (50 bootstrap refits with the served hyperparameters; 5th–95th percentile spread centred on the served estimate), status (likely / unlikely only when the 90% conformal set is a single outcome agreeing with the threshold decision, otherwise uncertain), operating thresholds (max-F1, or screening with sensitivity ≥ 0.90 and never above max-F1), applicability-domain flag, Owen-value SHAP explanations.

## Training data

Extension of Z-Alizadeh Sani (303 patients, single centre, Iran; see `docs/DATASHEET.md`). Stratified 80/20 split on the vessel pattern: 242 training, 61 locked test patients.

## Evaluation

Primary: nested CV of the whole selection procedure (10 repeats × 5 outer folds; inner 5-fold out-of-fold predictions; selection inside every outer fold); CIs with the Nadeau–Bengio correction.

| Target | ROC-AUC [95% CI] | PR-AUC | Brier | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| CAD | 0.934 [0.896, 0.973] | 0.972 | 0.089 | 0.880 | 0.909 | 0.926 | 0.916 |
| LAD | 0.866 [0.820, 0.911] | 0.894 | 0.144 | 0.805 | 0.789 | 0.913 | 0.845 |
| LCX | 0.708 [0.647, 0.769] | 0.578 | 0.204 | 0.630 | 0.515 | 0.848 | 0.638 |
| RCA | 0.737 [0.679, 0.796] | 0.583 | 0.194 | 0.654 | 0.527 | 0.848 | 0.647 |
| ≥ 2 vessels | 0.766 [0.711, 0.820] | | | | | | |
| 3 vessels | 0.755 [0.689, 0.821] | | | | | | |

Pattern log-loss 1.508. Coherence violations (P(CAD) below a vessel probability): 0% on all 50 outer folds.

Locked test set (n = 61; 44 CAD), ROC-AUC [95% bootstrap CI]: CAD 0.898 [0.80, 0.97], LAD 0.829 [0.71, 0.93], LCX 0.761 [0.64, 0.87], RCA 0.661 [0.51, 0.80]. Calibration slope / ECE: CAD 0.95 / 0.088, LAD 1.03 / 0.092, LCX 1.34 / 0.140, RCA 0.67 / 0.082. Conformal coverage (90% target): CAD 0.87, LAD 0.84, LCX 0.92, RCA 0.87.

**Subgroups (nested CV, ROC-AUC CAD / LCX / RCA):** men (n = 138) 0.905 / 0.718 / 0.689; women (n = 104) 0.960 / 0.693 / 0.792; age < 50 (n = 47) 0.911 / 0.740 / 0.832; 50–64 (n = 119) 0.949 / 0.707 / 0.751; ≥ 65 (n = 76) 0.840 / 0.539 / 0.588.

## Explanations

Owen values (SHAP with clinical groups → features) on the exact displayed output, 4 antithetic permutations, 50 real training patients as background; additivity exact; deletion-curve fidelity 0.217 vs 0.051 random. What-if sweeps of 11 features × 5 presets: P(CAD) never moves against the clinically expected direction.

## Limitations and risks

- Single-centre, small cohort; no external validation; test set small (wide CIs).
- LCX and RCA discrimination is modest (AUC ≈ 0.71–0.74); about half of test patients get an "uncertain" LCX/RCA status.
- The model tells whether CAD is present far better than which artery: within-CAD AUC LAD 0.686, LCX 0.515, RCA 0.566; in single-vessel LCX/RCA patients the true vessel was never ranked highest. Specificity (nested CV): CAD 0.768, LAD 0.655, LCX 0.491, RCA 0.537.
- Weakest in patients aged ≥ 65.
- Vessel-level labels only: no lesion location, severity grading or left-main label.
- Coronary territories are typical (AHA assignment), not patient-specific.
- Associations, not causes: SHAP values and counterfactuals describe model behaviour.

## Ethical considerations

Not validated for clinical use. The disclaimer appears in the UI banner, the footer and every exported report. No personal data beyond the public, de-identified dataset is processed or stored.
