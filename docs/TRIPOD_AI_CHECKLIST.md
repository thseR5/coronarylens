# TRIPOD+AI checklist (development and internal evaluation)

Items and numbering from Collins GS et al., *TRIPOD+AI statement*, BMJ 2024;385:e078378 (Table 2). "Report" = `docs/REPORT.pdf`; other files are in this repository. Where the public dataset does not document an item, we say so.

| Item | Topic | Where / how addressed |
|---|---|---|
| 1 | Title | Report title: coronary risk model (development + internal evaluation), outcome = angiographic CAD and LAD/LCX/RCA stenosis |
| 2 | Abstract | Report "Summary: four results"; README |
| 3a | Healthcare context, existing models | Report §3 "Comparison with published work"; README |
| 3b | Target population, intended use, users | Model card "Intended use": education/research only; users = students, clinicians exploring model behaviour |
| 3c | Known health inequalities | Datasheet "Known limitations and biases"; subgroup results (sex, age) in Report §3 |
| 4 | Objectives | Report summary: development and internal validation of a coherent multi-output model |
| 5a | Data sources, representativeness | Datasheet; single-centre angiography cohort (UCI id 411), referral population with 71% CAD |
| 5b | Dates of data collection | Not documented in the public dataset (donated 2017); stated as unknown in the datasheet |
| 6a | Setting, number/location of centres | Single centre, Iran (datasheet) |
| 6b | Eligibility criteria | Not documented in the public dataset; all 303 records used |
| 6c | Treatments | Not recorded in the dataset; not handled |
| 7 | Data preparation | Report §1; `ml/ingest.py`; audit `reports/data_audit.md`; same encoding for all groups |
| 8a | Outcome definition | Angiographic CAD (`Cath`) and ≥50% stenosis per vessel (dataset definition); diagnostic, no time horizon (Report §1) |
| 8b | Outcome assessors | Not documented in the public dataset |
| 8c | Blinding of outcome assessment | Not documented in the public dataset |
| 9a | Choice of predictors | All 54 available clinical features except the constant `Exertional CP`; outcome columns excluded (Report §1) |
| 9b | Predictor definitions | `config/features.yaml` (units, ranges, groups); datasheet "Quirks" |
| 9c | Predictor assessors | Not documented in the public dataset |
| 10 | Sample size | Fixed by the dataset (303); handled by nested CV + corrected CIs; small test set acknowledged (Report §3, §8) |
| 11 | Missing data | None in the dataset; at inference missing inputs are imputed with training medians/modes and listed (model card) |
| 12a | Use of data / partitioning | Stratified 80/20 split; nested CV on the 80% (Report §3) |
| 12b | Predictor handling | Binary/ordinal/one-hot encoding; scaling inside learner pipelines (Report §1; `ml/sprint/learners.py`) |
| 12c | Model type, building steps, tuning, internal validation | Report §2–3; `docs/EXPERIMENTS.md` (748 screened procedures, finalist rule, nested selection) |
| 12d | Heterogeneity across clusters | Not applicable (single centre) |
| 12e | Performance measures and plots | ROC-AUC, PR-AUC, Brier, log-loss, threshold metrics, calibration slope/ECE, reliability, decision curves, coherence; Nadeau–Bengio corrected CIs (Report §3–4) |
| 12f | Model updating after evaluation | None after the test evaluation |
| 12g | How predictions were calculated | `ml/predictor.py`, FastAPI `/api/analyze` (`docs/API.md`) |
| 13 | Class imbalance | No resampling (no SMOTE); calibrated probabilities; thresholds from out-of-fold predictions (Report §1, §2) |
| 14 | Fairness | Subgroup performance by sex and age reported; weakest group (age ≥ 65) named as a limitation |
| 15 | Model output | Probabilities with intervals, conformal status, max-F1 and screening thresholds (model card) |
| 16 | Training vs evaluation data differences | Random split of one cohort; no differences by design |
| 17 | Ethical approval | Public, de-identified data under CC BY 4.0; original approval not documented in the public release |
| 18a | Funding | None (hackathon project) |
| 18b | Conflicts of interest | None declared |
| 18c | Protocol | No separate protocol; pre-registered selection and reading rules are in `docs/EXPERIMENTS.md` (written before results) |
| 18d | Registration | Not registered |
| 18e | Data sharing | Public dataset (UCI id 411, doi:10.24432/C5461K); split indices in `ml/artifacts/split.json` |
| 18f | Code sharing | This repository (training, evaluation, app) |
| 19 | Patient and public involvement | None |
| 20a | Participant flow | 303 → 242 development / 61 test; outcomes per split in Report §3 and `metrics.json` |
| 20b | Characteristics | `reports/data_audit.md` (prevalence, patterns, feature ranges) |
| 20c | Development vs evaluation distributions | Test pattern counts in `metrics.json`; stratified split |
| 21 | Participants and events per analysis | 242 (training: 172 CAD), nested folds ≈ 194 / 48; test 61 (44 CAD) |
| 22 | Full model specification | Committed artifacts (`ml/artifacts/*.joblib`, calibrator), `config/models.yaml`, API |
| 23a | Performance with CIs, subgroups | Report Tables 1–2, test-set CIs, subgroup paragraph |
| 23b | Heterogeneity across clusters | Not applicable |
| 24 | Model updating results | Not applicable |
| 25 | Interpretation | Report summary and §3 ("ceiling held" for LCX/RCA) |
| 26 | Limitations | Report §8; model card |
| 27a | Poor or missing inputs at use | Imputation + listed imputed fields; out-of-range flags; applicability-domain warning |
| 27b | User interaction and expertise | Users enter inputs; disclaimer; no clinical use intended |
| 27c | Next steps | External validation on another centre; vessel-level imaging data; prospective evaluation |
