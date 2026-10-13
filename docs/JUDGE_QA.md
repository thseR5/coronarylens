# Judge Q&A

Short answers; every number is in `ml/artifacts/metrics.json` or the linked file.

**Why is your accuracy lower than papers that report 97–99% on this dataset?**
Because our numbers come from patients the model never saw during any modelling step. Our leakage ablation runs the same random forest two ways. With oversampling (SMOTE) and feature selection on the full dataset before cross-validation, CAD AUC is 0.968, LCX 0.842 and RCA 0.851. With both steps inside each training fold, the numbers are 0.930, 0.714 and 0.715. Most of the inflation comes from SMOTE before CV: feature selection before splitting alone gives only 0.931 / 0.741 / 0.726. So we do not claim that every high published number is leakage. We report what our own protocol gives: CAD 0.934 in nested CV.

**How do you prove there is no leakage?**
LAD, LCX, RCA and Cath are rejected as inputs in training, evaluation and inference (unit tests; the API returns HTTP 422). Every preprocessing, selection, calibration and threshold step runs inside the CV folds. A scan found no feature with within-fold AUC above 0.95 for any target. The 61 test patients were split off before any modelling.

**How many models did you try, and doesn't that inflate your results?**
We screened 748 procedures on the training split (learners, calibrations, ensembles, priors, stacking). Ten finalists, chosen by a rule written before we read the results, then went through 10×5 nested cross-validation. Model selection happened inside every outer fold, so the reported performance accounts for that search. The whole log, including failures, is in `docs/EXPERIMENTS.md`.

**Why a joint model instead of four classifiers?**
Overall CAD means at least one artery is narrowed, so P(CAD) can never be lower than any vessel's probability. Separate models break this rule for 32.1% of patients (four logistic regressions), 24.8% (four regularised ones), 23.4% (classifier chains) and 5.4% (a strong LR + RF + ExtraTrees ensemble, on the same folds as ours). Our model breaks it for 0%, by construction and measured. To be fair, the ensemble's contradictions are small (mean gap 0.03), and a one-line patch, P(CAD) = max(P(CAD), max vessel), hides them without changing its CAD log-loss (0.341 → 0.341). The joint model's real advantages are that it gives one distribution, so ≥2-vessel and 3-vessel risks are consistent with everything else, and that its risk estimates are better calibrated: on identical folds it matches the ensemble's AUC with better pattern log-loss (p = 0.0003) and Brier score (p = 0.049). Against the patched ensemble, its CAD Brier score is still better (0.089 vs 0.102, p = 0.034).

**Why is the log-loss gain significant but the AUC gain isn't?**
AUC only measures ranking. Log-loss and Brier also measure whether the probabilities are right, and the joint model with calibration improves exactly that. Small AUC gains (+0.007 CAD, +0.015 LAD over our first model) have p = 0.21 and 0.14. Repeated-CV folds share training data, so we use the Nadeau–Bengio corrected t-test, which makes the standard error about 3.7 times larger than the naive one. Log-loss improves by 0.067 (t = −3.29, p = 0.002, better in 46 of 50 folds). That matters here because the 3-D colours are probabilities.

**Can the model tell which artery is diseased?**
Only weakly, and we say so in the app, above the vessel cards: "Vessel-level estimates are less certain than overall CAD (nested-CV AUC: CAD 0.93 · LAD 0.87 · LCX 0.71 · RCA 0.74). Use them as supporting context." Vessel probabilities mostly carry overall CAD risk. Among patients who have CAD, the within-CAD AUC is LAD 0.686, LCX 0.515 and RCA 0.566. In single-vessel LCX or RCA patients, the true artery was never the highest-probability vessel (0/100 and 0/140 fold-instances), while LAD-only patients were found 458 of 460 times. Dedicated models trained only on CAD patients reached at most 0.62 for LCX in exploratory CV. The data tells you whether coronary disease is present far better than where it is.

**Why are LCX and RCA so hard?**
The inputs are mostly global findings: symptoms, risk factors, labs, ECG and echo. They say a lot about whether a patient has CAD, and less about which artery is affected. ECG and wall-motion clues are regional but imprecise, and coronary dominance varies between people. LCX and RCA disease are also less common than LAD disease here (119 and 114 vs 177 patients), and more often occur together with other vessels. After 748 procedures their AUC stayed at 0.71–0.74, and about half of test patients get an "uncertain" LCX/RCA status. We show that rather than hide it.

**Why did CAD AUC on the test set drop at the second look (0.920 → 0.898)?**
The test set has 61 patients and only 17 without CAD, so its 95% CIs are wide (CAD 0.898 [0.80, 0.97]). Both models are well inside each other's intervals. LCX moved the other way (0.692 → 0.761). The new model was chosen on nested CV (50 outer folds), never on the test set. Switching back because of the test result would mean selecting on test data. We report both looks.

**Isn't 303 patients too small?**
It is small, which is why we use nested CV with corrected CIs, report Brier and log-loss next to AUC, show uncertainty in the UI, and name the limits: no external validation, a single centre, and a weaker subgroup aged 65 and over (CAD AUC 0.840, n = 76).

**Why didn't you use TabPFN?**
We considered it. Its current weights require accepting a separate licence and an API token. Its CPU inference is also too slow for our live explanations: each explanation evaluates the model on about 5,000 coalition rows, within a 700 ms budget. Our served model explains in 57 ms median (p95 91 ms) per request in-process.

**What happens at an ejection fraction of 55%?**
In the what-if check, P(LAD) drops in one step as EF crosses about 52–55%, the edge of the normal reference range. The step is 0.03–0.11 depending on the patient (0.072 for the three-vessel preset, `reports/whatif/whatif.json`). The tree-based learners split there. The step is below our 0.15 jump threshold, and P(CAD) still moves in the expected direction (higher EF → lower risk). It is a real artefact of tree models on 242 patients, visible in the performance tab.

**Why does the regional wall-motion score lower LCX/RCA probability for some patients?**
Our expected direction in `features.yaml` is for overall CAD. A wall-motion abnormality raises P(CAD) everywhere in our sweep. At vessel level, the model can attribute it to one artery and lower the others slightly (by up to 0.048 in our presets). That is plausible, because wall-motion abnormalities are regional. Six such small vessel-level reversals are listed in the performance tab.

**Are the 3-D territories real for this patient?**
No. They show expected supply: the standard AHA 17-segment assignment (Cerqueira 2002), labelled "illustrative; coronary dominance varies between people". The 3-D segments and the bullseye use the same table (tested). An optional nearest-artery estimate agrees with the AHA assignment on 14 of 17 segments. It was adjusted toward the AHA convention during development, so that is alignment, not independent validation.

**Why is the left main grey?**
The dataset has no left-main label, so we show it as "not modelled" instead of inventing a value.

**What does "uncertain" mean statistically?**
We show "likely" or "unlikely" only when the class-conditional (Mondrian) conformal prediction set at 90% target coverage contains a single outcome *and* that outcome agrees with the threshold decision. Otherwise the vessel is "uncertain": the set contains both outcomes, or it disagrees with the threshold. Every output is also forced to "uncertain" when the patient is outside the training distribution or 10 or more inputs were not entered (imputed). Test coverage was 0.84–0.92. On the test set, 18% (CAD), 26% (LAD), 54% (LCX) and 52% (RCA) of patients were shown as uncertain.

**How are the intervals computed?**
50 bootstrap refits of the served learners with the served hyperparameters. The 5th–95th percentile spread is centred on the served estimate. The raw percentile interval already contained the served estimate for 95–100% of test patients. Mean width is 0.18 for CAD and 0.27 for LCX.

**Why no SMOTE?**
It distorts calibrated probabilities, and our colours are probabilities. We keep the natural class balance, calibrate, and choose thresholds on out-of-fold predictions.

**Do your explanations explain the model you show?**
Yes. SHAP (Owen values over clinical groups) runs on the exact displayed function, including calibration and the step from pattern to vessel probabilities. Base value + contributions equals the displayed probability exactly. Removing the top-ranked features changes the prediction four times more than removing random ones (0.217 vs 0.051). Across 8 random seeds, no contribution moves by more than 0.025 (SD) and the top-5 features overlap by at least 85% (`reports/explain_stability.json`).

**How fast is it without a GPU?**
On the production build on a laptop, colours update at p95 226 ms after a slider change and the explanation at p95 360 ms. The 3-D scene ran at 46–60 FPS with the GPU disabled (software rendering).

**How would you add a new vessel or feature?**
Add it to the YAML config (`features.yaml` or `anatomy.yaml`) and retrain or rebuild. The input form, API schema, explanations and 3-D mapping follow automatically, and tests check the mapping. A new vessel target grows the joint pattern space to 2^k.

**Is this a medical device?**
No. It is for education and research. The disclaimer is in the banner, the footer and every exported report, and counterfactuals are labelled "model sensitivity, not treatment advice".
