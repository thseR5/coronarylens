# Datasheet: data used by CoronaryLens

Format follows Gebru et al., "Datasheets for Datasets" (shortened). Facts below were re-computed from the file (`make audit` → `reports/data_audit.md`).

## Motivation and source

- **Dataset:** Extension of Z-Alizadeh Sani dataset (UCI Machine Learning Repository, id 411; doi:10.24432/C5461K).
- **Creators:** Roohallah Alizadehsani, Mohamad Roshanzamir, Zahra Alizadeh Sani. **Licence:** CC BY 4.0.
- **Purpose (by the creators):** machine-learning research on CAD and vessel-level stenosis diagnosis.
- **File used:** `data/raw/extention of Z-Alizadeh sani dataset.xlsx`, SHA-256 `739343245c2ba578b541370217531750d8e936022f928b83e0d91756caa3ff0b`.

## Composition

- 303 patients, 59 columns, **no missing values**. Single centre (Iran); collection details beyond the UCI description are not available to us.
- **Targets:** `Cath` (CAD vs Normal; 216 CAD), `LAD` (177 stenotic), `LCX` (119), `RCA` (114). There is no separate CAD column: `Cath` is the CAD label.
- **Vessel patterns (LAD/LCX/RCA):** 000: 86 · 100: 57 · 111: 63 · 110: 33 · 101: 24 · 001: 17 · 010: 13 · 011: 10.
- **Label consistency:** CAD equals "any vessel stenotic" for 302/303 patients; one patient has `Cath` = Normal with LAD stenotic (kept; `Cath` remains ground truth).
- **Feature groups:** demographics and anthropometrics (age, sex, weight, height, BMI), risk factors and history, symptoms and examination, ECG, laboratory, echocardiography (`config/features.yaml`).
- **Absent despite appearing in the challenge primer:** cardiac enzymes and total cholesterol. Blood pressure is a single value (no diastolic). There is no left-main label.

## Quirks handled

- `Sex` uses `Male` / `Fmale`; binary columns mix `Y`/`N` strings and 0/1 integers.
- `Exertional CP` is constant (all N) and is dropped.
- Very rare positives: CHF 1, LowTH Ang 2, CVA 5, Weak Peripheral Pulse 5, CRF 6, Function Class = 1 once.
- `BBB` ∈ {N, LBBB, RBBB} (one-hot); `VHD` ∈ {N, mild, Moderate, Severe} (ordinal); `Region RWMA` 0–4 (ordinal; exact coding not documented, used as given).
- `Length` is height in cm; `PR` is pulse rate; `CR` is creatinine.

## Use in this project

- Stratified 80/20 split on the 8 patterns (seed 42): 242 training, 61 test patients; indices in `ml/artifacts/split.json`.
- All modelling choices were made with the training split only; the test split was evaluated twice (before and after the model search) and never used for selection.
- No resampling or synthetic patients in training (no SMOTE). One clearly labelled synthetic out-of-domain preset exists for the demo only.

## Known limitations and biases

Single-centre cohort with a high CAD prevalence (71%) from patients referred for angiography; results may not transfer to screening populations or other health systems. Age and sex subgroups are small (e.g., 76 training patients aged ≥ 65).
