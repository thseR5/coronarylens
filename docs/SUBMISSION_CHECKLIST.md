# Submission checklist (definition of done)

Evidence = a file, a test that runs in `make test` / `make test-e2e`, or a measured artifact. Status as of the last commit on `main`.

## Challenge requirements

| Requirement (challenge text) | Status | Evidence |
|---|---|---|
| Classify overall CAD; predict LAD, LCX, RCA stenosis | Done | One joint 8-pattern model, all outputs are marginals (`ml/joint.py`, `ml/procedure.py`); `test_coherence_test_split`, `test_coherence_random_inputs` |
| Use demographic, exam, ECG, lab and echo features | Done | 54 inputs in `config/features.yaml`; `test_verified_data_facts`, `test_config_extensibility` |
| Exclude LAD, LCX, RCA and Cath from all model inputs | Done | `ml/leakage.py` guard in training, evaluation and inference; `test_forbidden_columns_rejected`, `test_training_rejects_feature_mapped_to_label`, `test_inference_rejects_forbidden`, API `test_invalid_input_422` (HTTP 422) |
| Evaluate with accuracy, precision, recall, F1, ROC-AUC | Done | Nested-CV table in `docs/MODEL_CARD.md` and the performance tab (Headline); locked test set with bootstrap CIs; `test_metrics_match_sklearn` |
| Validation methodology | Done | 10×5 nested CV with selection inside folds, 748 procedures, corrected statistics (`ml/sprint/`), leakage ablation; `test_corrected_variance_formula`, `test_paired_verdicts` |
| Interactive 3D heart/torso in the browser (R3F / WebGL) | Done | `frontend/src/three/` with BodyParts3D glTF; e2e `full stack: preset → API → dashboard + 3D agree` |
| Colour of the LAD, LCX, RCA nodes updates from predicted probabilities | Done | `AnatomyModel.tsx`; e2e compares 3D labels with a direct API call |
| Rotate, zoom, select regions for vessel-specific risk | Done | Orbit controls, camera presets, click a vessel → explanation switches; e2e vessel selection and hover sync |
| Dashboard: CAD status, vessel probabilities, SHAP/LIME, measurements beside their contribution | Done | `RiskSummary`, `VesselCards`, `WhyPanel` (Owen-value SHAP), `MeasurementsTable`; `test_shap_additivity_and_determinism`, vitest "every input feature in the config appears in the measurements table" |
| Open-source anatomy | Done | BodyParts3D 4.0 (CC BY 4.0), FMA IDs in `frontend/src/three/LICENSES.md`; `LICENSES.md` |
| Clear, visible disclaimer | Done | Banner, footer, exported report; e2e checks the banner text and the report |
| Responsive 3D without a dedicated GPU | Done | 46–60 FPS with the GPU disabled (SwiftShader, low-power mode), `docs/REPORT.md` §7; e2e `3D scene` FPS test |
| Add features, models or anatomy without redesign | Done | Four YAML files drive pipeline, API and UI (`docs/ARCHITECTURE.md`); `test_config_extensibility` |
| Consistent mapping between outputs and displayed LAD/LCX/RCA | Done | vitest "every vessel target maps to exactly one role:vessel structure", AHA 3D ↔ bullseye mapping tests |

## Deliverables

| Deliverable | Status | Evidence / what is left |
|---|---|---|
| Web prototype (3D viewer + ML backend) | Done | `make dev` / `make serve` / `docker compose up`; fresh-clone quickstart verified |
| Trained pipeline (code + weights) | Done | `ml/` and committed `ml/artifacts/`; `make train` reproduces; `test_train_deterministic` |
| Explanation dashboard | Done | Patient analysis and Model performance tabs |
| Documentation ≤ 6 pages | Done | `docs/REPORT.pdf`, 5 pages (`tools/build_report.py` prints the page count) |
| 3–10 min YouTube demo | **You** | Script and checklist in `docs/DEMO_SCRIPT.md`; click path verified by e2e at 1920×1080 and 1440×900 |
| Public GitHub repo with README (setup, prerequisites, run steps) | **Waiting for your go-ahead** | README ready; `tools/prepublish_check.sh` must pass first |
| Live demo (optional) | **Waiting for your go-ahead** | `deploy/hf-space/DEPLOY.md` |

## Quality gates (run before publishing)

- `make lint && make test && make test-e2e`: all green on the last commit.
- `tools/prepublish_check.sh`: clean tree, LICENSE author filled in, no assistant mentions in files or history, no private files or bundles tracked, no blob over 50 MB, docs numbers match artifacts.
- `uv run python tools/check_numbers.py`: every number in README, report, Devpost text, Q&A, demo script, model card and dashboard text is backed by a measured artifact.

## Your manual steps

1. **Licence name.** Done: `LICENSE` and `docs/MODEL_CARD.md` name TADI HEMA SAI ESWAR REDDY.
2. **GitHub CLI** (needed for "publish repo"): `brew install gh`, then `! gh auth login` (GitHub.com → HTTPS → log in with a browser).
3. **Record the video** with `docs/DEMO_SCRIPT.md` and its checklist (English audio; add English subtitles if needed).
4. **Upload to YouTube** → Visibility **Unlisted** (not Private) → language English → paste the GitHub link in the description → open the link in a private window to confirm it plays without signing in.
5. **Devpost** (project page):
   - Paste the sections of `docs/DEVPOST.md` (name, tagline, Inspiration, What it does, How we built it, Results, Challenges, Accomplishments, What we learned, What's next, Built with).
   - Add the GitHub repo link, the YouTube link and (if deployed) the live demo link; attach or link `docs/REPORT.pdf`.
   - Select Track A. Add every teammate: each needs a Devpost account and must accept the invite; use real full names.
   - Open every link in a private browser window.
6. **Submit before the deadline** shown on the Devpost page (you mentioned 15 October earlier; confirm the exact IST time there). Submit at least a few hours early; Devpost allows edits until the deadline.
