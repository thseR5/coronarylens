# Demo video script (target 6:30, limit 3–10 min)

**Setup before recording**
- Run `make dev`, open http://localhost:5173 in Chrome at 1600×1000 (or full-screen 1440p), dark OS theme.
- Close other tabs, and refresh once so the explanation worker is warm.
- Default state: Heartbeat on, Labels on, Ribs on, Territories on (AHA 17-segment).
- Screen recorder at 30–60 fps, microphone on. Keep the pointer slow; pause about 1 s after each click.
- Every number below is on screen in the app. Say them as written.

**Caption style:** short lower-third text, white on dark, 2–4 s each.

---

## 0:00–0:35 · Hook: a model that cannot contradict itself

**Clicks:** open the **Model performance** tab and scroll to **"Coherence · patients where P(CAD) < some vessel's P(stenosis)"**.

**Say:** "If a model says a patient's LAD artery is 70% likely to be narrowed, the chance that the patient has coronary artery disease at all must be at least 70%. Most models break this rule. Four separate logistic regressions break it for 32% of patients. A strong ensemble of separate models still breaks it for about 5%. CoronaryLens breaks it for nobody: every number comes from one model of all eight possible vessel patterns. Its accuracy is the same, and its risk estimates are better."

**Captions:**
- "Naive models: 32.1% of patients contradicted"
- "Strong ensemble: 5.4–5.9%"
- "CoronaryLens: 0% by construction"
- "Equal AUC · better log-loss (p = 0.0003)"

## 0:35–0:50 · Disclaimer and data

**Clicks:** point at the yellow banner at the top.

**Say:** "This is decision support and education only, not a diagnosis. It uses 303 patients from the public Z-Alizadeh Sani dataset. The four answers we predict, overall CAD and the three vessels, are never used as inputs."

**Caption:** "Decision support / education only · 303 patients · labels never used as inputs"

## 0:50–1:40 · Real anatomy in 3D

**Clicks:**
1. Click **Patient analysis**.
2. Drag to rotate the heart slowly.
3. Click camera presets **AP** → **LAO 30°** → **RAO 30° caudal 20°**.
4. Toggle **Ribs** off.
5. Hover the grey vessel at the top.

**Say:** "The anatomy is BodyParts3D, a medical 3D atlas. Each artery is mapped to the model by its anatomical ID: LAD, LCX and RCA. These buttons are standard angiography views. The grey vessel is the left main, which the data does not label, so we leave it uncoloured. The heart beats at the patient's own pulse rate."

**Captions:**
- "BodyParts3D anatomy · LAD / LCX / RCA by FMA ID"
- "Angiographic views: AP · LAO · RAO"
- "Grey left main = not modelled"

## 1:40–2:40 · Three-vessel patient: territories and bullseye

**Clicks:**
1. **Preset patients…** → **Three-vessel disease**.
2. Wait for the vessel colours to change.
3. Point at the CAD gauge, then the LAD / LCX / RCA cards and the "Multi-vessel (≥2)" and "Three-vessel" tiles.
4. Scroll to the **AHA 17-segment map**.
5. Hover segment **7**, then hover a segment on the 3D heart; show that it lights up in the bullseye.

**Say:** "This real test patient has disease in all three arteries. The colour of each vessel is its probability of narrowing, and the labels show the uncertainty interval. The heart wall is split into the standard 17 cardiology segments, coloured by the artery that usually supplies them. The 3D wall and this bullseye use the same table, so hovering one highlights the other."

**Captions:**
- "Vessel colour = P(stenosis) · interval in label"
- "17 AHA segments · 3D ↔ bullseye in sync"
- "Expected supply territory (illustrative)"

## 2:40–3:40 · Why: per-vessel explanation

**Clicks:**
1. Click the **LAD** vessel on the 3D heart. The camera flies to it and the SHAP tab switches to LAD.
2. Point at the waterfall, from the cohort average to this patient.
3. Toggle **By group**, then toggle it back.
4. Scroll to **Measurements · contribution to LAD stenosis** and point at two flagged values.
5. Click **Patient view** to show the plain-language sentences.

**Say:** "Click any artery to see why. These are SHAP values computed on exactly the number shown, from the cohort average up to this patient. Red raises the risk and blue lowers it. Every measurement sits next to its own contribution, with out-of-range values flagged. There is also a plain-language version for patients. These describe what the model does, not what causes disease."

**Captions:**
- "Click a vessel → its own explanation"
- "Owen-value SHAP on the displayed probability"
- "Model behaviour, not causation"

## 3:40–4:30 · What-if: live recolour

**Clicks:**
1. In **What-if**, click **Pin current as baseline**.
2. In the left panel, click **Typical chest pain** to switch it to *No*. Watch the vessels recolour and the deltas appear on every card.
3. Drag the **Age** slider from 75 down to 45, slowly.
4. Click **Restore baseline**.

**Say:** "Pin the patient, then change one thing. Without typical chest pain, every probability drops, the arteries recolour, and each card shows how much it changed. Dragging age updates live: colours change in under 200 milliseconds, the explanation in about a quarter of a second."

**Captions:**
- "Live update: colours p95 175 ms · explanation p95 241 ms"
- "Deltas vs pinned baseline"

*(Do not sweep ejection fraction across 55% here. The model has a small step at the edge of the reference range. It is shown on purpose in the Q&A.)*

## 4:30–5:10 · Honest uncertainty

**Clicks:**
1. **Preset patients…** → **Uncertain case**. Point at the moving stripes on a vessel and the **Uncertain** chip.
2. **Preset patients…** → **Synthetic out-of-domain**. Point at the yellow **"Outside training distribution"** badge.

**Say:** "When the data cannot separate yes from no for an artery, the vessel gets moving stripes and an 'uncertain' label, so you can see it without relying on colour. For patients unlike anyone in the training data, the app warns that its estimates are unreliable."

**Captions:**
- "Stripes = conformal 'uncertain' (90% sets)"
- "Out-of-distribution warning"

## 5:10–6:00 · How we validated

**Clicks:**
1. Open **Model performance**.
2. Show the **Headline** table.
3. Scroll to **Model search · 748 procedures screened**.
4. Scroll to **Leakage ablation**.

**Say:** "All results come from nested cross-validation: every modelling choice is made inside each training fold and tested on patients it never saw. We screened 748 approaches and picked finalists by a rule written beforehand. Overall CAD is at AUC 0.93 and LAD at 0.87. The circumflex and right coronary arteries stay around 0.71 to 0.74. We tried 748 approaches, and that ceiling held. Many papers on this dataset report 97 to 99%. When we run feature selection and oversampling before splitting the data, as some of them do, our own CAD AUC rises from 0.925 to 0.968. That is leakage, so we report the honest number."

**Captions:**
- "Nested CV · 748 procedures · selection inside every fold"
- "CAD 0.934 · LAD 0.866 · LCX 0.708 · RCA 0.737"
- "Leakage inflates 0.925 → 0.968"

## 6:00–6:20 · Everything else (montage, ~20 s)

**Clicks (quick, 2–3 s each):**
1. **Clip** on, drag depth.
2. **X-ray**.
3. Cohort map: click a dot.
4. Most similar patients.
5. **Suggest** counterfactuals.
6. **Report / PDF**.
7. **About & limits**.

**Say:** "There's more: a clipping plane, an X-ray view, a map of all training patients, similar real cases with their angiography, counterfactual suggestions and a printable report."

## 6:20–6:35 · Close

**Say:** "CoronaryLens: one coherent model, honest validation, real anatomy and explanations you can check. Decision support and education only. The code and the full report are linked below."

**Caption:** "CoronaryLens · github link · report PDF"
