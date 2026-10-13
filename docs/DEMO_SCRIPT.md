# Demo video script (target 6:30, limit 3–10 min)

**Setup before recording**
- Run `make dev`, open http://localhost:5173 in Chrome at 1600×1000 (or full-screen 1440p), dark OS theme.
- Close other tabs, and refresh once so the explanation worker is warm.
- Default state: Heartbeat on, Labels on, Ribs on, Territories on (AHA 17-segment).
- Screen recorder at 30–60 fps, microphone on. Keep the pointer slow; pause about 1 s after each click.
- Every number below is shown in the app. Say them as written.

**Caption style:** short lower-third text, white on dark, 2–4 s each.

---

## 0:00–0:40 · Hook: a heart you can question

**Clicks:**
1. **Preset patients…** → **Three-vessel disease**. Let the vessels recolour.
2. Drag to rotate slowly.
3. In the left panel, click **Typical chest pain** to switch it to *No*. Watch the arteries recolour.
4. Click it back to *Yes*.

**Say:** "This is CoronaryLens. Enter a patient's clinical findings and it shows how likely each coronary artery is to be narrowed, on a real 3D heart, and explains why. Change one finding, here typical chest pain, and every artery and number updates in about a fifth of a second."

**Captions:**
- "Real 3D anatomy · live risk per artery"
- "Update in 175 ms (p95)"

## 0:40–0:55 · Disclaimer and data

**Clicks:** point at the yellow banner at the top.

**Say:** "This is decision support and education only, not a diagnosis. It uses 303 patients from the public Z-Alizadeh Sani dataset. The answers we predict, overall CAD and the three arteries, are never used as inputs."

**Caption:** "Decision support / education only · 303 patients · labels never used as inputs"

## 0:55–1:45 · Real anatomy

**Clicks:**
1. Camera presets **AP** → **LAO 30°** → **RAO 30° caudal 20°**.
2. Toggle **Ribs** off.
3. Hover the grey vessel at the top.

**Say:** "The anatomy is BodyParts3D, a medical 3D atlas. Each artery is mapped to the model by its anatomical ID. These buttons are standard angiography views. The grey vessel is the left main, which the data does not label, so we leave it uncoloured. The heart beats at the patient's own pulse rate."

**Captions:**
- "BodyParts3D anatomy · LAD / LCX / RCA by FMA ID"
- "Angiographic views · grey left main = not modelled"

## 1:45–2:35 · Territories and bullseye

**Clicks:**
1. Point at the CAD gauge, the LAD / LCX / RCA cards, and the "Multi-vessel" and "Three-vessel" tiles.
2. Scroll to the **AHA 17-segment map**.
3. Hover segment **7**, then hover a segment on the 3D heart; show that it lights up in the bullseye.
4. Hover an **Uncertain** chip so its tooltip shows.

**Say:** "The heart wall is split into the standard 17 cardiology segments, coloured by the artery that usually supplies them. The 3D wall and this bullseye use one shared table, so hovering one highlights the other. Each artery also gets a status. 'Uncertain' means the data cannot rule out either answer for that artery at 90% confidence. It is calibrated per artery, which is why 78% for LAD can be uncertain while 57% for LCX is not."

**Captions:**
- "17 AHA segments · 3D ↔ bullseye in sync"
- "Expected supply territory (illustrative)"
- "Stripes = uncertain (90% conformal)"

## 2:35–3:30 · Why: per-vessel explanation

**Clicks:**
1. Click the **LAD** vessel on the 3D heart. The camera flies to it and the explanation switches to LAD.
2. Point at the waterfall, from the cohort average to this patient.
3. Toggle **By group**, then toggle it back.
4. Scroll to **Measurements**. Point at two flagged values and their contribution numbers.
5. Click **Patient view**.

**Say:** "Click any artery to see why. These are SHAP values computed on exactly the number shown, from the cohort average up to this patient. Red raises the risk and blue lowers it. Every measurement sits next to its own contribution, and there is a plain-language version for patients. These describe what the model does, not what causes disease."

**Captions:**
- "Click a vessel → its own explanation"
- "Model behaviour, not causation"

## 3:30–4:15 · What-if and an honest limit

**Clicks:**
1. In **What-if**, click **Pin current as baseline**.
2. Drag the **Age** slider slowly from 75 down to 45. Watch the colours and deltas.
3. Click **Restore baseline**.
4. **Preset patients…** → **RCA only**.

**Say:** "Pin the patient and change one thing to see every probability move. Now an honest limit. This patient has disease only in the right coronary artery, but the model rates the LAD higher. In this data, clinical findings tell us *whether* there is coronary disease far better than *which* artery. Among patients with disease, the model separates the circumflex and right coronary arteries barely better than chance. The app says so next to every vessel explanation, so nobody over-reads the colours."

**Captions:**
- "Deltas vs pinned baseline"
- "Which artery? Within-CAD AUC: LAD 0.69 · LCX 0.52 · RCA 0.57"

*(Do not sweep ejection fraction across 55% here. The model has a small step at the edge of the reference range, covered in the Q&A.)*

## 4:15–4:40 · Out-of-distribution warning

**Clicks:**
1. **Preset patients…** → **Synthetic out-of-domain**.
2. Point at the yellow **"Outside training distribution"** badge.

**Say:** "For patients unlike anyone in the training data, the app warns that its estimates are unreliable."

**Caption:** "Out-of-distribution warning"

## 4:40–5:40 · One coherent model, honestly validated

**Clicks:**
1. Open **Model performance**.
2. Show the **Headline** table.
3. Scroll to **Coherence**.
4. Scroll to **Served procedure vs independent ensemble**.
5. Scroll to **Model search** and **Leakage ablation**.

**Say:** "Every number comes from one model of all eight possible artery patterns, so overall risk, single-artery risk and multi-vessel risk always fit together. Separate models put the overall CAD risk below a single artery's risk for 5 to 32% of patients. Against a strong ensemble on the same folds, our AUC is the same and our risk estimates are significantly better calibrated. All results come from nested cross-validation: every choice is made inside each training fold, and we screened 748 approaches. CAD is at AUC 0.93 and LAD at 0.87, while the circumflex and right coronary arteries stay around 0.71 to 0.74. When we oversample and select features before splitting, as some published work does, CAD AUC rises from 0.93 to 0.97. That is leakage, so we report the honest number."

**Captions:**
- "Contradictions: 32.1% naive · 5.4% strong ensemble · 0% ours"
- "Same AUC · better log-loss (p = 0.0003) and Brier (p = 0.049)"
- "Nested CV · 748 procedures · CAD 0.934 · LAD 0.866 · LCX 0.708 · RCA 0.737"
- "Leakage: 0.930 → 0.968"

## 5:40–6:00 · Everything else (montage, ~20 s)

**Clicks (quick, 2–3 s each):**
1. **Clip** on, drag depth.
2. **X-ray**.
3. Cohort map: click a dot.
4. Most similar patients.
5. **Suggest** counterfactuals.
6. **Report / PDF**.
7. **About & limits**.

**Say:** "There's more: a clipping plane, an X-ray view, a map of all training patients, similar real cases with their angiography, counterfactual suggestions and a printable report."

## 6:00–6:15 · Close

**Say:** "CoronaryLens: one coherent model, honest validation, real anatomy and explanations you can check, including where the model reaches its limits. Decision support and education only."

**Caption:** "CoronaryLens · github link · report PDF"
