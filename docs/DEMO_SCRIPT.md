# Demo video script (7:00; limit 3–10 min)

Record at **1920×1080** (preferred) or 1440×900, Chrome, browser zoom 100%. Every number below is shown in the app at that moment; say them as written. Pace: short sentences, pause about 1 s after each click. Captions: white on dark lower third, 2–4 s each.

Read the **recording checklist** at the end before you start.

**Start state:** app open at http://localhost:8000 (or the live demo URL), **Model performance** tab selected, scrolled so the **Coherence** chart fills the dashboard panel.

---

## 0:00–0:20 · Hook: numbers that cannot contradict each other

**Clicks:** none. Hold on the **Coherence** chart; move the pointer slowly down the bars: *Naive logistic regression ×4* (32.1%) → *Independent, same folds as served* (5.4%) → *Served procedure* (0.0%).

**Say:** "Separate risk models can say a patient is more likely to have a narrowed artery than to have coronary disease at all, which is impossible: that happens for 32% of patients with simple models, 5.4% with a strong ensemble, and 0% with CoronaryLens."

**Caption:** "Impossible contradictions: 32.1% naive · 5.4% strong ensemble · 0% CoronaryLens"

## 0:20–0:40 · What it is, and the disclaimer

**Clicks:**
1. Click the **Patient analysis** tab.
2. Point at the yellow banner at the top.

**Say:** "CoronaryLens turns 54 routine clinical findings into a coronary risk estimate on a real 3D heart, and explains every number. It is decision support and education only, not a diagnosis."

**Captions:**
- "54 clinical, ECG, lab and echo inputs · 303 patients (Z-Alizadeh Sani)"
- "Decision support / education only"

## 0:40–1:40 · Real anatomy

The app opens on the **LAD only** preset.

**Clicks:**
1. Drag on the heart to rotate it slowly.
2. Camera buttons: **AP** → **LAO 30°** → **RAO 30° caudal 20°**.
3. Click **Ribs** to hide the rib cage.
4. Hover the grey vessel at the top (left main).
5. In the left panel click **Full (54)** and scroll to **Pulse rate** (100 bpm). **Heartbeat** stays on.
6. Click **Quick (13)** to go back.

**Say:** "The anatomy comes from BodyParts3D, a public 3D atlas. Each artery is linked to the model by its anatomical ID: the left anterior descending, the circumflex and the right coronary artery. These buttons are standard angiography views. The grey vessel is the left main; the data have no label for it, so it stays grey. The heart beats at the patient's own pulse, here 100 beats per minute."

**Captions:**
- "BodyParts3D anatomy · arteries mapped by FMA ID"
- "Angiographic views: AP · LAO 30° · RAO 30° caudal 20°"
- "Grey = not modelled · heartbeat = patient's pulse"

## 1:40–2:50 · Three-vessel patient: territories, bullseye, explanation

**Clicks:**
1. **Preset patients…** → **Three-vessel disease**. Let the arteries recolour.
2. Point at the CAD gauge (97%) and the three vessel cards (LAD 78%, LCX 57%, RCA 53%).
3. Scroll the dashboard to **AHA 17-segment map**. Hover segment **7** in the bullseye; then hover the front wall of the 3D heart and watch the bullseye segment light up.
4. Click the **LAD** artery on the 3D heart. The camera flies to it and **Why this estimate?** switches to LAD.
5. Point at the top two bars of the waterfall.
6. Scroll to **Measurements**; point at one flagged value and its contribution.

**Say:** "Here is a patient with disease in all three arteries. The heart wall is divided into the 17 standard cardiology segments, coloured by the artery that usually supplies them. The 3D wall and this bullseye share one table, so they always light up together. Click an artery to see why it has this risk. These bars are SHAP values on the exact number shown: red raises the risk, blue lowers it. Every measurement sits next to its own contribution. This describes what the model does, not what causes disease."

**Captions:**
- "17 AHA segments · 3D ↔ bullseye in sync (supply territory, not risk)"
- "Click an artery → its own SHAP explanation"
- "Model behaviour, not causation"

## 2:50–3:50 · What-if, counterfactuals, similar patients

**Clicks:**
1. **Preset patients…** → **Uncertain case** (CAD 78%).
2. In **Why this estimate?** click the **CAD** tab.
3. Scroll to **What-if** and click **Pin current as baseline**.
4. Scroll back up; drag the **Age** slider slowly from 70 down to 45 and back up. Watch the arteries recolour and the deltas.
5. Click **Restore baseline**.
6. In **What-if** click **Suggest**. Point at the first suggestion (triglycerides 198 → 149 mg/dL lowers CAD from 78% to 74%).
7. Scroll to **Most similar training patients · actual angiography**; point at the first row.

**Say:** "Pin a patient, change one finding, and every probability and colour updates live, within a quarter of a second. Suggest searches for the smallest changes to modifiable risk factors that would lower the estimate. That shows how the model responds; it is not treatment advice. Below are the most similar real patients from the training data, with their actual angiography result."

**Captions:**
- "Live update: colours p95 226 ms · explanation p95 360 ms"
- "Counterfactuals = model sensitivity, not treatment advice"
- "Nearest training patients with true angiography"

## 3:50–4:40 · Uncertainty: when not to trust the numbers

**Clicks:**
1. Scroll to the top. Point at the striped vessels on the heart and at a vessel card chip: **Uncertain**, "both outcomes plausible at 90% confidence".
2. **Preset patients…** → **Synthetic out-of-domain**. Point at **Outside training distribution** and a chip reason, "patient unlike the training data".
3. Click **Reset**. Point at the greyed, hatched gauge, the text **Estimate based mostly on cohort averages**, and the grey vessels.

**Say:** "Striped arteries are uncertain: at 90% confidence both answers are still possible, and each chip says why. A patient unlike anyone in the training data gets a warning. If most inputs are missing, the display greys out and says the estimate is based mostly on cohort averages. Confidence fades step by step as more of the important inputs go missing."

**Captions:**
- "Stripes = uncertain (90% conformal set has both outcomes)"
- "Out-of-distribution warning"
- "Missing inputs → colours fade · 'based mostly on cohort averages'"

## 4:40–5:30 · Validation

**Clicks:**
1. Click the **Model performance** tab. Point at the **Headline** table.
2. Scroll to **Model search** (748 procedures).
3. Scroll to **Served procedure vs independent ensemble**.
4. Scroll to **Leakage ablation**.

**Say:** "All results come from nested cross-validation: every modelling choice is made inside the training folds, and we screened 748 approaches that way. Overall CAD reaches an AUC of 0.93. Against a strong ensemble on the same folds, discrimination is equal and the risk estimates are significantly better, using statistics corrected for overlapping folds. When we oversample and select features before splitting, as some published work does, CAD AUC rises from 0.93 to 0.97. That is leakage, so we report 0.93."

**Captions:**
- "Nested CV · CAD 0.934 · LAD 0.866 · LCX 0.708 · RCA 0.737"
- "748 procedures · selection inside every fold"
- "Same AUC · better log-loss (p = 0.0003) and Brier (p = 0.049), corrected tests"
- "Leakage ablation: 0.930 → 0.968"

## 5:30–5:50 · What we learned

**Clicks:** scroll to **Finding · which artery?**

**Say:** "Our main lesson: these inputs say whether there is coronary disease far better than which artery. Among patients with disease, the model separates the left anterior descending modestly, and the other two no better than chance, because the ECG and echo inputs carry no location. So CoronaryLens gives a reliable overall risk, with artery-level estimates as supporting context."

**Caption:** "Among CAD patients · within-CAD AUC: LAD 0.69 [0.59–0.78] · LCX 0.52 [0.42–0.61] · RCA 0.57 [0.47–0.66]"

## 5:50–6:20 · Everything else (montage, about 5 s each)

**Clicks:**
1. **Patient analysis** tab → **Preset patients…** → **Three-vessel disease**.
2. Scroll to **Cohort map**; click a dot to load that training patient.
3. Click **Clip**: the front of the heart is cut away to show the left-ventricle segments. Click **Clip** again.
4. In **Why this estimate?** click **Patient view**.
5. Click **Report / PDF**; show the printable report for 2 s; close it.

**Say:** "There is more: a map of all training patients, a clipping plane, a plain-language patient view, and a printable report with the disclaimer."

**Caption:** "Cohort map · Clipping plane · Patient view · PDF report"

## 6:20–6:40 · Architecture and extensibility

**Clicks:** switch to the browser tab with the GitHub README, scrolled to the **How it works** diagram. Then show `config/features.yaml` (GitHub or editor) for 3 s.

**Say:** "One state store drives the 3D scene and every panel, so they never disagree. Features, targets, anatomy and models are defined in four YAML files: adding a feature or a model needs no code redesign."

**Caption:** "FastAPI + React Three Fiber · one shared state · 4 YAML files define features, targets, anatomy, models"

## 6:40–7:00 · Limitations and close

**Clicks:** back to the app; click **About & limits**; point at **Limitations**.

**Say:** "Limits: 303 patients from one centre, no external validation, and artery-level estimates are supporting context only. CoronaryLens: one coherent model, honest validation, real anatomy, and explanations you can check. Decision support and education only."

**Caption:** "CoronaryLens · github link · report PDF"

---

# Recording checklist (one page)

**Screen and browser**
- Display at **1920×1080** (preferred) or **1440×900**. On a Retina Mac, pick the matching scaled resolution in System Settings → Displays, or size the browser window to 1920×1080.
- Chrome, one window, **zoom 100%** (Cmd+0). Hide the bookmarks bar (Cmd+Shift+B). Full screen (Ctrl+Cmd+F).
- Only two tabs: the app and the GitHub README (for 6:20).

**App state before you press record**
- Run `make serve` (production build on http://localhost:8000) or open the live demo URL. Load the page, click any preset once and wait 10 s so the explanation worker is warm; then reload.
- Default toggles: **Heartbeat** on, **Labels** on, **Ribs** on, **Territories** on (AHA 17-segment), **Clip** off, **X-ray** off.
- Select **Model performance** and scroll so the **Coherence** chart fills the panel (start of 0:00).

**Machine**
- **Do Not Disturb** on (Control Centre → Focus). Quit Slack, Mail, Messages, calendar and anything that shows notifications.
- Charger plugged in; quit heavy apps (Docker Desktop, IDE indexing, other browsers) so the 3D view stays smooth.
- Hide the Dock (Cmd+Option+D) and desktop icons.

**Audio**
- **English** narration. Quiet room, headset or external microphone; make a 20 s test recording and listen back.
- Add English subtitles if the audio is weak (YouTube auto-captions, then correct the numbers by hand: 32%, 5.4%, 0.93, 748, 0.97).

**Recorder**
- QuickTime (File → New Screen Recording) or OBS at 30–60 fps, recording the browser window or full screen.
- Move the pointer slowly; pause 1 s after each click; scroll gently.

**After recording**
- Length 6–7 min (limit 3–10). Check that every number you said was visible on screen.
- Upload to YouTube as **Unlisted** (not Private), English language, title "CoronaryLens · IIT Mandi Hackathon 2026 · Track A", GitHub link in the description.
