# PPTX BRIEF — Handover for building/fixing the HIPPO-2 presentation (nm20275.pptx)

**For: a fresh Claude session (e.g. claude.ai in the browser) that will edit the slides.**
**From: the Linux training session that ran the full experiment.**

The student (Dagklis Spyros, nm20275) presents on **12/6/2026**. The deck `nm20275.pptx`
currently has **31 slides** and must become **20** (hard cap — see spec). Everything you need
is in this file. You do NOT have the machine that ran the experiment, so all numbers, rankings,
and the plot list are reproduced here.

## How to use this (browser Claude)
1. Ask the student to **upload `nm20275.pptx`** (you can read/edit pptx).
2. Optionally have them upload the figures: `plots_part_b/05_SHAP_bar_*`, `06_SHAP_beeswarm_*`,
   `07_Ablation_NOx.png`, `07_Ablation_Fuel_Consumption.png`, `03_Loss_*`, `04_Scatter_*`.
3. Use the sections below to (A) add 3 slides, (B) fix numbers/language, (C) cut 31→20.
4. Companion files also in the folder (ask student to paste if needed): `THEORY_NOTES_GR.md`
   (full Greek theory), `SPEAKER_NOTES.md` (per-slide scripts), `THEORY_NOTES.md` (EN + Q&A).

---

## 0. The assignment spec (2026, both parts) — what is REQUIRED
Two briefs: Part 1 (simple nets) + Part 2 (improvement). The student chose **NOx and Fuel
Consumption** as the two parameters (required throughout). Intake Pressure and λ were also
modeled but are **EXTRA** (not required) — good candidates to cut for the 20-slide cap.

**Part 1 (required):** 1) data graphs, 2) filter/transform, 3) train/val/test split, 4) **2
simple NNs (NOx, Fuel), 1 hidden layer ≤20 neurons**, 5) test eval + train/val loss plots.

**Part 2 (required), for NOx & Fuel:** 1) feature engineering + justification, 2) deeper nets
+ justification, 3) Optuna/WandB, 4) test eval + plots + **compare vs Part-1 simple**, 5) **for
each change (features/neurons/layers/…): show effect + a SUMMARY TABLE (each change → total test
loss)**, 6) **SHAP for best models + discuss physical consistency + whether engineered features
contribute**.

**Hard rules:** Presentation = **20 slides, both parts combined** (Part 2 Note 7). Plots must
have **axis labels, curve labels, grids, no screenshots** (Note 3). Cite literature where used
(Optuna [Akiba 2019], WandB [Biewald 2020], SHAP [Lundberg & Lee 2017]).

---

## 1. AUTHORITATIVE FINAL NUMBERS (source of truth — replace all slide numbers with these)
From `final_results.csv`. test_MAE is in physical units. **Use these, ignore older slide values.**

| Target | Mode | n_layers / units | test MAE | R² | RMSE |
|---|---|---|---|---|---|
| **NOx** | none | 2 (97→65) | **13.56 ppm** ✅best | 0.968 | 18.03 |
| NOx | poly | 2 (94→96) | 17.91 | 0.945 | 23.64 |
| NOx | gplearn | 2 (79→100) | 18.38 | 0.943 | 24.05 |
| **Fuel** | none | 1 (89) | 0.345 | 0.9992 | 0.475 |
| Fuel | poly | 1 (40) | **0.285** ✅best | 0.9995 | 0.378 |
| Fuel | gplearn | 1 (100) | 0.292 | 0.9995 | 0.391 |
| *(extra)* Intake | none | 1 (65) | 1.21 mbar | 0.9991 | 1.55 |
| *(extra)* lambda | none | 2 (36→70) | 0.039 | 0.9978 | 0.067 |

**Part-1 SHALLOW baseline (≤20 neurons, raw features), from the old slides — keep, but use ONE
consistent value:** NOx baseline **MAE 84.49 ppm, R² 0.59** (NOT 85.49/85.5). Fuel baseline
R² 0.99, MAE ≈ 0.40 kg/h.
> ⚠️ Two more rows (Intake-gplearn, lambda-gplearn) finish after this brief is written — they are
> EXTRA params, not needed for required slides. If present in `final_results.csv`, use them; else
> ignore.

---

## 2. THE CORRECTED HEADLINE STORY (tell this honestly — it is stronger than the old deck)
The old deck claimed gplearn features caused the NOx improvement. The full 3-mode sweep shows:
- **NOx:** the plain **deep network with RAW features (none) wins (13.56)**; poly (17.91) and
  gplearn (18.38) are WORSE. → **Network DEPTH/capacity fixed NOx underfitting, not the
  engineered features.** (The ablation confirms: wider→better; extra depth→diminishing.)
- **Fuel:** feature engineering **helps** — poly (0.285) and gplearn (0.292) beat none (0.345).
- **Unified message:** "Capacity solved the non-linear target (NOx); feature engineering helped
  the near-linear one (Fuel). We followed the data, not an assumption." This directly satisfies
  Part-2 step 5 (effect of each change) and reads as mature science.

Baseline → best: **NOx 84.49 → 13.56 ppm** (~6× better); **Fuel ≈0.40 → 0.285 kg/h**.

---

## 3. THREE SLIDES TO ADD (two are REQUIRED, one the student explicitly wants)

### 3a. SHAP slide — REQUIRED (Part 2, step 6). Insert after the Optuna/architecture slides.
**Title:** *Model Interpretability with SHAP — the model learned the physics*
**Use figures:** `05_SHAP_bar_NOx_NONE_PartB.png`, `05_SHAP_bar_Fuel_Consumption_*`.
**Feature importance (mean |SHAP|), from the run:**
- **NOx:** Rot.Speed (0.166) > Engine_Torque (0.089) > Exh.Gas_Temp (0.026) > Torque_Ref (0.015)
  > EGR (0.010) > Exh.Mass_Flow (0.004)
- **Fuel:** Engine_Torque (0.239, dominant) > Exh.Gas_Temp (0.036) > Exh.Mass_Flow (0.022) > …
- *(extra) λ:* Engine_Torque (0.371) > EGR (0.134) > Exh.Gas_Temp (0.119) …
- *(extra) Intake:* Exh.Mass_Flow (0.065) > Engine_Torque (0.044) > Exh.Gas_Temp (0.037) …
**IMPORTANT — use the BEST model's SHAP per target:** NOx best = `none` → use
`05_SHAP_bar_NOx_NONE_PartB.png`; Fuel best = `poly` → use
`05_SHAP_bar_Fuel_Consumption_POLY_PartB.png` (NOT the none one).
**Bullets:** SHAP = Shapley values (game theory) → fair per-feature attribution of each
prediction. Rankings **match engine physics** (NOx ← speed/load → Zeldovich thermal NOx; Fuel ←
torque). **Required discussion of engineered features (REAL DATA from the run):**
- **Fuel (poly, the best Fuel model):** the top SHAP features are polynomial interactions —
  Engine_Torque×Exh_Gas_Temp (0.210), Torque_Ref×Exh_Gas_Temp (0.164), Torque_Ref×Rot_Speed
  (0.158) → **engineered features DO contribute meaningfully** (poly beats none: 0.285 vs 0.345).
- **NOx (gplearn):** top SHAP features are RAW physical — Engine_Torque (0.229), EGR (0.183),
  Exh_Gas_Temp (0.088), Torque_Ref (0.080); the engineered **GP_3 (0.078), GP_4 (0.070), GP_1
  (0.056) rank LOWER** → **engineered features do NOT contribute meaningfully for NOx** (consistent
  with gplearn 18.38 being worse than none 13.56). State this — it's exactly the brief's question.

### 3b. Ablation / summary-table slide — REQUIRED (Part 2, step 5). 
**Title:** *Architecture Ablation — each change vs test loss*
**Use figure:** `07_Ablation_NOx.png` (and Fuel). **Table (NOx, none; normalized test MSE):**
| Change | Config | test MSE |
|---|---|---|
| width | 1L×5 / 10 / 20 / 50 / 100 | 0.00130 / 0.00065 / 0.00074 / 0.00056 / **0.00052** |
| depth | 1 / 2 / 3 layers (×50) | **0.00060** / 0.00070 / 0.00076 |
**Comment (the brief literally gives this expectation):** more neurons → lower error; more layers
→ initially fine but **diminishing/worse beyond ~1–2 layers** (mild overfitting). Plus the
feature-mode comparison (none/poly/gplearn) is itself a "change" → include the NOx & Fuel MAE
rows from §1 as part of the summary.

### 3c. "How the model is optimized" slide — STUDENT EXPLICITLY WANTS THIS.
**Title:** *Two-Level Optimization: Optuna searches, Adam learns*
**Bullets:**
- **Adam (inner loop)** optimizes the network **weights** via gradient descent with momentum +
  per-parameter adaptive step. Acts at `optimizer.step()` after `loss.backward()`.
- **Optuna (outer loop)** optimizes the **hyperparameters** — learning rate, #layers (1–2),
  #neurons (10–100), dropout — keeping the lowest validation loss (Bayesian TPE, minimize).
- **The link:** Optuna *chooses* the learning rate → Adam *uses* it:
  `optim.Adam(model.parameters(), lr=trial.suggest_float('lr',1e-4,1e-3,log=True))`.
- `study.optimize(objective, n_trials=20)` calls `objective()` 20× (each: build MLP, train
  briefly with Adam, return val_loss); then `best=study.best_params` rebuilds + fully trains the
  winner, evaluated on the held-out **1400 RPM** test set.
**Flow diagram (recreate as boxes/arrows):**
```
FOR each target:
   study.optimize(objective, n_trials=20):          ┐ OUTER (Optuna: hyperparameters)
       ×20 trials → objective():                    │
           Optuna → lr, #layers, #neurons, dropout  │
           build MLP (Linear→ReLU→Dropout…)         │
           train with Adam ───────────┐             │
              per batch: zero_grad →   │ INNER       │
              backward → step          │ (weights,   │
              (Adam updates weights)   ┘  step=lr)   │
           return val_loss                           ┘
   best = study.best_params  →  rebuild + full train  →  eval on TEST (1400 RPM)
```
**One-liner:** *Adam asks "how do I change the weights?"; Optuna asks "which lr/architecture do I
give Adam?" — Optuna chooses, Adam applies.*

---

## 4. NUMBER / CONSISTENCY FIXES
- NOx shallow baseline: **84.49 ppm** everywhere (slides showed 84.49 / 85.49 / ~85.5 — unify).
- Fuel: drop the 0.42-vs-0.32 conflict; use the §1 values (none 0.345 / poly 0.285 / gplearn 0.292).
- Update the **comparison table (current S27)** with §1 numbers; keep it consistent with the
  per-target result slides.

## 5. LANGUAGE / FORMAT FIXES
- **S20** (Greek) and **S25** (Greek) → translate to English to match the deck.
- **S25**: broken LaTeX `$0.9989$` → render as `R² = 0.9989`.
- **S21**: starts with stray `:` and no title → fix title + remove colon.
- Verify all figures have axis labels, curve labels, grids, are not screenshots (loss plots were
  fixed to show BOTH train+val with labels/grid; check scatter/heatmap/ablation/SHAP).

## 6. CUT 31 → 20 SLIDES — FINAL PLAN (student confirmed 20, per official 2026 Part-2 rule)
Target deck = exactly 20 slides. Map each NEW slide to its position; merge/cut the rest.
λ and Intake are EXTRA params (assignment only requires NOx & Fuel) → drop their dedicated
slides but KEEP their result rows in the comparison table (slide 19), so the work still shows.

| New # | Slide | Built from | Action |
|---|---|---|---|
| 1 | Title | S1 | keep |
| 2 | Agenda | S2 | keep |
| 3 | HIPPO-2 facility & objectives | S3 | keep |
| 4 | Targets & input features | S4 | keep |
| 5 | Preprocessing & filtering | S5 | keep |
| 6 | Data analysis: heatmap + thermo NOx | S6 + S7 | merge |
| 7 | EDA: engine operating map | S8 | keep |
| 8 | Data split: Leave-One-RPM-Out | S9 + S10 | merge (key slide) |
| 9 | Part A baseline architecture | S11 | keep |
| 10 | Part A NOx result (84.49) | S12 | keep |
| 11 | Part A Fuel result | S13 | keep |
| 12 | Part B strategy + symbolic regression | S14 + S15 | merge |
| 13 | Feature selection: Mutual Information | S18 (drop the code dump) | keep |
| 14 | **Two-Level Optimization: Optuna + Adam** | NEW (§3c); replaces S16, absorbs S17 | ADD |
| 15 | **Ablation / summary table** | NEW (§3b) | ADD — REQUIRED (step 5) |
| 16 | Part B NOx result | S21 + S22 | merge |
| 17 | Part B Fuel result | S19 + S20 | merge |
| 18 | **SHAP** | NEW (§3a) | ADD — REQUIRED (step 6) |
| 19 | Comparison table Part1 vs Part2 (incl. λ, Intake rows) | S27 | keep |
| 20 | Conclusions + future work | S28 + S29 + S30 + S31 | merge |

**Eliminated as dedicated slides:** S7→6, S10→8, S15→12, S16→14, S17→14, S20→17, S22→16,
**S23+S24 (λ) and S25+S26 (Intake) CUT** (kept only as comparison-table rows), S29+S30+S31→20.
Net: 31 − 14 removed/merged + 3 added = **20.** ✓

## 7. Quick coverage status (what's already DONE vs deck)
- Part 1: steps 1–5 all present (S5–S13). ✅
- Part 2: feature-eng ✅(S14,15,18), deeper+justify ✅(S16,17 + ablation), Optuna/WandB ✅,
  compare-vs-simple ✅(S27). **Missing as SLIDES: SHAP (→3a) and the summary/ablation table
  (→3b)** — data/plots exist, just not on a slide. Add them.

---
*Generated from the live run. final_results.csv / ablation_results.csv / plots_part_b are the
ground truth; if a number here disagrees with the latest CSV, trust the CSV.*
