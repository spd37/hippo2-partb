# Speaker Notes — HIPPO-2 Digital Twin (per slide)

A 2–4 sentence spoken script per slide, keyed to the **current 31-slide deck**. Paste each
into the slide's Notes pane. ⚠️ = a number/wording to fix (see THEORY_NOTES / PPTX_BRIEF).
Numbers marked *(final run)* should be taken from `final_results.csv` once gplearn finishes.

---

**S1 — Title.**
"Good morning. I'm Spyridon Dagklis. This is my project on data-driven virtual sensors and a
digital twin for the HIPPO-2 hybrid diesel-electric engine, supervised by Dr. Papalambrou."

**S2 — Agenda.**
"I'll cover the facility and the problem, how I cleaned and split the data, the Part-1 baseline
networks, the Part-2 deep models with feature engineering, the comparison in physical units,
and the engineering conclusions. About 30 minutes."

**S3 — Facility & objectives.**
"HIPPO-2 is a hybrid rig: a 261 kW diesel and a 90 kW electric motor on a common shaft.
Measuring emissions and in-cylinder parameters directly is expensive, intrusive and
failure-prone. My goal is software virtual sensors — and ultimately a digital twin — that
estimate those quantities from cheap, routine measurements using neural networks."

**S4 — Targets & inputs.**
"Inputs are six operational, control and thermodynamic signals — torque, speed, EGR, exhaust
temperature and mass flow. Part 1 predicts four targets with a strictly shallow network; Part
2 uses deeper networks plus feature engineering. My two focus targets are NOx and fuel
consumption — one strongly non-linear, one nearly linear."

**S5 — Preprocessing.**
"From 451,228 raw samples I first removed NaNs and physically impossible readings — negative
NOx or fuel, non-positive intake pressure. Then a 3-sigma z-score filter removes sensor noise,
keeping data within mean ±3 standard deviations. Finally MinMax scaling to [0,1] for stable
training. That leaves about 428,000 high-fidelity samples."
*(Mention: scaler fit on train only, to avoid leakage.)*

**S6 — Correlation heatmap.**
"The Pearson matrix maps linear relationships. Torque reference correlates above 0.9 with fuel
consumption, and EGR correlates strongly *negatively* with NOx — exactly what engine
thermodynamics predicts. I used this to check for redundant features and guide selection."

**S7 — NOx thermodynamics.**
"Three physical effects govern NOx: temperature, where high flame temps trigger exponential
thermal NOx via the Zeldovich mechanism; EGR, which lowers peak temperature and suppresses
NOx; and the air-fuel ratio, where NOx peaks at slightly lean mixtures. This non-linearity is
why NOx is the hardest target."

**S8 — Exploratory data / engine map.**
"This is how the data was generated: at each fixed speed step the load is swept across its
range, producing a steady-state engine map. The lower plot shows actual shaft torque, from
braking-negative through to full load — the operating envelope the model must learn."

**S9 — Data splitting strategy.**
"A standard random split would leak: steady-state samples are highly correlated, so neighbors
end up in both train and test and the model just memorizes them. Instead I used
Leave-One-Operating-Condition-Out — I removed *all* 1400 RPM data and reserved it purely for
testing. The model must interpolate to a speed band it never saw."

**S10 — Integrity & training strategy.**
"I dropped the Time column because the virtual sensor maps thermodynamic state, not temporal
history. I shuffle the training data so each batch represents the whole operating envelope,
preventing sequential bias. And the 1400 RPM hold-out stays completely isolated — a watertight
split."

**S11 — Part-A baseline architecture.**
"The Part-1 constraint is deliberately restrictive: one hidden layer, at most 20 neurons. I
used Optuna to search neurons and learning rate, with Adam, ReLU and MSE loss over 100 epochs.
This tests how far a minimal network can go."

**S12 — Part-A NOx result. ⚠️**
"For NOx the shallow model converges smoothly but underfits — R² about 0.59, MAE around
**84.49 ppm**, with heavy scatter. It simply lacks the capacity to capture the exponential
thermal dynamics. This motivates the deeper Part-2 approach."
*(Use 84.49 consistently — fix the 85.49/85.5 elsewhere.)*

**S13 — Part-A fuel result.**
"Fuel consumption is the opposite story: even the shallow network nails it — R² 0.99, MAE
about 0.4 kg/h, points tight on the y=x line. That's because fuel flow is nearly linear in
engine load, so little capacity is needed."

**S14 — Part-B strategy.**
"To overcome NOx underfitting I combined three moves: feature engineering via symbolic
regression, more model capacity — up to two layers and 100 neurons — and systematic
hyperparameter tuning with Optuna."

**S15 — Symbolic regression.**
"Genetic programming, via gplearn, evolves mathematical expressions that combine the raw
sensors — products, ratios, powers — keeping those most predictive of the target. It can
effectively rediscover non-linear physical relationships the raw inputs can't express alone."

**S16 — How Optuna works.**
"Optuna uses Bayesian optimization with the TPE algorithm: rather than brute-force grid or
random search, it learns from past trials to focus on promising regions of the hyperparameter
space. Its define-by-run design even lets it test one- versus two-layer architectures on the
fly."

**S17 — Optimization framework / winning architectures. ⚠️**
"I tracked every trial live in Weights & Biases. The search space was one-to-two layers,
10–100 neurons, and a log-uniform learning rate. Fuel's best architecture stayed shallow,
while NOx needed two layers — confirming it genuinely requires deep learning."
*(Architectures change per run — read final values from final_results.csv before quoting.)*

**S18 — Feature selection (Mutual Information).**
"For final feature selection I went beyond Pearson correlation and used Mutual Information
Regression. Pearson only captures linear association, whereas mutual information measures any
statistical dependence — so it correctly ranks the non-linear engineered features."

**S19 — Part-B fuel result. ⚠️**
"With the deep setup fuel stays excellent — R² about 0.99, MAE around **0.35 kg/h** *(final
run)* — validation tracking training, zero overfitting. Confirms fuel scales directly with
mechanical load."
*(Replace the 0.42/0.32 inconsistency with the final_results.csv value.)*

**S20 — Fuel training (currently Greek). ⚠️**
"The loss curves show near-zero validation loss and very fast convergence with no overfitting."
*(Translate this slide to English to match the deck — or merge into S19.)*

**S21 — Part-B NOx result. ⚠️ KEY SLIDE.**
"This is the headline. The baseline underfit at MAE ~84.5 ppm; the deep, tuned model
collapses the error dramatically and the predictions align with the y=x diagonal. *(State the
final NOx MAE from final_results.csv, and the honest cause — depth vs. gplearn — once the run
finishes.)*"
*(Fix the stray ':' and missing title; reconcile 85.49 → 84.49.)*

**S22 — NOx feature engineering. ⚠️**
"Fast, stable convergence with no overfitting. The engineered features capture the bulk of the
correlation with NOx, outperforming raw physical variables."
*(Re-check the 'GP features dominate' claim against this run — depth may explain most of it.)*

**S23 — λ symbolic features.**
"For lambda the engineered features reach a correlation around -0.97 — the genetic algorithm
essentially rediscovered the thermodynamic relationships linking mass flow, temperature and
torque to air-fuel ratio. By withholding fuel as an input I forced the model to learn real
physics rather than take a shortcut."

**S24 — λ result.**
"Lambda comes out excellent — MAE about 0.05, R² 0.997. In practical terms, if true lambda is
1.50 the model predicts within 1.45–1.55, which is inside the tolerance of a physical wideband
O₂ sensor."

**S25 — Intake pressure. ⚠️**
"Intake pressure is predicted to about 1.35 mbar MAE, R² 0.999, with near-perfect alignment to
the ideal line — accurate enough to act as a reliable virtual sensor in the digital twin."
*(Fix: render 'R² = 0.9989' — remove the literal $…$; translate the Greek words.)*

**S26 — Intake feature engineering.**
"Symbolic regression again provided the most predictive features, and Optuna chose a two-layer
network — deeper capacity is needed to map the complex air-manifold dynamics. Training
converged smoothly with no overfitting."

**S27 — Comparison table. ⚠️**
"Summarizing in physical units: NOx improves dramatically from the shallow baseline; fuel
improves only marginally because it was already near-linear; and intake pressure and lambda,
not modeled in Part 1, reach lab-grade accuracy. The big non-linear leap is NOx."
*(Replace all cells with final_results.csv numbers; keep them consistent with S12/19/21/24/25.)*

**S28 — Conclusions.**
"In short: the networks successfully mapped the engine's thermodynamic states, with errors
inside real-sensor tolerances. Symbolic regression exposed the underlying physics and Optuna
balanced capacity against generalization. The models are lightweight and deployment-ready."

**S29 — What is a digital twin.**
"A digital twin has three pillars: the physical asset, the virtual model, and the live data
flow linking them. For marine engineering it enables virtual sensing, predictive maintenance
and fault diagnosis without intrusive hardware."
*(Candidate to cut/condense for the 20-slide cap.)*

**S30 — Generalization.**
"Because I removed the entire 1400 RPM band from training, accurate test predictions prove the
model approximated the underlying physics rather than memorizing the lab data. That strict
separation is the ultimate validation of the virtual sensors."

**S31 — Future work.**
"Next steps: export the networks to ONNX for edge deployment on shipboard devices; explore
LSTM networks to capture transient, time-dependent behavior beyond steady state; and a
continuous-learning pipeline that retrains on live vessel data."

---

### Add: SHAP slide (insert after S17, before the per-target results)
"To make the models explainable I applied SHAP, which uses Shapley values from game theory to
fairly attribute each prediction to the input features. Crucially, the rankings match engine
physics: NOx is driven by speed and torque, fuel by torque, lambda by torque and EGR, and
intake pressure by exhaust mass flow. This confirms the networks learned real physical
relationships, not spurious correlations — turning the black box into a trustworthy virtual
sensor."
