# Theory Notes — HIPPO-2 Digital Twin Presentation

Speaker/study notes following the deck's flow. For each concept: **what it is**, **why
you did it**, and **what to say if the examiner pushes (Q&A)**.

---

## 0. The big picture (slides 1–4)

**Virtual sensor / Digital Twin.**
- A *virtual sensor* is a software model that **estimates a physical quantity from other,
  cheaper measurements** instead of measuring it directly. Here: predict NOx, fuel
  consumption, intake pressure, λ from torque, RPM, EGR, exhaust temp, exhaust mass flow.
- A *digital twin* is a living virtual replica of the physical asset, fed by real sensor
  data, used for monitoring, prediction, fault diagnosis.
- **Why it matters on a ship:** emissions/in-cylinder sensors are expensive, intrusive,
  and fail. A trained NN replaces them — cheap, robust, runs in real time on an edge device.

**Q&A:** *"Why a neural network and not physics equations?"* → The thermodynamics (esp.
NOx) are highly non-linear and coupled; a data-driven ANN learns the real engine map
including effects hard to model from first principles, and runs far faster than a CFD model.

---

## 1. Engine physics you must be able to explain

### 1.1 NOx formation — the Zeldovich (thermal NOx) mechanism (slides 7, 12, 21)
- NOx (mostly NO) forms when N₂ and O₂ in the combustion air react at **high temperature**.
- The **extended Zeldovich mechanism** is the chain of reactions; the key fact: NOx
  production is **exponentially sensitive to peak flame temperature** and needs available
  oxygen and residence time.
- So NOx ↑ strongly with load/speed (hotter combustion) → this is *why* SHAP ranks
  **Rotational Speed and Engine Torque** as the top NOx drivers (your run confirms this).

**Q&A:** *"Why was NOx the hardest target?"* → Because the temperature→NOx relationship is
**exponential / strongly non-linear**, a shallow 1-layer ≤20-neuron net underfits it
(R²≈0.59). It needs more capacity (depth) to approximate that curvature.

### 1.2 EGR — Exhaust Gas Recirculation (slides 6, 7)
- Recirculates inert exhaust gas back into the intake. It acts as a **thermal sink /
  diluent**: lowers peak cylinder temperature → **drastically cuts thermal NOx**.
- Hence the **strong negative correlation between EGR command and NOx** — textbook, and
  your heatmap shows it.

### 1.3 Lambda (λ) — air–fuel ratio (slides 7, 23, 24)
- λ = actual air/fuel ÷ stoichiometric air/fuel. λ=1 stoichiometric, λ>1 **lean** (excess
  air, diesels run lean), λ<1 **rich**.
- **NOx vs λ is non-linear and peaks at slightly lean** mixtures: there you get *both* high
  temperature *and* surplus O₂ — the worst case for NOx. Very rich or very lean reduces NOx.
- A wideband O₂ sensor measures λ physically; your model emulates it (MAE ~0.05 = within
  real sensor tolerance).

### 1.4 Intake (manifold) pressure & volumetric efficiency (slides 25, 26)
- Boost/intake pressure governs how much air mass enters the cylinder (volumetric
  efficiency). It tracks **exhaust gas mass flow and load** — which is exactly what SHAP
  ranks top for intake pressure.

### 1.5 Fuel consumption — the (near-)linear target (slides 13, 19)
- Fuel mass flow scales **almost linearly with delivered torque/load**. That's why even the
  *shallow* baseline modeled it near-perfectly (R²≈0.99) — little non-linearity to capture.
- Good contrast point: NOx = non-linear/hard; Fuel = linear/easy. Same method, two regimes.

---

## 2. The ML pipeline — theory behind each step

### 2.1 Data cleaning (slide 5)
- **Physical-constraint filtering:** drop NaNs and physically impossible readings
  (NOx<0, Fuel<0, Intake≤0) — sensor glitches, not real states.
- **3-σ (z-score) outlier removal:** keep only μ−3σ ≤ x ≤ μ+3σ. Assumes roughly Gaussian
  noise; ~99.7% of valid data lies in ±3σ, so points outside are treated as sensor noise.
  - *Q&A risk:* 3σ assumes normality and can clip genuine extreme operating points — defend
    it as a deliberate noise-vs-signal trade-off for stability.
- **MinMax scaling to [0,1]:** puts all features on the same scale so no single large-unit
  feature dominates the gradient; improves numerical stability and convergence of the NN.
  - *Important:* fit the scaler on **train only**, then apply to test — otherwise you leak
    test statistics into training.

### 2.2 The data-splitting strategy — Leave-One-RPM-Out (slides 9, 10, 30)
- **The trap (data leakage):** steady-state engine data has highly correlated adjacent
  samples. A random train/test split puts near-identical points in both → the model
  "memorizes" neighbors and **test score is falsely optimistic**.
- **Your fix:** hold out an *entire operating condition* — **all 1400 RPM data** — for the
  test set; train on the other RPM bands. The model must **interpolate to an unseen regime**.
- **Why this is the real proof:** good performance on a completely withheld RPM band shows
  the network learned the **underlying physics**, not the dataset — i.e. it *generalizes*.
- Also: dropped the **Time** column (`df.drop('Time')`) — you model thermodynamic *state*,
  not temporal history; and `shuffle=True` in the DataLoader breaks sequential bias so each
  batch represents the whole operating envelope.

### 2.3 Neural network basics (slides 4, 11, 16, 17)
- **MLP (multilayer perceptron):** layers of `Linear` (weighted sum) + **ReLU** activation.
  ReLU = max(0,x); the non-linearity is what lets the network approximate curved functions
  (a stack of purely linear layers collapses to one linear map — useless for NOx).
- **Depth vs width:** more neurons/layers = more **capacity** to represent complex mappings.
  Part A constrained you to **1 layer ≤20 neurons** (deliberately weak → underfits NOx).
  Part B allowed **up to 2 layers, 100 neurons** → enough capacity to capture NOx curvature.
- **Adam optimizer:** adaptive-learning-rate gradient descent (momentum + per-parameter
  scaling); robust, fast default for training NNs.
- **MSE loss:** mean squared error — penalizes large errors quadratically; standard for
  regression; differentiable for gradient descent.
- **Dropout:** randomly zeroes a fraction of neurons during training → prevents
  co-adaptation/overfitting, acts as regularization. (Optuna tuned ~0.01–0.14 here.)
- **Epoch:** one full pass over the training data; you trained ~100.

**Underfitting vs overfitting (key narrative):**
- *Underfitting* = model too simple, high error on BOTH train and test (Part A NOx).
- *Overfitting* = model memorizes train, fails on test (train loss ≪ val loss). You avoided
  it — your loss plots show train and val curves tracking together.

### 2.4 Hyperparameter optimization with Optuna (slides 11, 16, 17)
- **Hyperparameters** (set before training, not learned): #layers, #neurons, learning rate,
  dropout. The search space here: 1–2 layers, 10–100 neurons, log-uniform LR.
- **Optuna = Bayesian optimization** via **TPE (Tree-structured Parzen Estimator):** instead
  of trying combinations blindly, it **builds a probabilistic model of which hyperparameters
  gave good results and focuses the search there.**
- **vs Grid/Random search:** grid = exhaustive, exponential cost; random = no learning; TPE
  = **uses past trials to be sample-efficient** → fewer trials for a better config.
- **Define-by-run:** the network is built dynamically each trial, so Optuna can test
  different *architectures* (1 vs 2 layers) on the fly, not just numeric values.
- **Objective:** minimize validation loss (MSE).

**Q&A:** *"How many trials and why?"* → 20 trials/target; TPE is sample-efficient so the
search converges on a strong region quickly; more trials gave diminishing returns.

### 2.5 Feature engineering (slides 14, 15, 18, 22, 26)
You compared **three modes** — this *is* the Part-2 experiment:
- **`none`** — raw physical features only (baseline for Part B).
- **`poly`** — add **polynomial features (degree 2)**: squares and pairwise products of
  inputs. Lets a model capture simple curvature/interactions, but **explodes feature count**
  and can add noise/multicollinearity. *(Your result: it actually hurt NOx — useful finding.)*
- **`gplearn` — Symbolic Regression via Genetic Programming:** evolves **mathematical
  expressions** (combinations like products, ratios, powers of inputs) using an
  evolutionary algorithm — mutation/crossover over generations — keeping expressions that
  correlate best with the target. It can **"rediscover" physical relationships**
  (e.g. Pressure ≈ MassFlow × Temp) automatically.
  - Why it's elegant: it doesn't just fit, it produces **interpretable engineered features**
    that encode non-linear physics the raw inputs alone can't express.

**Feature selection — Mutual Information (slide 18):**
- After generating features you picked the **top features by Mutual Information Regression**,
  not Pearson correlation.
- **Pearson** measures only *linear* association; **Mutual Information** measures *any*
  statistical dependence (information shared, in entropy terms) — so it correctly ranks the
  **non-linear** engineered features that Pearson would miss.

### 2.6 SHAP — model interpretability (the NEW slide)
- **SHAP (SHapley Additive exPlanations):** from cooperative game theory. The **Shapley
  value** fairly distributes a model's prediction among its input features by averaging each
  feature's marginal contribution over all possible feature orderings.
- Output = **mean |SHAP| per feature** = how much each input drives the prediction → a
  ranked, model-agnostic importance.
- **Why it matters here:** it turns the NN from a black box into something **explainable**,
  and your rankings **match engine physics** (NOx←speed/torque, Fuel←torque, λ←torque/EGR,
  intake←mass flow). That's the strongest validation: the model learned *physics*, not noise.
- (Implementation: `KernelExplainer` with a sampled background set — model-agnostic.)

**Q&A:** *"Correlation vs SHAP?"* → Correlation is a property of the *data*; SHAP explains
the *trained model's* actual decisions, including non-linear feature effects.

---

## 3. Evaluation metrics (slides 12, 13, 19, 21, 24, 25, 27)
- **MAE (Mean Absolute Error):** average absolute error, **in the target's physical units**
  (ppm, kg/h, mbar, λ-units). Intuitive: "on average we're off by X." Robust to outliers.
- **MSE (Mean Squared Error):** average squared error; penalizes big misses more; same as
  the training loss; units are squared.
- **RMSE:** √MSE — back in physical units, but outlier-sensitive vs MAE.
- **R² (coefficient of determination):** fraction of variance explained, 0→1 (1 = perfect).
  R²=0.97 means the model explains 97% of the variability in the target.
- **Reporting in physical units** (not normalized) is what makes the result engineering-
  meaningful: "λ MAE 0.05" = within a real wideband O₂ sensor's tolerance.

---

## 4. Your headline story (have this crisp)
1. **Shallow + raw (Part A)** underfits NOx badly (R²≈0.59, MAE≈84.5 ppm) because NOx is
   exponentially non-linear. Fuel, being ~linear, is already near-perfect.
2. **Part B = add capacity (deep, ≤2 layers/100 neurons) + Optuna tuning + feature
   engineering**, validated on a **completely held-out 1400 RPM band** (no leakage).
3. NOx error collapses dramatically; SHAP confirms the model relies on the **physically
   correct drivers**. → A validated, lightweight virtual sensor ready for an edge digital twin.

> ⚠️ Keep the exact "what caused the NOx improvement" line until the gplearn run finishes —
> this session's data suggests **network depth** (not just gplearn features) does much of the
> heavy lifting. Tell the true story once we have the final gplearn number.

---

## 5. Likely Q&A traps — quick answers
- *"Did you test deep network without gplearn?"* → Yes (the `none`/`poly`/`gplearn`
  comparison) — that's exactly the Part-2 ablation. *(Numbers pending final run.)*
- *"Why 1400 RPM for the test set?"* → A mid-range band fully removed so the model must
  interpolate an unseen operating condition — the strict generalization test.
- *"Isn't 3σ clipping risky?"* → It can remove genuine extremes; deliberate noise-reduction
  trade-off; physical-bound filtering handles the hard-invalid cases separately.
- *"Why MinMax not standardization?"* → Bounds features to [0,1] for stable gradients; fine
  for non-Gaussian sensor ranges; fit on train only to avoid leakage.
- *"Overfitting?"* → Loss curves: train and validation track together; dropout + held-out
  RPM band guard against it.
- *"Why MLP not LSTM?"* → You model steady-state thermodynamic *state*, not time dynamics;
  LSTM is listed as future work for transient behavior.
