# Claude Handoff — HIPPO-2 Digital Twin (Special Ship Automation Systems 2026)

**Purpose of this file:** context for any future Claude Code session (e.g. continuing on the
Linux home machine) to pick up exactly where we left off. Read this first.

## Project
- Course HW: "System Modeling Using Experimental Measurement Data", Parts 1 & 2.
- Goal: neural-network **virtual sensors / digital twin** for the HIPPO-2 hybrid diesel-electric
  rig. Predict engine parameters from the other measured signals.
- Student: Dagklis Spyros (nm20275, spd37@hotmail.com). Exam date **12/6/2026**.
- Single script: `nm20275.py`. Data: `Data2026.csv` (30MB, tab-separated, units row is line 2,
  skipped via `skiprows=[1]`). Filename is **case-sensitive on Linux** — keep it `Data2026.csv`.
- Targets in code loop: NOx, Fuel_Consumption, Intake_Pressure, lambda. The two *chosen* Part-1
  parameters (per the presentation) are **NOx and Fuel_Consumption**.
- Pipeline: clean (physical bounds + 3-sigma) → feature-eng switch → Optuna deep MLP → WandB →
  test eval on a held-out **1400 RPM** band (Leave-One-RPM-Out) → plots.
- `FEATURE_ENG_MODE` (line ~89) switch: `'none'` / `'poly'` / `'gplearn'`. Run all three to build
  the feature-engineering comparison. `'gplearn'` gives the strong NOx result (~19 ppm MAE).
- WandB entity hardcoded `dagklis-`, project `Advanced_Control_Systems`. Use `WANDB_MODE=offline`
  to skip the dashboard.

## What was already done this session (committed)
Closed the genuinely-missing Part-2 requirements after comparing code + both PDFs + the 31-slide
`nm20275.pptx` (pptx/pdf are NOT in the repo — they live on the laptop):
1. **SHAP (Part 2, Step 6)** — `run_shap_analysis()`. KernelExplainer + numpy predict wrapper,
   `shap.sample(X_train,100)` background, capped test subset. Saves `05_SHAP_bar_*` and
   `06_SHAP_beeswarm_*`, logs to the same WandB run, prints feature ranking. Smoke-tested OK.
2. **Ablation table (Part 2, Step 5)** — `run_architecture_ablation()` (width sweep 5/10/20/50/100
   @1 layer, depth sweep 1/2/3 layers) → `07_Ablation_*` plot + `ablation_results.csv`. Plus
   `log_final_metrics_csv()` → `final_results.csv` (accumulates one row per run/mode). Gated by
   `RUN_ABLATION = True` (line ~91).
3. **Plot/filename fixes** — loss plot now draws BOTH train+val with axis labels/grid;
   heatmap filename carries `_{FEATURE_ENG_MODE}` so modes stop overwriting each other.
4. **Data filename** — changed `data.csv` → `Data2026.csv`.
5. `prepare_loaders_part_b` now also returns `X_train, X_test, selected` (needed by SHAP).

## What is still TODO
- **Presentation (not code):** add one SHAP slide from the new outputs; trim 31→**20** slides
  (Part 2 note 7 cap, both parts combined); fix mixed Greek text on slides 20 & 25; reconcile
  inconsistent numbers (baseline NOx MAE 84.49/85.49/85.5; Fuel Part2 MAE 0.42 vs 0.32).
- **Full comparison table:** run the script in all three feature modes (`none`,`poly`,`gplearn`)
  so `final_results.csv` has all rows.
- **Optional (offered, not yet done):** auto-commit/push of `plots_part_b/` + result CSVs at the
  end of the script so artifacts return to GitHub without remote-desktop file transfer.

## How to run (Linux home machine, via remote desktop)
```bash
cd hippo2-partb
python3 -m venv venv && source venv/bin/activate
pip install torch optuna optuna-integration wandb gplearn shap seaborn pandas scikit-learn matplotlib
wandb login                       # or: export WANDB_MODE=offline
# edit FEATURE_ENG_MODE on line ~89 if desired (VS Code or nano)
tmux new -s train                 # tmux so it survives disconnect
python nm20275.py                 # detach: Ctrl+b then d ; reattach: tmux attach -t train
```
Results: wandb.ai (live, incl. SHAP images) + local `plots_part_b/`, `final_results.csv`,
`ablation_results.csv`, `best_model_*.pth`.

## Gotchas
- `.gitignore` excludes generated artifacts (`plots_part_b/`, `*.db`, `*.pth`, the two result
  CSVs). Only `nm20275.py` + `Data2026.csv` + this file are tracked.
- This shell environment (Windows laptop) lacks optuna/wandb/gplearn, so validation here was
  `py_compile` + an isolated SHAP smoke test — NOT a full end-to-end run. First real run happens
  on the Linux box.
- Optuna's wandb callback import is `from optuna.integration.wandb import WeightsAndBiasesCallback`;
  on newer Optuna you may need `pip install optuna-integration` (already in the install line).
