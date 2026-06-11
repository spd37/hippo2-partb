# HIPPO-2 Digital Twin (nm20275)

Neural-network virtual sensors for the HIPPO-2 hybrid diesel–electric engine — coursework for
*Special Ship Automation Systems 2026* (NTUA, Prof. G. Papalambrou). Predicts engine parameters
(NOx, Fuel Consumption, Intake Pressure, λ) from other measured signals.

- **Code:** `nm20275.py` — clean (3σ) → feature engineering (`none`/`poly`/`gplearn`) → deep MLP
  with Optuna → eval on held-out 1400 RPM (Leave-One-RPM-Out) → SHAP + ablation.
- **Data:** `Data2026.csv` (tab-separated; not redistributed publicly).
- **WandB:** https://wandb.ai/nm20275ntua/Advanced_Control_Systems
- **Results:** `final_results.csv` (best: NOx 13.56 ppm / R² 0.968, Fuel 0.285 kg/h / R² 0.9995).

## Run
```bash
pip install -r requirements.txt          # or see the header of nm20275.py
export WANDB_MODE=offline                 # or: wandb login
FEATURE_ENG_MODE=none python nm20275.py   # also: poly, gplearn
```

See `PPTX_BRIEF.md` / `THEORY_NOTES_GR.md` / `SPEAKER_NOTES.md` for presentation material.
