# Battery Health & SOH Forecasting ML Pipeline

## 1. System Overview

This subsystem implements reproducible data extraction, model training, evaluation, and real-time inference for Lithium-ion battery State of Health (SOH) and degradation forecasting.

### Models Implemented
1. **Dummy Mean Baseline**: Predicts average training SOH.
2. **Linear Degradation Baseline**: Fits SOH degradation purely against cycle count.
3. **Random Forest Regressor** (Primary Model): Multi-feature regressor capturing non-linear interactions across discharge duration, load current, voltage profile, and thermal dynamics.
4. **Degradation Forecaster**: Multi-cycle trajectory forecaster projecting future health degradation and estimating Remaining Useful Life (RUL) to an 80% EOL threshold.

---

## 2. Evaluation Results on Unseen Battery (Cell B0018)

Models were trained on NASA batteries **B0005, B0006, and B0007** (504 cycles) and evaluated strictly on held-out battery **B0018** (132 cycles):

| Model | MAE (%) | RMSE (%) | $R^2$ Score | Performance Summary |
| :--- | :---: | :---: | :---: | :--- |
| **Dummy Baseline (Mean)** | 7.034% | 7.861% | -0.038 | Fails to capture degradation |
| **Linear Cycle Degradation**| 5.178% | 5.519% | 0.488 | Over-simplifies non-linear aging |
| **Random Forest Regressor** | **1.897%** | **2.315%** | **0.910** | **Superior generalization on unseen cell** |

### Top Feature Importances (Random Forest)
1. `discharge_duration_s` (**53.31%**): Direct reflection of usable ampere-hour capacity.
2. `cycle_number` (**26.33%**): Sequence aging factor.
3. `current_mean` (**17.05%**): Discharge rate impact.
4. `voltage_mean` (**2.80%**): Polarization voltage drop.
5. `voltage_min` (**0.13%**): Cutoff threshold.

---

## 3. Future SOH Degradation Forecasting & RUL Estimation

Using strictly historical observations up to **Cycle 80**, models forecasted the remaining 88 future cycles of cell B0005:

- **Ground Truth EOL Cycle ($\text{SOH} \le 80\%$)**: Cycle 75
- **Linear Trend Extrapolation**:
  - Full Future Horizon MAE: **2.963%**
  - Projected EOL Cycle: **Cycle 85** (RUL Error: 10 cycles)
- **Polynomial Degradation Model**:
  - Full Future Horizon MAE: 17.420%
  - Projected EOL Cycle: **Cycle 80** (RUL Error: 5 cycles)

Diagnostic visualizations saved in [`reports/figures/`](file:///d:/battery-health-prediction/battery-health-ml/reports/figures):
- `actual_vs_predicted_soh.png`
- `soh_degradation_trajectory.png`
- `residuals_analysis.png`
- `feature_importance.png`
- `soh_future_forecast.png`

---

## 4. Setup and Execution Instructions (Windows)

### Step 1: Install Requirements
```powershell
py -3.13 -m pip install -r battery-health-ml\requirements.txt
```

### Step 2: Prepare Dataset
```powershell
py -3.13 battery-health-ml\src\prepare_data.py
```

### Step 3: Train and Evaluate Models
```powershell
py -3.13 battery-health-ml\src\train_model.py
py -3.13 battery-health-ml\src\evaluate_model.py
py -3.13 battery-health-ml\src\forecast_soh.py
```

### Step 4: Run Automated Tests
```powershell
py -3.13 -m pytest battery-health-ml\tests -v
```

### Step 5: Launch Inference REST API
```powershell
py -3.13 battery-health-ml\src\app.py
```

---

## 5. API Testing Examples

### Health Check:
```powershell
curl.exe -s http://127.0.0.1:5000/health
```

### Live ESP32 Telemetry Inference:
```powershell
curl.exe -s -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{\"pack_voltage\": 7.45, \"current_mA\": 850.0, \"temperature_c\": 28.5, \"cycle_number\": 35}'
```

Expected JSON response:
```json
{
  "estimated_usable_capacity_ah": 1.373,
  "health_state": "Critical (Replacement Recommended)",
  "input_validation": "passed",
  "model_status": "prediction_available",
  "predicted_soh_percent": 68.64
}
```
