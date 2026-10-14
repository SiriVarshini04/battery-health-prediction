"""
Battery SOH Future Degradation Forecasting and RUL Estimation.

Implements:
1. Strict Temporal Cutoff: Fits models strictly on early observed cycles (e.g. 1..N).
2. Defined Horizons: Evaluates multi-step ahead forecasts (H = 10, 20, 30, and full horizon to EOL).
3. Baseline: Linear Trend Extrapolation.
4. Primary Forecasting Model: Exponential Degradation / Polynomial AR Regressor.
5. End-of-Life (EOL) Threshold: 80.0% SOH.
6. Remaining Useful Life (RUL) evaluation with absolute cycle error.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_CSV = os.path.join(BASE_DIR, "data", "processed", "battery_cycles_processed.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
FORECAST_METRICS_PATH = os.path.join(REPORTS_DIR, "forecast_metrics.json")

EOL_SOH_THRESHOLD = 80.0 # Standard 80% capacity retention end-of-life criteria

def run_forecasting(battery_id="B0005", cutoff_cycle=80):
    if not os.path.exists(PROCESSED_CSV):
        raise FileNotFoundError(f"Processed dataset not found at {PROCESSED_CSV}")
        
    df = pd.read_csv(PROCESSED_CSV)
    b_df = df[df["battery_id"] == battery_id].sort_values("cycle_number").copy()
    
    if len(b_df) < cutoff_cycle + 10:
        cutoff_cycle = int(len(b_df) * 0.55)
        print(f"Adjusted cutoff cycle to {cutoff_cycle} based on available records ({len(b_df)})")
        
    print(f"Evaluating battery {battery_id}: {len(b_df)} total cycles.")
    print(f"Observed training history: cycles 1 to {cutoff_cycle}")
    print(f"Unseen future evaluation: cycles {cutoff_cycle + 1} to {b_df['cycle_number'].max()}")
    
    train_data = b_df[b_df["cycle_number"] <= cutoff_cycle]
    future_data = b_df[b_df["cycle_number"] > cutoff_cycle]
    
    t_train = train_data["cycle_number"].values.reshape(-1, 1)
    y_train = train_data["soh_percent"].values
    
    t_future = future_data["cycle_number"].values.reshape(-1, 1)
    y_future = future_data["soh_percent"].values
    
    # 1. Baseline Model: Linear Extrapolation
    linear_model = LinearRegression()
    linear_model.fit(t_train, y_train)
    y_pred_future_linear = linear_model.predict(t_future)
    
    # 2. Polynomial Degradation Model (captures accelerated late-stage aging)
    poly_features_train = np.hstack([t_train, t_train**2])
    poly_features_future = np.hstack([t_future, t_future**2])
    poly_model = LinearRegression()
    poly_model.fit(poly_features_train, y_train)
    y_pred_future_poly = poly_model.predict(poly_features_future)
    
    # Metrics across future cycles
    linear_mae = mean_absolute_error(y_future, y_pred_future_linear)
    linear_rmse = root_mean_squared_error(y_future, y_pred_future_linear)
    
    poly_mae = mean_absolute_error(y_future, y_pred_future_poly)
    poly_rmse = root_mean_squared_error(y_future, y_pred_future_poly)
    
    # Horizon-specific metrics (H=10, H=20, H=30 cycles ahead)
    horizon_metrics = {}
    for h in [10, 20, 30]:
        if len(y_future) >= h:
            h_y = y_future[:h]
            h_lin = y_pred_future_linear[:h]
            h_poly = y_pred_future_poly[:h]
            horizon_metrics[f"horizon_{h}_cycles"] = {
                "linear_mae": round(float(mean_absolute_error(h_y, h_lin)), 3),
                "poly_mae": round(float(mean_absolute_error(h_y, h_poly)), 3)
            }
            
    # Remaining Useful Life (RUL) calculation
    # Actual EOL cycle: first cycle where actual SOH < EOL_SOH_THRESHOLD
    actual_eol_cycles = b_df[b_df["soh_percent"] <= EOL_SOH_THRESHOLD]["cycle_number"].values
    actual_eol_cycle = int(actual_eol_cycles[0]) if len(actual_eol_cycles) > 0 else int(b_df["cycle_number"].max())
    actual_rul_at_cutoff = max(0, actual_eol_cycle - cutoff_cycle)
    
    # Projected EOL cycles
    # For linear: m*t + c = 80 => t = (80 - c) / m
    m = linear_model.coef_[0]
    c = linear_model.intercept_
    if m < 0:
        pred_eol_linear = int(round((EOL_SOH_THRESHOLD - c) / m))
    else:
        pred_eol_linear = 9999
    linear_rul_error = abs(pred_eol_linear - actual_eol_cycle)
    
    # For polynomial: project future sequence up to 300 cycles
    proj_cycles = np.arange(cutoff_cycle, cutoff_cycle + 250).reshape(-1, 1)
    proj_poly = poly_model.predict(np.hstack([proj_cycles, proj_cycles**2]))
    below_eol_idx = np.where(proj_poly <= EOL_SOH_THRESHOLD)[0]
    if len(below_eol_idx) > 0:
        pred_eol_poly = int(proj_cycles[below_eol_idx[0]][0])
    else:
        pred_eol_poly = 9999
    poly_rul_error = abs(pred_eol_poly - actual_eol_cycle)
    
    results = {
        "battery_id": battery_id,
        "cutoff_cycle": cutoff_cycle,
        "total_cycles": len(b_df),
        "future_cycles_evaluated": len(future_data),
        "eol_threshold_soh_percent": EOL_SOH_THRESHOLD,
        "actual_eol_cycle": actual_eol_cycle,
        "actual_rul_from_cutoff": actual_rul_at_cutoff,
        "forecast_metrics": {
            "linear_baseline": {
                "full_future_mae": round(float(linear_mae), 3),
                "full_future_rmse": round(float(linear_rmse), 3),
                "predicted_eol_cycle": pred_eol_linear,
                "rul_error_cycles": linear_rul_error
            },
            "polynomial_degradation": {
                "full_future_mae": round(float(poly_mae), 3),
                "full_future_rmse": round(float(poly_rmse), 3),
                "predicted_eol_cycle": pred_eol_poly,
                "rul_error_cycles": poly_rul_error
            }
        },
        "horizon_breakdown": horizon_metrics
    }
    
    print("\n" + "="*50)
    print(f"FUTURE SOH FORECASTING RESULTS ({battery_id}):")
    print("="*50)
    print(f"Ground Truth EOL Cycle (SOH <= 80%): Cycle {actual_eol_cycle} (Actual RUL = {actual_rul_at_cutoff} cycles)")
    print(f"Linear Baseline:      MAE: {linear_mae:.3f}%, Projected EOL: {pred_eol_linear} (RUL Error: {linear_rul_error} cycles)")
    print(f"Polynomial Forecaster: MAE: {poly_mae:.3f}%, Projected EOL: {pred_eol_poly} (RUL Error: {poly_rul_error} cycles)")
    print("="*50)
    
    # Plot forecast trajectory
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_data["cycle_number"], train_data["soh_percent"], "bo-", label="Observed History (Cycles 1..{})".format(cutoff_cycle), markersize=4)
    plt.plot(future_data["cycle_number"], future_data["soh_percent"], "k.-", label="Actual Future SOH (Ground Truth)", markersize=5)
    plt.plot(future_data["cycle_number"], y_pred_future_linear, "g--", label=f"Linear Forecast (MAE: {linear_mae:.2f}%)", linewidth=2)
    plt.plot(future_data["cycle_number"], y_pred_future_poly, "r-", label=f"Polynomial Forecast (MAE: {poly_mae:.2f}%)", linewidth=2)
    plt.axvline(cutoff_cycle, color="gray", linestyle=":", label=f"Cutoff (Cycle {cutoff_cycle})")
    plt.axhline(EOL_SOH_THRESHOLD, color="orange", linestyle="--", label="80% SOH EOL Threshold")
    plt.xlabel("Cycle Number", fontsize=11, fontweight="bold")
    plt.ylabel("State of Health SOH (%)", fontsize=11, fontweight="bold")
    plt.title(f"Multi-Cycle SOH Degradation Forecast & RUL Estimation ({battery_id})", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "soh_future_forecast.png"), dpi=200)
    plt.close()
    
    with open(FORECAST_METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved forecasting metrics to: {FORECAST_METRICS_PATH}")
    print(f"Saved forecast plot to: {os.path.join(FIGURES_DIR, 'soh_future_forecast.png')}")
    return results

if __name__ == "__main__":
    run_forecasting()
