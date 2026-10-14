"""
Model Training Pipeline for Battery Health Prediction.

Trains:
1. Baseline Mean Regressor (Dummy)
2. Linear Degradation Baseline (SOH ~ cycle_number)
3. Primary Model: RandomForestRegressor

Validation Strategy:
- Held-out Battery Split (Train on B0005, B0006, B0007; Test on unseen B0018)
- Reports MAE, RMSE, and R2 on strictly held-out data.
Saves model artifact to models/random_forest_soh.joblib and metrics to reports/metrics.json.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_CSV = os.path.join(BASE_DIR, "data", "processed", "battery_cycles_processed.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")
MODEL_ARTIFACT_PATH = os.path.join(MODEL_DIR, "random_forest_soh.joblib")

# Core predictive features available without target leakage
FEATURE_COLUMNS = [
    "cycle_number",
    "discharge_duration_s",
    "voltage_mean",
    "voltage_min",
    "voltage_max",
    "voltage_std",
    "current_mean",
    "temperature_mean",
    "temperature_max",
    "temperature_rise",
    "internal_resistance_proxy"
]

TARGET_COLUMN = "soh_percent"

def train_and_evaluate(csv_path=PROCESSED_CSV, test_battery="B0018", random_state=42):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Processed dataset not found at {csv_path}. Run prepare_data.py first.")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {len(df)} cycles across batteries {df['battery_id'].unique().tolist()}")
    
    # Validation split: train on remaining batteries, evaluate on strictly held-out test battery
    available_batteries = df["battery_id"].unique().tolist()
    if test_battery not in available_batteries:
        test_battery = available_batteries[-1]
        print(f"Target test battery not found, defaulting to: {test_battery}")
        
    train_df = df[df["battery_id"] != test_battery].copy()
    test_df = df[df["battery_id"] == test_battery].copy()
    
    print(f"Training batteries: {train_df['battery_id'].unique().tolist()} ({len(train_df)} samples)")
    print(f"Held-out test battery: {test_battery} ({len(test_df)} samples)")
    
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMN].values
    
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMN].values
    
    # 1. Baseline Model 1: Mean Dummy Predictor
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    
    dummy_mae = mean_absolute_error(y_test, y_pred_dummy)
    dummy_rmse = root_mean_squared_error(y_test, y_pred_dummy)
    dummy_r2 = r2_score(y_test, y_pred_dummy)
    
    # 2. Baseline Model 2: Linear Cycle-Degradation Model
    linear = LinearRegression()
    linear.fit(X_train[:, [0]], y_train) # Fit solely on cycle_number
    y_pred_linear = linear.predict(X_test[:, [0]])
    
    linear_mae = mean_absolute_error(y_test, y_pred_linear)
    linear_rmse = root_mean_squared_error(y_test, y_pred_linear)
    linear_r2 = r2_score(y_test, y_pred_linear)
    
    # 3. Primary Model: Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_rmse = root_mean_squared_error(y_test, y_pred_rf)
    rf_r2 = r2_score(y_test, y_pred_rf)
    
    # Feature importances
    importances = dict(zip(FEATURE_COLUMNS, [round(float(v), 4) for v in rf.feature_importances_]))
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    
    metrics = {
        "evaluation_strategy": "Held-Out Battery Validation",
        "train_batteries": train_df["battery_id"].unique().tolist(),
        "test_battery": test_battery,
        "n_train_samples": len(train_df),
        "n_test_samples": len(test_df),
        "features": FEATURE_COLUMNS,
        "models": {
            "dummy_mean_baseline": {
                "mae": round(float(dummy_mae), 3),
                "rmse": round(float(dummy_rmse), 3),
                "r2": round(float(dummy_r2), 3)
            },
            "linear_cycle_baseline": {
                "mae": round(float(linear_mae), 3),
                "rmse": round(float(linear_rmse), 3),
                "r2": round(float(linear_r2), 3)
            },
            "random_forest": {
                "mae": round(float(rf_mae), 3),
                "rmse": round(float(rf_rmse), 3),
                "r2": round(float(rf_r2), 3)
            }
        },
        "feature_importances": sorted_importances
    }
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS ON HELD-OUT BATTERY:")
    print("="*50)
    print(f"Dummy Baseline       -> MAE: {dummy_mae:.3f}%, RMSE: {dummy_rmse:.3f}%, R2: {dummy_r2:.3f}")
    print(f"Linear Degradation   -> MAE: {linear_mae:.3f}%, RMSE: {linear_rmse:.3f}%, R2: {linear_r2:.3f}")
    print(f"Random Forest Model  -> MAE: {rf_mae:.3f}%, RMSE: {rf_rmse:.3f}%, R2: {rf_r2:.3f}")
    print("="*50)
    print("Top Feature Importances:")
    for feat, imp in list(sorted_importances.items())[:5]:
        print(f"  {feat:28s}: {imp * 100:.2f}%")
        
    # Save model artifact
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    artifact = {
        "model": rf,
        "model_type": "RandomForestRegressor",
        "feature_names": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "train_batteries": train_df["battery_id"].unique().tolist(),
        "test_battery": test_battery,
        "metrics": metrics["models"]["random_forest"],
        "random_state": random_state
    }
    joblib.dump(artifact, MODEL_ARTIFACT_PATH)
    print(f"\nSaved model artifact to: {MODEL_ARTIFACT_PATH}")
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to: {METRICS_PATH}")
    
    return rf, test_df, y_pred_rf, metrics

if __name__ == "__main__":
    train_and_evaluate()
