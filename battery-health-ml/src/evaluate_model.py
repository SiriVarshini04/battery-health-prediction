"""
Model Evaluation and Diagnostic Visualizations for Battery SOH Prediction.

Generates:
1. Actual vs Predicted SOH Scatter Plot (Held-Out Battery)
2. Cycle-by-Cycle SOH Degradation Tracking (Ground Truth vs Predicted)
3. Residuals & Error Distribution Plot
4. Feature Importance Horizontal Bar Chart
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_CSV = os.path.join(BASE_DIR, "data", "processed", "battery_cycles_processed.csv")
MODEL_ARTIFACT_PATH = os.path.join(BASE_DIR, "models", "random_forest_soh.joblib")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

def evaluate_and_plot():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_ARTIFACT_PATH):
        raise FileNotFoundError(f"Model artifact not found at {MODEL_ARTIFACT_PATH}. Run train_model.py first.")
    if not os.path.exists(PROCESSED_CSV):
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_CSV}. Run prepare_data.py first.")
        
    artifact = joblib.load(MODEL_ARTIFACT_PATH)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    test_battery = artifact["test_battery"]
    
    df = pd.read_csv(PROCESSED_CSV)
    test_df = df[df["battery_id"] == test_battery].copy()
    
    X_test = test_df[feature_names].values
    y_test = test_df["soh_percent"].values
    y_pred = model.predict(X_test)
    
    test_df["predicted_soh"] = y_pred
    test_df["residual_error"] = y_test - y_pred
    
    # Set plot styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. Actual vs Predicted SOH Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.8, color="#1f77b4", edgecolors="k", s=50, label="Test Predictions")
    min_val = min(min(y_test), min(y_pred)) - 2
    max_val = max(max(y_test), max(y_pred)) + 2
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal 1:1 Line")
    plt.xlabel("Actual Ground Truth SOH (%)", fontsize=12, fontweight="bold")
    plt.ylabel("Random Forest Predicted SOH (%)", fontsize=12, fontweight="bold")
    plt.title(f"Actual vs Predicted Battery SOH (Held-Out Cell: {test_battery})", fontsize=13, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "actual_vs_predicted_soh.png"), dpi=200)
    plt.close()
    
    # 2. Cycle-by-Cycle Degradation Trajectory Plot
    plt.figure(figsize=(9, 5))
    plt.plot(test_df["cycle_number"], test_df["soh_percent"], "k-o", label="Actual SOH (%)", linewidth=2, markersize=4)
    plt.plot(test_df["cycle_number"], test_df["predicted_soh"], "r--s", label="RF Predicted SOH (%)", linewidth=2, markersize=4)
    plt.axhline(80.0, color="orange", linestyle=":", label="80% SOH End-of-Life (EOL) Threshold")
    plt.xlabel("Discharge Cycle Number", fontsize=12, fontweight="bold")
    plt.ylabel("Battery State of Health (%)", fontsize=12, fontweight="bold")
    plt.title(f"SOH Degradation Over Full Lifecycle ({test_battery})", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "soh_degradation_trajectory.png"), dpi=200)
    plt.close()
    
    # 3. Residual Error Distribution Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(test_df["cycle_number"], test_df["residual_error"], color="#2ca02c", alpha=0.8, edgecolors="k")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Discharge Cycle Number", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Residual Error (Actual - Predicted) %", fontsize=11, fontweight="bold")
    axes[0].set_title("Residual Errors vs Cycle Number", fontsize=12, fontweight="bold")
    
    sns.histplot(test_df["residual_error"], kde=True, ax=axes[1], color="#2ca02c", bins=15)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual Error (%)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Frequency", fontsize=11, fontweight="bold")
    axes[1].set_title("Residual Error Distribution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "residuals_analysis.png"), dpi=200)
    plt.close()
    
    # 4. Feature Importance Horizontal Bar Plot
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices], color="#17becf", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=10)
    plt.xlabel("Relative Feature Importance (Gini)", fontsize=11, fontweight="bold")
    plt.title("Random Forest Feature Importances for SOH", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=200)
    plt.close()
    
    print(f"Generated 4 diagnostic figures in {FIGURES_DIR}:")
    print(" - actual_vs_predicted_soh.png")
    print(" - soh_degradation_trajectory.png")
    print(" - residuals_analysis.png")
    print(" - feature_importance.png")

if __name__ == "__main__":
    evaluate_and_plot()
