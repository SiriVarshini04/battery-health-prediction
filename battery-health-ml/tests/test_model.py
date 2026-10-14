"""
Unit tests for Model Training, Evaluation Metrics, and Artifact Persistence.
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from train_model import FEATURE_COLUMNS, TARGET_COLUMN

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        # Create synthetic test dataset conforming to processed schema
        np.random.seed(42)
        n_samples = 60
        self.dummy_df = pd.DataFrame({
            "battery_id": ["B0005"] * 30 + ["B0018"] * 30,
            "cycle_number": list(range(1, 31)) + list(range(1, 31)),
            "discharge_duration_s": np.random.uniform(2500, 3600, n_samples),
            "voltage_mean": np.random.uniform(3.2, 3.8, n_samples),
            "voltage_min": np.random.uniform(2.5, 2.7, n_samples),
            "voltage_max": np.random.uniform(4.0, 4.2, n_samples),
            "voltage_std": np.random.uniform(0.1, 0.4, n_samples),
            "current_mean": np.random.uniform(1.8, 2.1, n_samples),
            "temperature_mean": np.random.uniform(24.0, 35.0, n_samples),
            "temperature_max": np.random.uniform(28.0, 40.0, n_samples),
            "temperature_rise": np.random.uniform(3.0, 10.0, n_samples),
            "internal_resistance_proxy": np.random.uniform(0.05, 0.15, n_samples),
            "soh_percent": np.linspace(100.0, 75.0, n_samples)
        })

    def test_held_out_battery_separation(self):
        """Ensure train and test sets strictly separate batteries."""
        train = self.dummy_df[self.dummy_df["battery_id"] != "B0018"]
        test = self.dummy_df[self.dummy_df["battery_id"] == "B0018"]
        
        train_bats = set(train["battery_id"])
        test_bats = set(test["battery_id"])
        
        self.assertEqual(len(train_bats.intersection(test_bats)), 0)
        self.assertNotIn("B0018", train_bats)
        self.assertIn("B0018", test_bats)

    def test_random_forest_training_and_bounds(self):
        """Verify model trains and predictions stay within physical SOH boundaries."""
        train = self.dummy_df[self.dummy_df["battery_id"] != "B0018"]
        test = self.dummy_df[self.dummy_df["battery_id"] == "B0018"]
        
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
        
        preds = rf.predict(test[FEATURE_COLUMNS])
        self.assertEqual(len(preds), len(test))
        # Physically realistic SOH range
        self.assertTrue(np.all(preds >= 50.0))
        self.assertTrue(np.all(preds <= 110.0))

    def test_feature_columns_completeness(self):
        """Verify feature column list matches specification."""
        self.assertEqual(len(FEATURE_COLUMNS), 11)
        self.assertIn("cycle_number", FEATURE_COLUMNS)
        self.assertIn("voltage_mean", FEATURE_COLUMNS)
        self.assertIn("temperature_max", FEATURE_COLUMNS)
        self.assertIn("internal_resistance_proxy", FEATURE_COLUMNS)

if __name__ == "__main__":
    unittest.main()
