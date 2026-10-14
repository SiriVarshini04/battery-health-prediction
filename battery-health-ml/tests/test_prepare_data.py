"""
Unit tests for Data Preparation and Feature Extraction logic.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from prepare_data import RATED_CAPACITY_AH, build_data_dictionary

class TestDataPreparation(unittest.TestCase):
    def test_rated_capacity_values(self):
        """Verify rated capacity constants for standard NASA 18650 cells."""
        for b_id in ["B0005", "B0006", "B0007", "B0018"]:
            self.assertIn(b_id, RATED_CAPACITY_AH)
            self.assertEqual(RATED_CAPACITY_AH[b_id], 2.0)

    def test_capacity_based_soh_calculation(self):
        """Verify SOH = (measured / rated) * 100 formula."""
        rated = 2.0
        # 1.85 Ah usable capacity -> 92.5% SOH
        self.assertAlmostEqual((1.85 / rated) * 100.0, 92.5, places=2)
        # 1.60 Ah usable capacity -> 80.0% SOH (Standard EOL)
        self.assertAlmostEqual((1.60 / rated) * 100.0, 80.0, places=2)
        # 1.40 Ah usable capacity -> 70.0% SOH
        self.assertAlmostEqual((1.40 / rated) * 100.0, 70.0, places=2)

    def test_data_dictionary_schema(self):
        """Ensure data dictionary contains required documentation and sensor mappings."""
        d = build_data_dictionary()
        self.assertIn("features", d)
        self.assertIn("esp32_hardware_mapping", d)
        self.assertIn("soh_percent", d["features"])
        self.assertIn("voltage_mean", d["features"])
        self.assertIn("internal_resistance_proxy", d["features"])
        self.assertIn("voltage_mapping", d["esp32_hardware_mapping"])

    def test_no_data_leakage(self):
        """Verify target variables are never in the training feature list."""
        from train_model import FEATURE_COLUMNS, TARGET_COLUMN
        self.assertNotIn("soh_percent", FEATURE_COLUMNS)
        self.assertNotIn("measured_capacity_ah", FEATURE_COLUMNS)
        self.assertNotIn("rated_capacity_ah", FEATURE_COLUMNS)
        self.assertNotIn("battery_id", FEATURE_COLUMNS)
        self.assertEqual(TARGET_COLUMN, "soh_percent")

if __name__ == "__main__":
    unittest.main()
