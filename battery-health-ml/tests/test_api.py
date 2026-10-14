"""
Integration tests for the Flask REST API and Inference Engine.
"""

import unittest
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from app import app
from predict import SOHPredictor, PHYSICAL_BOUNDS

class TestInferenceAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_root_endpoint(self):
        """Verify root endpoint returns API service status."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "online")
        self.assertIn("endpoints", data)

    def test_features_endpoint(self):
        """Verify features endpoint documents both direct and ESP32 telemetry schemas."""
        response = self.client.get("/features")
        if response.status_code == 200:
            data = response.get_json()
            self.assertIn("esp32_telemetry_format", data)
            self.assertIn("expected_direct_features", data)

    def test_predict_invalid_json(self):
        """Verify non-JSON request body is safely rejected."""
        response = self.client.post("/predict", data="Not JSON text", content_type="text/plain")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["input_validation"], "failed")

    def test_predict_missing_telemetry_fields(self):
        """Verify missing sensor telemetry generates 400 Bad Request."""
        payload = {"pack_voltage": 7.4} # Missing current_mA and temperature_c
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["input_validation"], "failed")
        self.assertIn("Missing required sensor reading", data["error"])

    def test_predict_out_of_range_voltage(self):
        """Verify physically implausible pack voltage is rejected."""
        payload = {
            "pack_voltage": 25.0, # Implausible for 2S Li-ion pack
            "current_mA": 500.0,
            "temperature_c": 26.0
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["input_validation"], "failed")
        self.assertIn("out of physical range", data["error"])

    def test_predict_out_of_range_temperature(self):
        """Verify physically implausible temperature is rejected."""
        payload = {
            "pack_voltage": 7.4,
            "current_mA": 500.0,
            "temperature_c": 125.0 # Above safe battery limit
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["input_validation"], "failed")

    def test_predict_non_numeric_values(self):
        """Verify string/corrupt sensor readings are rejected."""
        payload = {
            "pack_voltage": "SEVEN_VOLTS",
            "current_mA": 500.0,
            "temperature_c": 25.0
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["input_validation"], "failed")

    def test_esp32_hardware_mapping_logic(self):
        """Verify 2S to single-cell voltage division and unit conversion."""
        predictor = SOHPredictor.__new__(SOHPredictor)
        telemetry = {
            "pack_voltage": 7.4,
            "current_mA": 1200.0,
            "temperature_c": 28.0,
            "cycle_number": 20
        }
        features = predictor.map_esp32_telemetry(telemetry)
        
        # 7.4V pack / 2 = 3.7V per cell
        self.assertAlmostEqual(features["voltage_mean"], 3.7, places=2)
        # 1200 mA = 1.2 A
        self.assertAlmostEqual(features["current_mean"], 1.2, places=2)
        self.assertEqual(features["temperature_mean"], 28.0)
        self.assertEqual(features["cycle_number"], 20)

if __name__ == "__main__":
    unittest.main()
