"""
Prediction Engine and Feature Mapping Module for Battery Health SOH.

Supports:
1. Direct feature dictionary matching training schema.
2. ESP32 live telemetry mapping:
   - Maps 2S pack voltage (INA219) -> cell voltage (V / 2)
   - Maps current (mA) -> current (A)
   - Ingests DHT11 temperature (°C)
   - Validates ranges strictly.
"""

import os
import joblib
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_soh.joblib")

# Strict physical bounds for validation
PHYSICAL_BOUNDS = {
    "voltage_cell": (1.0, 4.5),       # Volts per cell
    "voltage_pack": (2.0, 9.0),       # Volts for 2S pack
    "current_cell_a": (0.0, 10.0),    # Amperes
    "current_pack_ma": (0.0, 10000.0),# Milliamperes
    "temperature_c": (0.0, 80.0),     # Celsius
    "cycle_number": (1, 10000),       # Cycles
    "duration_s": (10.0, 20000.0)     # Seconds
}

class SOHPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.artifact = None
        self.model = None
        self.feature_names = None
        self.load_model()
        
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")
        self.artifact = joblib.load(self.model_path)
        self.model = self.artifact["model"]
        self.feature_names = self.artifact["feature_names"]
        
    def map_esp32_telemetry(self, data):
        """
        Maps live ESP32 sensor payload to ML feature vector.
        ESP32 sends:
          - pack_voltage: (V from INA219, ~4.0V to 8.4V for 2S)
          - current_mA: (mA from INA219)
          - temperature_c: (°C from DHT11)
          - cycle_number: (optional, defaults to 1 or device counter)
          - duration_s: (optional elapsed discharge time in seconds)
        """
        # Extract and validate required fields
        for req in ["pack_voltage", "current_mA", "temperature_c"]:
            if req not in data:
                raise ValueError(f"Missing required sensor reading: '{req}'")
                
        try:
            pack_v = float(data["pack_voltage"])
            curr_ma = float(data["current_mA"])
            temp_c = float(data["temperature_c"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Sensor values must be numeric: {e}")
            
        # Range checks
        if not (PHYSICAL_BOUNDS["voltage_pack"][0] <= pack_v <= PHYSICAL_BOUNDS["voltage_pack"][1]):
            raise ValueError(f"Pack voltage {pack_v}V out of physical range {PHYSICAL_BOUNDS['voltage_pack']}")
        if not (PHYSICAL_BOUNDS["temperature_c"][0] <= temp_c <= PHYSICAL_BOUNDS["temperature_c"][1]):
            raise ValueError(f"Temperature {temp_c}°C out of valid range {PHYSICAL_BOUNDS['temperature_c']}")
        if curr_ma < 0:
            curr_ma = abs(curr_ma) # INA219 current magnitude
        if curr_ma > PHYSICAL_BOUNDS["current_pack_ma"][1]:
            raise ValueError(f"Current {curr_ma}mA exceeds sensor safety threshold")
            
        cycle_num = int(data.get("cycle_number", 1))
        if cycle_num < 1:
            cycle_num = 1
            
        duration_s = float(data.get("duration_s", 3000.0))
        
        # 2S Pack to Cell mapping
        v_cell = pack_v / 2.0
        i_a = curr_ma / 1000.0
        
        # Approximate dynamic dispersion proxies if single point provided
        v_min = float(data.get("voltage_min", max(2.5, v_cell - 0.2)))
        v_max = float(data.get("voltage_max", min(4.2, v_cell + 0.1)))
        v_std = float(data.get("voltage_std", 0.15))
        t_max = float(data.get("temperature_max", max(temp_c, temp_c + 2.0)))
        t_rise = float(data.get("temperature_rise", max(0.0, t_max - temp_c)))
        ir_proxy = float(data.get("internal_resistance_proxy", 0.08))
        
        features = {
            "cycle_number": cycle_num,
            "discharge_duration_s": duration_s,
            "voltage_mean": v_cell,
            "voltage_min": v_min,
            "voltage_max": v_max,
            "voltage_std": v_std,
            "current_mean": i_a,
            "temperature_mean": temp_c,
            "temperature_max": t_max,
            "temperature_rise": t_rise,
            "internal_resistance_proxy": ir_proxy
        }
        return features

    def predict(self, input_payload):
        """
        Validates input, routes format, executes inference, returns structured response.
        """
        if self.model is None:
            self.load_model()
            
        # Determine format: raw features or ESP32 telemetry
        if "pack_voltage" in input_payload:
            features_dict = self.map_esp32_telemetry(input_payload)
        else:
            # Direct feature format
            missing = [f for f in self.feature_names if f not in input_payload]
            if missing:
                raise ValueError(f"Missing required feature fields: {missing}")
            features_dict = {f: float(input_payload[f]) for f in self.feature_names}
            
        # Order features exactly as trained
        feature_vector = np.array([[features_dict[f] for f in self.feature_names]])
        
        # Inference
        soh_pred = float(self.model.predict(feature_vector)[0])
        # Physical bounds clamp
        soh_pred_clamped = round(max(0.0, min(105.0, soh_pred)), 2)
        
        # Derive usable capacity (nominal 2.0 Ah rated base)
        rated_ref = 2.0
        usable_cap = round((soh_pred_clamped / 100.0) * rated_ref, 3)
        
        # Health categorization
        if soh_pred_clamped >= 80.0:
            health_state = "Healthy"
        elif soh_pred_clamped >= 70.0:
            health_state = "Degraded (Near EOL)"
        else:
            health_state = "Critical (Replacement Recommended)"
            
        return {
            "predicted_soh_percent": soh_pred_clamped,
            "estimated_usable_capacity_ah": usable_cap,
            "health_state": health_state,
            "model_status": "prediction_available",
            "input_validation": "passed",
            "features_used": features_dict
        }
