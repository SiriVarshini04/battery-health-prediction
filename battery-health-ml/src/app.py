"""
Flask REST API for Battery Health (SOH) Prediction.

Endpoints:
  GET  /health     - Liveness & model readiness check
  POST /predict    - Real-time SOH inference from ESP32 telemetry or feature dictionary
  GET  /features   - Schema documentation of expected inputs

Designed for local LAN deployment with ESP32 microcontrollers.
"""

import os
import sys
from flask import Flask, request, jsonify
from predict import SOHPredictor

app = Flask(__name__)

# Initialize predictor on startup
try:
    predictor = SOHPredictor()
    model_loaded = True
    load_error = None
except Exception as e:
    predictor = None
    model_loaded = False
    load_error = str(e)
    print(f"Warning during API initialization: {load_error}", file=sys.stderr)

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Battery Health & SOH Prediction API",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "predict": "/predict [POST]",
            "features": "/features"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    if model_loaded and predictor is not None:
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "model_type": predictor.artifact.get("model_type", "RandomForestRegressor"),
            "test_battery": predictor.artifact.get("test_battery", "Unknown"),
            "model_mae": predictor.artifact.get("metrics", {}).get("mae")
        }), 200
    else:
        return jsonify({
            "status": "unhealthy",
            "model_loaded": False,
            "error": load_error
        }), 503

@app.route("/features", methods=["GET"])
def features():
    if predictor is None:
        return jsonify({"error": "Predictor uninitialized"}), 503
    return jsonify({
        "expected_direct_features": predictor.feature_names,
        "esp32_telemetry_format": {
            "pack_voltage": "Measured battery pack voltage in Volts (e.g. 7.4)",
            "current_mA": "Measured load current in mA (e.g. 500)",
            "temperature_c": "Ambient/pack temperature in Celsius (e.g. 26.5)",
            "cycle_number": "Optional integer cycle count (e.g. 25)",
            "duration_s": "Optional discharge duration in seconds"
        }
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded or predictor is None:
        return jsonify({
            "model_status": "unavailable",
            "error": "Model artifact not loaded. Train model before calling inference."
        }), 503
        
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({
            "model_status": "error",
            "input_validation": "failed",
            "error": "Request body must be valid JSON."
        }), 400
        
    try:
        result = predictor.predict(payload)
        return jsonify(result), 200
    except ValueError as val_err:
        return jsonify({
            "model_status": "rejected",
            "input_validation": "failed",
            "error": str(val_err)
        }), 400
    except Exception as exc:
        return jsonify({
            "model_status": "error",
            "input_validation": "failed",
            "error": f"Internal inference error: {str(exc)}"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Battery SOH Inference API on 0.0.0.0:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
