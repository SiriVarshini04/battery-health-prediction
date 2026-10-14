"""
Dataset Preparation Pipeline for NASA Li-ion Battery Aging Dataset.

Extracts cycle-level measurements from raw MATLAB (.mat) files for batteries
B0005, B0006, B0007, and B0018. Calculates true capacity-based SOH:
    SOH (%) = (measured_capacity_Ah / rated_capacity_Ah) * 100

Extracts voltage, current, temperature, duration, and internal resistance features.
Saves processed CSV and data dictionary.
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.io

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PROCESSED_CSV = os.path.join(PROCESSED_DIR, "battery_cycles_processed.csv")
DATA_DICT_PATH = os.path.join(PROCESSED_DIR, "data_dictionary.json")

# Documented rated capacity for NASA Ames 18650 Li-ion cells (commercial LiCoO2 cells)
RATED_CAPACITY_AH = {
    "B0005": 2.0,
    "B0006": 2.0,
    "B0007": 2.0,
    "B0018": 2.0
}

def find_mat_files(search_dir):
    """Recursively search for target battery .mat files."""
    target_batteries = ["B0005", "B0006", "B0007", "B0018"]
    found = {}
    for root, _, files in os.walk(search_dir):
        for f in files:
            for b in target_batteries:
                if f.upper() == f"{b}.MAT" or f.upper().startswith(f"{b}_"):
                    if b not in found:
                        found[b] = os.path.join(root, f)
    return found

def extract_battery_cycles(battery_id, mat_file_path, rated_capacity):
    """
    Parse a single NASA battery .mat file and extract discharge cycle metrics.
    """
    print(f"Loading {battery_id} from {mat_file_path}...")
    mat = scipy.io.loadmat(mat_file_path)
    
    # Key usually matches battery name (e.g. 'B0005')
    key = None
    for k in mat.keys():
        if not k.startswith("__"):
            key = k
            break
            
    if not key:
        raise ValueError(f"No valid data structure key found in {mat_file_path}")
        
    struct = mat[key]
    cycles_raw = struct["cycle"][0, 0]
    num_total_cycles = cycles_raw.shape[1] if len(cycles_raw.shape) > 1 else len(cycles_raw)
    print(f"Total operational records in {battery_id}: {num_total_cycles}")
    
    records = []
    discharge_cycle_num = 0
    
    for i in range(num_total_cycles):
        cycle = cycles_raw[0, i] if len(cycles_raw.shape) > 1 else cycles_raw[i]
        cycle_type = str(cycle["type"][0])
        
        # We focus on discharge cycles where capacity and degradation are measured
        if cycle_type != "discharge":
            continue
            
        data = cycle["data"][0, 0]
        
        # Check if Capacity is present and valid
        if "Capacity" not in data.dtype.names:
            continue
            
        try:
            capacity_raw = data["Capacity"]
            if capacity_raw.size == 0:
                continue
            capacity = float(capacity_raw[0][0])
        except (IndexError, TypeError, ValueError):
            continue
            
        # Extract time series measurements
        voltage = np.array(data["Voltage_measured"][0], dtype=float)
        current = np.array(data["Current_measured"][0], dtype=float)
        temp = np.array(data["Temperature_measured"][0], dtype=float)
        time_vec = np.array(data["Time"][0], dtype=float)
        
        # Handle empty or corrupted measurements
        if len(voltage) < 5 or len(current) < 5 or len(temp) < 5:
            continue
            
        discharge_cycle_num += 1
        
        # Derived physical cycle features
        duration_s = float(time_vec[-1] - time_vec[0]) if len(time_vec) > 1 else 0.0
        v_start = float(voltage[0])
        v_end = float(voltage[-1])
        v_min = float(np.min(voltage))
        v_max = float(np.max(voltage))
        v_mean = float(np.mean(voltage))
        v_std = float(np.std(voltage))
        
        # Discharge current magnitude (typically negative in raw NASA records)
        i_abs = np.abs(current)
        i_mean = float(np.mean(i_abs))
        i_max = float(np.max(i_abs))
        
        # Thermal characteristics
        t_start = float(temp[0])
        t_max = float(np.max(temp))
        t_mean = float(np.mean(temp))
        t_rise = float(t_max - t_start)
        
        # Internal resistance proxy: initial voltage drop / initial loaded current
        if len(voltage) > 2 and i_abs[1] > 0.1:
            ir_proxy = float((v_start - voltage[1]) / i_abs[1])
        else:
            ir_proxy = 0.0
            
        # Target capacity and SOH
        soh_pct = (capacity / rated_capacity) * 100.0
        
        # Basic sanity filters (reject corrupted/dummy readouts)
        if capacity <= 0 or capacity > rated_capacity * 1.5:
            continue
        if v_mean < 1.0 or v_max > 5.0:
            continue
            
        records.append({
            "battery_id": battery_id,
            "cycle_number": discharge_cycle_num,
            "rated_capacity_ah": rated_capacity,
            "measured_capacity_ah": round(capacity, 5),
            "soh_percent": round(soh_pct, 3),
            "discharge_duration_s": round(duration_s, 2),
            "voltage_start": round(v_start, 4),
            "voltage_end": round(v_end, 4),
            "voltage_min": round(v_min, 4),
            "voltage_max": round(v_max, 4),
            "voltage_mean": round(v_mean, 4),
            "voltage_std": round(v_std, 4),
            "current_mean": round(i_mean, 4),
            "current_max": round(i_max, 4),
            "temperature_start": round(t_start, 2),
            "temperature_max": round(t_max, 2),
            "temperature_mean": round(t_mean, 2),
            "temperature_rise": round(t_rise, 2),
            "internal_resistance_proxy": round(ir_proxy, 5)
        })
        
    print(f"Extracted {len(records)} valid discharge cycles for {battery_id}.")
    return records

def build_data_dictionary():
    """Define metadata schema and sensor mapping documentation."""
    return {
        "dataset_name": "NASA Ames Prognostics Center of Excellence (PCoE) Li-ion Battery Aging Dataset",
        "reference_batteries": ["B0005", "B0006", "B0007", "B0018"],
        "cell_chemistry": "Lithium Cobalt Oxide (LiCoO2) cathode, graphite anode (18650 cylindrical)",
        "rated_nominal_capacity_Ah": 2.0,
        "soh_definition": "Capacity-based SOH: (measured_capacity_Ah / rated_capacity_Ah) * 100",
        "features": {
            "battery_id": "Identifier of the battery cell under test (B0005, B0006, B0007, B0018)",
            "cycle_number": "Ordered discharge cycle count starting from cycle 1 to end-of-life",
            "rated_capacity_ah": "Manufacturer nominal rated capacity in Ampere-hours (2.0 Ah)",
            "measured_capacity_ah": "Ground-truth usable discharge capacity in Ampere-hours measured during standard CC discharge to cutoff",
            "soh_percent": "Ground-truth target State of Health as percentage of rated capacity (Target)",
            "discharge_duration_s": "Total time elapsed in seconds during discharge cycle",
            "voltage_start": "Initial cell voltage at start of discharge (V)",
            "voltage_end": "Final cell voltage at end of discharge / cutoff threshold (V)",
            "voltage_min": "Minimum cell voltage recorded during cycle (V)",
            "voltage_max": "Maximum cell voltage recorded during cycle (V)",
            "voltage_mean": "Mean cell voltage across discharge period (V)",
            "voltage_std": "Standard deviation of cell voltage during discharge (V)",
            "current_mean": "Mean discharge current magnitude (A)",
            "current_max": "Peak discharge current magnitude (A)",
            "temperature_start": "Initial cell surface temperature at start of cycle (°C)",
            "temperature_max": "Maximum cell surface temperature recorded during discharge (°C)",
            "temperature_mean": "Mean surface temperature across cycle (°C)",
            "temperature_rise": "Temperature rise (temperature_max - temperature_start) during discharge (°C)",
            "internal_resistance_proxy": "Initial ohmic IR voltage drop delta_V / delta_I (Ohms)"
        },
        "esp32_hardware_mapping": {
            "description": "Hardware adapter mapping live ESP32 sensors (INA219 + DHT11) to cell ML features",
            "battery_configuration": "2S Li-ion pack (2 cells in series, nominal 7.4V)",
            "voltage_mapping": "Cell voltage = Pack bus voltage (INA219 V) / 2.0",
            "current_mapping": "Cell current = Pack load current (INA219 mA) / 1000.0 (in Amperes)",
            "temperature_mapping": "Ambient / pack casing temperature from DHT11 (°C)",
            "note_on_leakage": "Future cycle measurements and target capacity are strictly excluded from inference feature set."
        }
    }

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    mat_files = find_mat_files(RAW_DIR)
    if not mat_files:
        print(f"No .mat files found in {RAW_DIR}. Please run download_data.py or fast_download.py first.")
        return False
        
    print(f"Found target battery files: {list(mat_files.keys())}")
    
    all_records = []
    for b_id, file_path in mat_files.items():
        rated = RATED_CAPACITY_AH.get(b_id, 2.0)
        records = extract_battery_cycles(b_id, file_path, rated)
        all_records.extend(records)
        
    if not all_records:
        print("No valid discharge records extracted.")
        return False
        
    df = pd.DataFrame(all_records)
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"\nSaved processed dataset to {PROCESSED_CSV}")
    print(f"Total samples: {len(df)}")
    print(f"Batteries included: {df['battery_id'].value_counts().to_dict()}")
    print("\nSummary Statistics:")
    print(df[["cycle_number", "measured_capacity_ah", "soh_percent", "voltage_mean", "temperature_max"]].describe().to_string())
    
    data_dict = build_data_dictionary()
    with open(DATA_DICT_PATH, "w") as f:
        json.dump(data_dict, f, indent=2)
    print(f"Saved data dictionary to {DATA_DICT_PATH}")
    return True

if __name__ == "__main__":
    main()
