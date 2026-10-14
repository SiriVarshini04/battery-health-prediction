# Battery Health Prediction Data Dictionary & Schema Documentation

## 1. Dataset Overview

- **Source**: NASA Ames Prognostics Center of Excellence (PCoE) Li-ion Battery Aging Dataset
- **Cells**: B0005, B0006, B0007, B0018 (commercial 18650 cylindrical cells)
- **Cathode Chemistry**: Lithium Cobalt Oxide ($\text{LiCoO}_2$)
- **Anode Chemistry**: Graphite
- **Documented Rated Nominal Capacity ($C_{\text{rated}}$)**: **2.0 Ah (2000 mAh)**
- **End-of-Life (EOL) Standard Definition**: 30% fade in rated capacity ($\text{SOH} \le 70\%$, with first-life retirement typically at $\text{SOH} \le 80\%$)

---

## 2. Capacity-Based SOH Definition

$$\text{SOH } (\%) = \frac{\text{Measured Usable Capacity (Ah)}}{C_{\text{rated}}\text{ (Ah)}} \times 100$$

> [!IMPORTANT]
> SOH represents the irreversible chemical capacity loss across lifecycle aging. It must never be confused with SOC (State of Charge), which represents the reversible charge remaining during an individual discharge cycle.

---

## 3. Cycle Feature Definitions

| Feature Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `battery_id` | String | - | Identifier of the cell under test (`B0005`, `B0006`, `B0007`, `B0018`) |
| `cycle_number` | Integer | - | Sequence index of the discharge cycle (1..N) |
| `rated_capacity_ah` | Float | Ah | Nominal rated capacity (2.0 Ah) |
| `measured_capacity_ah`| Float | Ah | Ground-truth discharge capacity measured during standard CC discharge to cutoff |
| `soh_percent` | Float | % | **Target Variable**: Usable capacity expressed as percentage of rated capacity |
| `discharge_duration_s`| Float | Seconds | Total time duration of discharge cycle |
| `voltage_start` | Float | Volts | Initial terminal voltage under load at onset of discharge |
| `voltage_end` | Float | Volts | Final cutoff voltage reached at end of discharge |
| `voltage_min` | Float | Volts | Minimum cell terminal voltage recorded during cycle |
| `voltage_max` | Float | Volts | Maximum cell terminal voltage recorded during cycle |
| `voltage_mean` | Float | Volts | Mean terminal voltage across the discharge cycle |
| `voltage_std` | Float | Volts | Standard deviation of terminal voltage across discharge cycle |
| `current_mean` | Float | Amperes | Mean discharge current magnitude |
| `current_max` | Float | Amperes | Peak discharge current magnitude |
| `temperature_start` | Float | °C | Cell surface temperature at cycle initiation |
| `temperature_max` | Float | °C | Peak cell surface temperature during discharge |
| `temperature_mean` | Float | °C | Average cell surface temperature during discharge |
| `temperature_rise` | Float | °C | Thermal increase during discharge ($\Delta T = T_{\text{max}} - T_{\text{start}}$) |
| `internal_resistance_proxy` | Float | $\Omega$ | Initial ohmic drop proxy ($\Delta V / \Delta I$ at initial load step) |

---

## 4. Hardware Sensor to ML Feature Mapping

The physical hardware uses an **ESP32** reading an **INA219** sensor and a **DHT11** sensor connected to a **2S Li-ion battery pack** (2 cells in series, nominal 7.4V, max 8.4V, cutoff 6.0V).

To prevent silent feature mismatch, the inference engine applies the following transformations:

1. **Pack Voltage $\to$ Cell Voltage**:
   $$V_{\text{cell}} = \frac{V_{\text{pack}}}{2.0}$$
2. **Current Scaling**:
   $$I_{\text{cell}} (\text{A}) = \frac{I_{\text{pack}} (\text{mA})}{1000.0}$$
3. **Temperature**:
   $$T_{\text{cell}} = T_{\text{DHT11}} \text{ (°C)}$$
4. **Target Leakage Prevention**:
   Target capacity and SOH are strictly excluded from the inference feature set. Future cycle statistics are never passed into earlier cycle predictions.
