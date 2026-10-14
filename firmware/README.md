# ESP32 Battery Monitor Firmware Documentation

## 1. Hardware Pin Connections

| Component | Pin / Signal | ESP32 GPIO | Description |
| :--- | :--- | :--- | :--- |
| **INA219** | SDA | **GPIO 21** | I2C Data line |
| **INA219** | SCL | **GPIO 22** | I2C Clock line |
| **INA219** | VCC / GND | **3.3V / GND** | Sensor power supply |
| **DHT11** | DATA | **GPIO 18** | One-Wire temperature & humidity input |
| **DHT11** | VCC / GND | **3.3V / GND** | Power supply |
| **16x2 LCD** | RS | **GPIO 15** | Register Select |
| **16x2 LCD** | EN | **GPIO 2** | Enable signal |
| **16x2 LCD** | D4 | **GPIO 4** | Data Bit 4 |
| **16x2 LCD** | D5 | **GPIO 16** | Data Bit 5 |
| **16x2 LCD** | D6 | **GPIO 17** | Data Bit 6 |
| **16x2 LCD** | D7 | **GPIO 5** | Data Bit 7 |
| **16x2 LCD** | RW / VSS | **GND** | Ground reference |
| **16x2 LCD** | VDD | **5V / VBUS** | Power |
| **Active Buzzer** | (+) | **GPIO 23** | Local audible alarm |
| **Red LED** | Anode (+) | **GPIO 32** | Visual hazard indicator (via 220Ω resistor) |
| **Green LED** | Anode (+) | **GPIO 33** | Visual normal status indicator (via 220Ω resistor) |

---

## 2. Arduino IDE Setup & Required Libraries

Install the following libraries in Arduino IDE via **Sketch -> Include Library -> Manage Libraries**:

1. **Adafruit INA219** by Adafruit (v1.2.3 or later)
2. **DHT sensor library** by Adafruit (v1.4.6 or later) + **Adafruit Unified Sensor**
3. **LiquidCrystal** (built-in standard library)
4. **ThingSpeak** by MathWorks (v2.0.0 or later)

Select Board: **ESP32 Dev Module** in Arduino IDE (**Tools -> Board -> ESP32 Arduino -> ESP32 Dev Module**).

---

## 3. Network & API Configuration

Open [esp32_battery_monitor.ino](file:///d:/battery-health-prediction/firmware/esp32_battery_monitor/esp32_battery_monitor.ino) and adjust the network constants:

```cpp
// Wi-Fi Access Point Credentials
const char* WIFI_SSID = "Your_WiFi_SSID";
const char* WIFI_PASS = "Your_WiFi_Password";

// Computer's reachable LAN IP (Found using 'ipconfig' on Windows)
// NEVER use localhost (127.0.0.1) as the ESP32 is a separate physical device.
const char* ML_SERVER_URL = "http://172.18.148.38:5000/predict";
```

---

## 4. ThingSpeak Channel 443288 Configuration

### Field Mapping

| Field | Measurement / Metric | Source | Unit |
| :---: | :--- | :--- | :---: |
| **Field 1** | Pack Bus Voltage | INA219 Sensor | Volts (V) |
| **Field 2** | Load Current | INA219 Sensor | Milliamperes (mA) |
| **Field 3** | Power Output | INA219 Sensor | Milliwatts (mW) |
| **Field 4** | Temperature | DHT11 Sensor | Celsius (°C) |
| **Field 5** | Predicted SOH | Python Random Forest ML API | Percentage (%) |
| **Field 6** | Estimated SOC | Voltage Mapping (6.0V - 8.4V) | Percentage (%) |
| **Field 7** | Safety Status Code | Local Hardware Logic | Enum (0..3) |

*Safety Status Codes:*
- `0`: Normal / Safe
- `1`: High Temperature Alert (>45.0°C)
- `2`: Low Voltage Warning (<6.0V)
- `3`: Overcurrent Alert (>2500 mA)

### Update Cadence

Data is posted to ThingSpeak every **20 seconds**, respecting the free-tier limit of at least 15 seconds between posts.

---

## 5. Local Safety Controls

> [!IMPORTANT]
> The buzzer and alert LEDs execute deterministically inside the fast loop (every 500ms). They do **NOT** depend on Wi-Fi connection, ThingSpeak responsiveness, or the ML server.
> If the ML API is offline or returns an error, the firmware displays `SOH: OFFL` without fabricating dummy values, and continues local battery protection uninterrupted.

---

## 6. Troubleshooting

- **INA219 Fail on LCD**: Check I2C wiring (SDA to GPIO 21, SCL to GPIO 22). Ensure pull-up resistors are intact or INA219 module has built-in 10k pull-ups.
- **WiFi Disconnected**: Verify 2.4 GHz Wi-Fi credentials (ESP32 does not support 5 GHz Wi-Fi).
- **ML API Returns HTTP -1 or 503**:
  1. Confirm Flask server is running on the laptop: `python src/app.py`.
  2. Verify Windows Firewall allows incoming connections on port 5000.
  3. Confirm the ESP32 and host computer are connected to the same Wi-Fi subnet.
- **ThingSpeak Error -301**: Write request failed due to rate limiting. Ensure update interval is >= 15 seconds.
