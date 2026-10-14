/*
 * ==============================================================================
 * Project: Battery Health Forecasting & SOH Prediction Using ESP32 and Machine Learning
 * Target Hardware: ESP32 Dev Module (WROOM-32)
 * Sensors: Adafruit INA219 (I2C), DHT11 (GPIO 18)
 * Display: 16x2 Character LCD (Parallel 4-bit mode)
 * Alerts: Active Buzzer (GPIO 23), Red LED (GPIO 32), Green LED (GPIO 33)
 * Cloud: ThingSpeak Channel 443288
 * ML Engine: Python Flask REST API running Random Forest SOH Regressor on Local LAN
 * ==============================================================================
 */

#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Adafruit_INA219.h>
#include <LiquidCrystal.h>
#include <ThingSpeak.h>
#include "DHT.h"

// ======================== HARDWARE PIN ASSIGNMENTS ========================
// 16x2 LCD Pin Definitions (4-bit interface)
const int rs = 15, en = 2, d4 = 4, d5 = 16, d6 = 17, d7 = 5;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

// INA219 Current & Voltage Sensor (I2C)
// SDA = GPIO 21, SCL = GPIO 22
Adafruit_INA219 ina219;

// DHT11 Temperature & Humidity Sensor
#define DHTPIN 18
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// Local Safety Alert Indicators
const int buzPin   = 23; // Active buzzer
const int redPin   = 32; // Red Alert LED
const int greenPin = 33; // Green Normal LED

// ======================== BATTERY SPECIFICATIONS ========================
// Battery Chemistry: 2S Lithium-Ion 18650 Battery Pack
// Rated Nominal Voltage: 7.4V (3.7V / cell)
// Rated Capacity: 2.2 Ah (2200 mAh)
const float BATTERY_RATED_CAPACITY_AH = 2.2;
const float MIN_BATTERY_VOLTAGE       = 6.0;  // 0% SOC Cutoff threshold for 2S pack (3.0V/cell)
const float MAX_BATTERY_VOLTAGE       = 8.4;  // 100% SOC Fully charged threshold for 2S pack (4.2V/cell)

// Hardware Safety Thresholds (Operate independently of ML API)
const float ALERT_OVERTEMP_C          = 45.0; // High Temperature Alert limit (°C)
const float ALERT_LOW_VOLT_V          = 6.0;  // Critical Undervoltage threshold (V)
const float ALERT_OVERCURRENT_MA      = 2500.0;// Overload protection threshold (mA)

// ======================== NETWORK & CLOUD CONFIGURATION ========================
// Wi-Fi Access Point Credentials
const char* WIFI_SSID = "project";
const char* WIFI_PASS = "12345678";

// Local Python ML Inference Server (Use reachable LAN IP of host PC, NOT localhost)
// Default host IP discovered: 172.18.148.38 (update if network changes)
const char* ML_SERVER_URL = "http://172.18.148.38:5000/predict";
const int   ML_HTTP_TIMEOUT_MS = 2000; // 2-second timeout to prevent blocking sensor loop

// ThingSpeak Telemetry Configuration
WiFiClient client;
unsigned long myChannelNumber = 443288;          // Channel ID
const char*   myWriteAPIKey   = "6BB9V3OTG8797LD9"; // Write API Key

// ======================== TIMING & TELEMETRY VARIABLES ========================
unsigned long lastSensorLoopTime    = 0;
const unsigned long sensorInterval   = 500;   // Sensor sampling & safety alert check every 500 ms

unsigned long lastMLInferenceTime   = 0;
const unsigned long mlInterval      = 10000; // Request ML SOH inference every 10 seconds

unsigned long lastThingSpeakUpdate  = 0;
const unsigned long thingSpeakInterval = 20000; // ThingSpeak upload every 20 seconds (>=15s limit)

// Measured and Inferred States
float busvoltage      = 0.0; // Pack Voltage (V)
float current_mA      = 0.0; // Pack Current (mA)
float power_mW        = 0.0; // Pack Power (mW)
float temp_c          = 0.0; // DHT11 Temperature (°C)
float hum_pct         = 0.0; // DHT11 Humidity (%)

float estimatedSOC    = 0.0; // Voltage-mapped State of Charge (%)
float predictedSOH    = -1.0;// ML-predicted SOH (-1.0 indicates API unavailable / uninitialized)
int   cycleCounter    = 1;   // Operational cycle counter
int   safetyStatus    = 0;   // 0=Normal, 1=OverTemp, 2=LowVolt, 3=OverCurrent

bool  toggleDisplay   = false;
unsigned long lastDisplayToggle = 0;

// ======================== FUNCTION DECLARATIONS ========================
void readSensors();
void calculateSOC();
void evaluateLocalSafety();
void updateLCD();
void requestMLSOHPrediction();
void uploadThingSpeakTelemetry();

// ======================== SETUP ========================
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n[SYSTEM] ESP32 Battery Health & ML SOH Monitor Starting...");

  // Initialize Alert Pins
  pinMode(buzPin, OUTPUT);
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  digitalWrite(buzPin, LOW);
  digitalWrite(redPin, LOW);
  digitalWrite(greenPin, HIGH);

  // Initialize 16x2 LCD
  lcd.begin(16, 2);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("BATTERY MONITOR");
  lcd.setCursor(0, 1);
  lcd.print(" ML SOH SYSTEM ");
  delay(1500);

  // Initialize I2C and Sensors
  Wire.begin(21, 22); // SDA = GPIO 21, SCL = GPIO 22
  dht.begin();

  if (!ina219.begin()) {
    Serial.println("[ERROR] INA219 current/voltage sensor not detected!");
    lcd.clear();
    lcd.print("INA219 FAIL!");
    digitalWrite(redPin, HIGH);
    digitalWrite(greenPin, LOW);
    // Allow system to continue with degraded sensor state
  } else {
    Serial.println("[OK] INA219 initialized successfully.");
  }

  // Connect to Local Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  lcd.clear();
  lcd.print("Connecting WiFi");

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 15) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[OK] WiFi Connected! IP: " + WiFi.localIP().toString());
    lcd.clear();
    lcd.print("WiFi Connected!");
    lcd.setCursor(0, 1);
    lcd.print(WiFi.localIP().toString());
    delay(1000);
  } else {
    Serial.println("\n[WARN] WiFi connection timed out. Proceeding in offline mode.");
    lcd.clear();
    lcd.print("WiFi: Offline");
    lcd.setCursor(0, 1);
    lcd.print("Sensors Active");
    delay(1000);
  }

  // Initialize ThingSpeak
  ThingSpeak.begin(client);
  lcd.clear();
}

// ======================== MAIN LOOP ========================
void loop() {
  unsigned long now = millis();

  // 1. HIGH-FREQUENCY SENSOR READING & SAFETY CHECK (Every 500ms)
  if (now - lastSensorLoopTime >= sensorInterval) {
    lastSensorLoopTime = now;
    readSensors();
    calculateSOC();
    evaluateLocalSafety();
    updateLCD();
  }

  // 2. PERIODIC ML SOH INFERENCE (Every 10 seconds)
  if (now - lastMLInferenceTime >= mlInterval) {
    lastMLInferenceTime = now;
    if (WiFi.status() == WL_CONNECTED) {
      requestMLSOHPrediction();
    } else {
      predictedSOH = -1.0; // Clear SOH to prevent displaying stale/fake numbers
    }
  }

  // 3. PERIODIC CLOUD LOGGING TO THINGSPEAK (Every 20 seconds)
  if (now - lastThingSpeakUpdate >= thingSpeakInterval) {
    lastThingSpeakUpdate = now;
    if (WiFi.status() == WL_CONNECTED) {
      uploadThingSpeakTelemetry();
    }
  }
}

// ======================== SENSOR READING ========================
void readSensors() {
  // Read INA219 Bus Voltage, Shunt Current, and Power
  busvoltage = ina219.getBusVoltage_V();
  current_mA = ina219.getCurrent_mA();
  power_mW   = ina219.getPower_mW();

  // Filter out noise floor near zero
  if (busvoltage < 0.2) busvoltage = 0.0;
  if (abs(current_mA) < 2.0) current_mA = 0.0;
  if (power_mW < 2.0)   power_mW = 0.0;

  // Read DHT11 Temperature and Humidity
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  if (!isnan(t)) temp_c = t;
  if (!isnan(h)) hum_pct = h;
}

// ======================== SOC ESTIMATION ========================
void calculateSOC() {
  // Linear voltage-to-charge mapping based on 2S battery pack limits
  // NOTE: This represents estimated State of Charge (remaining capacity in current cycle),
  // NOT State of Health (SOH degradation across lifecycle).
  if (busvoltage <= MIN_BATTERY_VOLTAGE) {
    estimatedSOC = 0.0;
  } else if (busvoltage >= MAX_BATTERY_VOLTAGE) {
    estimatedSOC = 100.0;
  } else {
    estimatedSOC = ((busvoltage - MIN_BATTERY_VOLTAGE) / (MAX_BATTERY_VOLTAGE - MIN_BATTERY_VOLTAGE)) * 100.0;
  }
}

// ======================== LOCAL HARDWARE SAFETY ========================
void evaluateLocalSafety() {
  // Safety checks operate locally, deterministically, and independently of any network/ML response
  if (temp_c > ALERT_OVERTEMP_C) {
    safetyStatus = 1; // High Temperature Alert
    digitalWrite(redPin, HIGH);
    digitalWrite(greenPin, LOW);
    digitalWrite(buzPin, HIGH);
    delay(100);
    digitalWrite(buzPin, LOW);
  } else if (busvoltage > 1.0 && busvoltage < ALERT_LOW_VOLT_V) {
    safetyStatus = 2; // Critical Low Voltage Warning
    digitalWrite(redPin, HIGH);
    digitalWrite(greenPin, LOW);
    digitalWrite(buzPin, HIGH);
    delay(100);
    digitalWrite(buzPin, LOW);
  } else if (abs(current_mA) > ALERT_OVERCURRENT_MA) {
    safetyStatus = 3; // Overcurrent / Overload Alert
    digitalWrite(redPin, HIGH);
    digitalWrite(greenPin, LOW);
    digitalWrite(buzPin, HIGH);
    delay(100);
    digitalWrite(buzPin, LOW);
  } else {
    safetyStatus = 0; // Normal Safe State
    digitalWrite(redPin, LOW);
    digitalWrite(greenPin, HIGH);
    digitalWrite(buzPin, LOW);
  }
}

// ======================== LCD DISPLAY UPDATE ========================
void updateLCD() {
  // Toggle secondary display metrics every 2.5 seconds
  if (millis() - lastDisplayToggle > 2500) {
    toggleDisplay = !toggleDisplay;
    lastDisplayToggle = millis();
  }

  lcd.setCursor(0, 0);
  // Line 0: Voltage & Current
  char line0[17];
  snprintf(line0, sizeof(line0), "V:%4.1fV I:%4dmA", busvoltage, (int)current_mA);
  lcd.print(line0);

  lcd.setCursor(0, 1);
  char line1[17];
  if (safetyStatus == 1) {
    lcd.print("! OVERTEMP ALERT");
  } else if (safetyStatus == 2) {
    lcd.print("! LOW VOLT ALERT");
  } else if (safetyStatus == 3) {
    lcd.print("! OVERLOAD ALERT");
  } else {
    // Normal operation: alternate between SOC and ML-Predicted SOH
    if (!toggleDisplay) {
      snprintf(line1, sizeof(line1), "T:%2dC  SOC:%3d%%", (int)temp_c, (int)estimatedSOC);
      lcd.print(line1);
    } else {
      if (predictedSOH >= 0.0) {
        snprintf(line1, sizeof(line1), "T:%2dC  SOH:%3d%%", (int)temp_c, (int)predictedSOH);
      } else {
        snprintf(line1, sizeof(line1), "T:%2dC  SOH:OFFL", (int)temp_c);
      }
      lcd.print(line1);
    }
  }
}

// ======================== ML SOH INFERENCE OVER WI-FI ========================
void requestMLSOHPrediction() {
  HTTPClient http;
  http.begin(ML_SERVER_URL);
  http.setTimeout(ML_HTTP_TIMEOUT_MS);
  http.addHeader("Content-Type", "application/json");

  // Construct JSON telemetry payload conforming to inference API schema
  String jsonPayload = "{";
  jsonPayload += "\"pack_voltage\":" + String(busvoltage, 2) + ",";
  jsonPayload += "\"current_mA\":" + String(current_mA, 1) + ",";
  jsonPayload += "\"temperature_c\":" + String(temp_c, 1) + ",";
  jsonPayload += "\"cycle_number\":" + String(cycleCounter);
  jsonPayload += "}";

  Serial.println("[ML-API] Querying SOH: " + jsonPayload);
  int httpCode = http.POST(jsonPayload);

  if (httpCode == HTTP_CODE_OK) {
    String response = http.getString();
    Serial.println("[ML-API] Response: " + response);

    // Extract predicted_soh_percent from JSON response
    int idx = response.indexOf("\"predicted_soh_percent\":");
    if (idx != -1) {
      int startIdx = idx + 24;
      int endIdx = response.indexOf(",", startIdx);
      if (endIdx == -1) endIdx = response.indexOf("}", startIdx);
      if (endIdx > startIdx) {
        String sohStr = response.substring(startIdx, endIdx);
        sohStr.trim();
        predictedSOH = sohStr.toFloat();
        Serial.println("[ML-API] Updated ML Predicted SOH: " + String(predictedSOH, 1) + "%");
      }
    }
  } else {
    Serial.println("[ML-API] Warning: Inference request failed. HTTP Code: " + String(httpCode));
    // Keep predictedSOH as -1.0 or previous state without fabricating fake data
  }
  http.end();
}

// ======================== THINGSPEAK INTEGRATION ========================
void uploadThingSpeakTelemetry() {
  Serial.println("[THINGSPEAK] Uploading channel telemetry...");

  // Channel 443288 Field Mapping:
  // Field 1: Pack Voltage (V)
  // Field 2: Pack Current (mA)
  // Field 3: Pack Power (mW)
  // Field 4: Temperature (°C)
  // Field 5: ML Predicted SOH (%) [Updated only when valid prediction exists]
  // Field 6: Voltage-estimated SOC (%)
  // Field 7: Safety Status Code (0=Normal, 1=OverTemp, 2=LowVolt, 3=OverLoad)
  ThingSpeak.setField(1, busvoltage);
  ThingSpeak.setField(2, current_mA);
  ThingSpeak.setField(3, power_mW);
  ThingSpeak.setField(4, temp_c);
  if (predictedSOH >= 0.0) {
    ThingSpeak.setField(5, predictedSOH);
  }
  ThingSpeak.setField(6, estimatedSOC);
  ThingSpeak.setField(7, safetyStatus);

  int response = ThingSpeak.writeFields(myChannelNumber, myWriteAPIKey);
  if (response == 200) {
    Serial.println("[THINGSPEAK] Success! Channel updated.");
  } else {
    Serial.println("[THINGSPEAK] Error. Response code: " + String(response));
  }
}