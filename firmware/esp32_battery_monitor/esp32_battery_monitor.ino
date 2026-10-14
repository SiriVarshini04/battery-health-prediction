#include <Wire.h>
#include <WiFi.h>
#include <Adafruit_INA219.h>
#include <LiquidCrystal.h>
#include <ThingSpeak.h>
#include "DHT.h"

// LCD Configuration (4-bit mode)
const int rs = 15, en = 2, d4 = 4, d5 = 16, d6 = 17, d7 = 5;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

// INA219 Current & Voltage Sensor
Adafruit_INA219 ina219;

// DHT11 Sensor
#define DHTPIN 18
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// Output Pin Definitions
const int buzPin   = 23;
const int redPin   = 32;
const int greenPin = 33;

// Battery Specification Constants for SOC & SOH Calculations (Adjust for your battery type)
const float MIN_BATTERY_VOLTAGE = 4.0;  // 0% SOC Voltage (e.g., 3S Li-ion or 12V Lead Acid low limit)
const float MAX_BATTERY_VOLTAGE = 7.6; // 100% SOC Voltage

// Battery Health Tracking Variables
float initialCapacityAh = 2.2;          // Nominal battery capacity in Ah
float currentCapacityAh = 2.2;          // Estimated remaining battery capacity in Ah
unsigned long lastTimeMs = 0;

// Wi-Fi & ThingSpeak Credentials
const char* WIFI_SSID = "project";
const char* WIFI_PASS = "12345678";

WiFiClient client;
unsigned long myChannelNumber = 443288;          // Replace with your Channel ID
const char * myWriteAPIKey = "6BB9V3OTG8797LD9"; // Replace with your Write API Key

unsigned long lastThingSpeakUpdate = 0;
const unsigned long thingSpeakInterval = 15000; // Send data every 15 seconds

void setup() {
  Serial.begin(115200);

  pinMode(buzPin, OUTPUT);
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);

  digitalWrite(buzPin, LOW);
  digitalWrite(redPin, LOW);
  digitalWrite(greenPin, HIGH);

  // Initialize LCD
  lcd.begin(16, 2);
  lcd.print("   WELCOME    ");
  delay(2000);
  lcd.clear();

  // Initialize I2C Bus & DHT Sensor
  Wire.begin(21, 22); // SDA = GPIO 21, SCL = GPIO 22
  dht.begin();

  if (!ina219.begin()) {
    Serial.println("Failed to find INA219 chip");
    lcd.clear();
    lcd.print("INA219 Error!");
    while (1) { delay(10); }
  }

  // Connect to Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  lcd.clear();
  lcd.print("Connecting WiFi");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  lcd.clear();
  lcd.print("WiFi Connected!");
  delay(1000);

  // Initialize ThingSpeak Client
  ThingSpeak.begin(client);
  lastTimeMs = millis();
}

void loop() {
  // Read INA219 Sensor Data
  float busvoltage = ina219.getBusVoltage_V();
  float current_mA = ina219.getCurrent_mA();
  float power_mW   = ina219.getPower_mW();

  if (busvoltage < 1.1) busvoltage = 0;
  if (current_mA < 1.5) current_mA = 0;
  if (power_mW < 1.5)   power_mW = 0;

  // Read DHT11 Temperature Data
  float temp = dht.readTemperature();
  float hum  = dht.readHumidity();
  if (isnan(temp)) temp = 0;
  if (isnan(hum))  hum = 0;

  // 1. Calculate State of Charge (SOC) via Voltage Mapping
  float soc = ((busvoltage - MIN_BATTERY_VOLTAGE) / (MAX_BATTERY_VOLTAGE - MIN_BATTERY_VOLTAGE)) * 100.0;
  if (soc > 100.0) soc = 100.0;
  if (soc < 0.0)   soc = 0.0;
  unsigned long currentTimeMs = millis();
  float deltaTimeHours = (currentTimeMs - lastTimeMs) / 3600000.0;
  lastTimeMs = currentTimeMs;
  if (temp > 45.0) {
    currentCapacityAh -= 0.00001; // Minor thermal degradation factor
  }

  float soh = (currentCapacityAh / initialCapacityAh) * 100.0;
  if (soh > 100.0) soh = 100.0;
  if (soh < 0.0)   soh = 0.0;
  lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("V:" + String(busvoltage, 1) + "V I:" + String((int)current_mA) +"mA");
    lcd.setCursor(0, 1);
    lcd.print("T:" + String((int)temp)+" SOH:"+ String((int)soh));
    delay(500);

  // Safety Controls & Threshold Checks
  if (temp > 37.0) {
    // High Temperature Alert
    digitalWrite(redPin, HIGH);
    digitalWrite(greenPin, LOW);
    digitalWrite(buzPin, HIGH);
    lcd.setCursor(10, 0);
    lcd.print("OVERTEMP");
    delay(200);
     digitalWrite(buzPin,0);
  } 
  else if (busvoltage < 4) {
    // Low Voltage Warning
    digitalWrite(redPin, HIGH);
    digitalWrite(greenPin, LOW);
    digitalWrite(buzPin, HIGH);
    lcd.setCursor(10, 0);
    lcd.print("LOW VOLT");
    delay(200);
     digitalWrite(buzPin,0);
  }  
  else if (current_mA>50) {
    // Low Voltage Warning
    digitalWrite(redPin, HIGH);
    digitalWrite(greenPin, LOW);
    digitalWrite(buzPin, HIGH);
    lcd.setCursor(10, 0);
    lcd.print("ov lod");
    delay(200);
     digitalWrite(buzPin,0);
  } 
  else {
    // Normal Operation State
    digitalWrite(redPin, LOW);
    digitalWrite(greenPin, HIGH);
    digitalWrite(buzPin, LOW);
  }

  // Post Telemetry Data to ThingSpeak
  if (millis() - lastThingSpeakUpdate > thingSpeakInterval) {
    ThingSpeak.setField(1, busvoltage);
    ThingSpeak.setField(2, current_mA);
    ThingSpeak.setField(3, power_mW);
    ThingSpeak.setField(4, temp);
    ThingSpeak.setField(5, soh);

    int response = ThingSpeak.writeFields(myChannelNumber, myWriteAPIKey);
    if (response == 200) {
      Serial.println("ThingSpeak Update Successful!");
    } else {
      Serial.println("ThingSpeak Error Code: " + String(response));
    }
    lastThingSpeakUpdate = millis();
  }

  delay(2000);
}