/*
 * ESP32 Agricrawler Integration Example
 * Hardware: ESP32 + ESP32-CAM + DHT22 + Capacitive Soil Moisture Sensor
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "DHT.h"

// Hardware Configuration
#define DHT_PIN 4
#define DHT_TYPE DHT22
#define SOIL_MOISTURE_PIN A0

// WiFi Configuration
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// API Configuration
const char* api_url = "http://YOUR_SERVER_IP:8000/analyze";

// Sensor Objects
DHT dht(DHT_PIN, DHT_TYPE);

// Soil moisture calibration values
const int dry_value = 4095;  // Value when soil is completely dry
const int wet_value = 0;     // Value when soil is saturated

void setup() {
  Serial.begin(115200);
  
  // Initialize sensors
  dht.begin();
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("WiFi connected");
  
  // Initialize camera
  camera_init();
}

void loop() {
  // Read sensor data
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  float soil_moisture = readSoilMoisture();
  
  // Check if readings are valid
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("Failed to read from DHT sensor");
    delay(2000);
    return;
  }
  
  // Capture image
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    delay(2000);
    return;
  }
  
  // Send data to API
  sendToAPI(temperature, humidity, soil_moisture, fb->buf, fb->len);
  
  // Release camera buffer
  esp_camera_fb_return(fb);
  
  // Wait before next reading
  delay(30000); // 30 seconds
}

float readSoilMoisture() {
  int raw_value = analogRead(SOIL_MOISTURE_PIN);
  
  // Convert to percentage (0-100%)
  float percentage = map(raw_value, dry_value, wet_value, 0, 100);
  percentage = constrain(percentage, 0, 100);
  
  return percentage;
}

void sendToAPI(float temp, float humidity, float soil, uint8_t* image_data, size_t image_len) {
  HTTPClient http;
  http.begin(api_url);
  
  // Create multipart form data
  String boundary = "----AgricrawlerBoundary";
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
  
  String body = "";
  
  // Add sensor data
  body += "--" + boundary + "\r\n";
  body += "Content-Disposition: form-data; name=\"soil\"\r\n\r\n";
  body += String(soil) + "\r\n";
  
  body += "--" + boundary + "\r\n";
  body += "Content-Disposition: form-data; name=\"temp\"\r\n\r\n";
  body += String(temp) + "\r\n";
  
  body += "--" + boundary + "\r\n";
  body += "Content-Disposition: form-data; name=\"humidity\"\r\n\r\n";
  body += String(humidity) + "\r\n";
  
  // Add image data
  body += "--" + boundary + "\r\n";
  body += "Content-Disposition: form-data; name=\"image\"; filename=\"crop_image.jpg\"\r\n";
  body += "Content-Type: image/jpeg\r\n\r\n";
  
  // Send headers and body
  http.POST((uint8_t*)body.c_str(), body.length());
  
  // Send image data
  WiFiClient* stream = http.getStreamPtr();
  stream->write(image_data, image_len);
  stream->print("\r\n--" + boundary + "--\r\n");
  
  // Get response
  int httpCode = http.GET();
  if (httpCode > 0) {
    String response = http.getString();
    Serial.println("API Response:");
    Serial.println(response);
    
    // Parse JSON response
    parseAPIResponse(response);
  } else {
    Serial.println("HTTP Error: " + String(httpCode));
  }
  
  http.end();
}

void parseAPIResponse(String response) {
  DynamicJsonDocument doc(2048);
  deserializeJson(doc, response);
  
  // Extract key information
  float fused_stress = doc["analysis"]["fused_stress"];
  String alert_level = doc["analysis"]["alert_level"];
  String recommendation = doc["analysis"]["recommendation"];
  
  Serial.println("Analysis Results:");
  Serial.println("Fused Stress: " + String(fused_stress, 3));
  Serial.println("Alert Level: " + alert_level);
  Serial.println("Recommendation: " + recommendation);
  
  // Take action based on alert level
  if (alert_level == "high") {
    // Implement high priority actions
    Serial.println("HIGH ALERT: Immediate attention required");
  } else if (alert_level == "medium") {
    // Implement medium priority actions
    Serial.println("MEDIUM ALERT: Monitor closely");
  } else {
    // Low alert - continue normal operation
    Serial.println("LOW ALERT: Normal operation");
  }
}

void camera_init() {
  // Camera configuration
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;
  config.pin_d1 = 18;
  config.pin_d2 = 19;
  config.pin_d3 = 21;
  config.pin_d4 = 36;
  config.pin_d5 = 39;
  config.pin_d6 = 34;
  config.pin_d7 = 35;
  config.pin_xclk = 0;
  config.pin_pclk = 22;
  config.pin_vsync = 25;
  config.pin_href = 23;
  config.pin_sscb_sda = 26;
  config.pin_sscb_scl = 27;
  config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // Image quality settings
  if (psramFound()) {
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }
  
  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
  
  Serial.println("Camera initialized successfully");
}
