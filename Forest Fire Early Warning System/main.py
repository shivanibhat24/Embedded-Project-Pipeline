#!/usr/bin/env python3
"""
Forest Fire Early Warning System for Raspberry Pi Zero
Detects abnormal temperature and humidity changes that might indicate forest fire conditions.

Requirements:
- Raspberry Pi Zero
- DHT22 temperature and humidity sensor
- Optional: LED indicators, buzzer for alerts
- Optional: LoRa module for remote communication in forest environments

Installation:
pip install adafruit-circuitpython-dht RPi.GPIO pandas numpy
"""

import time
import os
import json
import numpy as np
import pandas as pd
import board
import adafruit_dht
import RPi.GPIO as GPIO
from datetime import datetime, timedelta
from collections import deque

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Configure pins
DHT_PIN = board.D4        # DHT22 sensor connected to GPIO4
RED_LED_PIN = 17          # Red LED for high alert
YELLOW_LED_PIN = 27       # Yellow LED for medium alert
GREEN_LED_PIN = 22        # Green LED for normal conditions
BUZZER_PIN = 18           # Buzzer for audible alerts

# Setup output pins
for pin in [RED_LED_PIN, YELLOW_LED_PIN, GREEN_LED_PIN, BUZZER_PIN]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Initialize DHT sensor
dht = adafruit_dht.DHT22(DHT_PIN)

# System configuration
CONFIG = {
    "sampling_interval": 60,            # Seconds between readings
    "data_window_size": 24*60,          # Number of readings to keep (24 hours at 1 per minute)
    "temp_rise_threshold": 5.0,         # Celsius temperature rise in an hour to trigger warning
    "humidity_drop_threshold": 15.0,    # Percentage humidity drop in an hour to trigger warning
    "high_temp_threshold": 35.0,        # Celsius temperature to trigger high alert
    "low_humidity_threshold": 20.0,     # Percentage humidity to trigger high alert
    "log_file": "forest_fire_monitor.csv",
    "alert_log": "fire_alerts.json",
    "calibration_period": 24*60         # Number of readings for initial calibration (1 day)
}

class ForestFireMonitor:
    def __init__(self, config):
        self.config = config
        self.data = deque(maxlen=config["data_window_size"])
        self.df = None
        self.alert_state = "NORMAL"  # NORMAL, WARNING, DANGER
        self.alerts = []
        self.load_history()
        self.calibration_complete = False
        self.calibration_count = 0
        
    def load_history(self):
        """Load historical data if available"""
        if os.path.exists(self.config["log_file"]):
            try:
                df = pd.read_csv(self.config["log_file"])
                # Keep only the most recent records within our window size
                if len(df) > self.config["data_window_size"]:
                    df = df.tail(self.config["data_window_size"])
                    
                # Convert data to deque
                for _, row in df.iterrows():
                    self.data.append({
                        "timestamp": datetime.fromisoformat(row["timestamp"]),
                        "temperature": row["temperature"],
                        "humidity": row["humidity"]
                    })
                
                print(f"Loaded {len(self.data)} historical records")
                
                # If we have enough data, we can skip calibration
                if len(self.data) >= self.config["calibration_period"]:
                    self.calibration_complete = True
                    self.calibration_count = self.config["calibration_period"]
                    print("Calibration complete based on historical data")
                    
            except Exception as e:
                print(f"Error loading history: {e}")
                
        # Load previous alerts
        if os.path.exists(self.config["alert_log"]):
            try:
                with open(self.config["alert_log"], "r") as f:
                    self.alerts = json.load(f)
            except Exception as e:
                print(f"Error loading alerts: {e}")
                
    def save_data(self):
        """Save data to CSV file"""
        if not self.data:
            return
            
        df = pd.DataFrame([
            {
                "timestamp": d["timestamp"].isoformat(),
                "temperature": d["temperature"],
                "humidity": d["humidity"]
            } for d in self.data
        ])
        
        df.to_csv(self.config["log_file"], index=False)
        
        # Save alerts
        with open(self.config["alert_log"], "w") as f:
            json.dump(self.alerts, f, indent=2)
            
    def read_sensor(self):
        """Read temperature and humidity from DHT sensor with error handling"""
        max_retries = 3
        for _ in range(max_retries):
            try:
                temperature = dht.temperature
                humidity = dht.humidity
                if temperature is not None and humidity is not None:
                    return temperature, humidity
            except Exception as e:
                print(f"Sensor read error: {e}")
                time.sleep(2)
                
        # If we reach here, all attempts failed
        # Return previous values if available, otherwise defaults
        if self.data:
            last_reading = self.data[-1]
            return last_reading["temperature"], last_reading["humidity"]
        return 25.0, 50.0  # Default values
        
    def detect_anomalies(self):
        """Analyze recent data to detect conditions indicative of potential fire"""
        if len(self.data) < 60:  # Need at least 60 readings for meaningful analysis
            return "NORMAL"
            
        # Convert deque to DataFrame for easier analysis
        df = pd.DataFrame([{
            "timestamp": d["timestamp"],
            "temperature": d["temperature"],
            "humidity": d["humidity"]
        } for d in self.data])
        
        # Current values
        current_temp = df["temperature"].iloc[-1]
        current_humidity = df["humidity"].iloc[-1]
        
        # Calculate changes over the last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        hour_data = df[df["timestamp"] >= one_hour_ago]
        
        if len(hour_data) > 10:  # Need sufficient data points
            temp_change = current_temp - hour_data["temperature"].iloc[0]
            humidity_change = hour_data["humidity"].iloc[0] - current_humidity
            
            # Check for anomalies
            if (current_temp >= self.config["high_temp_threshold"] and
                current_humidity <= self.config["low_humidity_threshold"]):
                return "DANGER"
            elif (temp_change >= self.config["temp_rise_threshold"] or
                  humidity_change >= self.config["humidity_drop_threshold"]):
                return "WARNING"
                
        return "NORMAL"
        
    def update_indicators(self, state):
        """Update LEDs and buzzer based on alert state"""
        # Turn off all indicators first
        for pin in [RED_LED_PIN, YELLOW_LED_PIN, GREEN_LED_PIN]:
            GPIO.output(pin, GPIO.LOW)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        
        # Set appropriate indicators
        if state == "DANGER":
            GPIO.output(RED_LED_PIN, GPIO.HIGH)
            # Pulse buzzer
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
        elif state == "WARNING":
            GPIO.output(YELLOW_LED_PIN, GPIO.HIGH)
        else:  # NORMAL
            GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
            
    def log_alert(self, state, temp, humidity):
        """Log alerts when state changes"""
        if self.alert_state != state:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "state": state,
                "temperature": temp,
                "humidity": humidity
            }
            self.alerts.append(alert)
            print(f"ALERT: {state} - Temp: {temp}°C, Humidity: {humidity}%")
            
            # Save immediately on state change
            self.save_data()
            
        self.alert_state = state
        
    def run(self):
        """Main monitoring loop"""
        print("Starting Forest Fire Early Warning System...")
        print("System will calibrate during initial operation...")
        
        try:
            while True:
                try:
                    # Read sensor data
                    temp, humidity = self.read_sensor()
                    timestamp = datetime.now()
                    
                    # Add to data queue
                    self.data.append({
                        "timestamp": timestamp,
                        "temperature": temp,
                        "humidity": humidity
                    })
                    
                    # Display current readings
                    print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Temp: {temp:.1f}°C, Humidity: {humidity:.1f}%")
                    
                    # Update calibration status
                    if not self.calibration_complete:
                        self.calibration_count += 1
                        print(f"Calibrating: {self.calibration_count}/{self.config['calibration_period']}")
                        if self.calibration_count >= self.config["calibration_period"]:
                            self.calibration_complete = True
                            print("Calibration complete")
                    
                    # Only perform anomaly detection after calibration
                    if self.calibration_complete:
                        state = self.detect_anomalies()
                        self.update_indicators(state)
                        self.log_alert(state, temp, humidity)
                    else:
                        # During calibration, just show green light
                        self.update_indicators("NORMAL")
                    
                    # Save data periodically (every 10 minutes)
                    if timestamp.minute % 10 == 0 and timestamp.second < self.config["sampling_interval"]:
                        self.save_data()
                        
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    
                # Wait for next sampling interval
                time.sleep(self.config["sampling_interval"])
                
        except KeyboardInterrupt:
            print("System shutting down...")
            self.save_data()
            GPIO.cleanup()
            
if __name__ == "__main__":
    monitor = ForestFireMonitor(CONFIG)
    monitor.run()
