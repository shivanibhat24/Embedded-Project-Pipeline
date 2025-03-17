#!/usr/bin/env python3
"""
Smart Aquaponics System with Edge AI
------------------------------------
This program monitors and maintains optimal conditions in an aquaponics system
using a Raspberry Pi with sensors and actuators. It includes edge AI for predictive
maintenance and automated system adjustments.
"""

import time
import datetime
import numpy as np
import pandas as pd
import threading
import json
import os
import logging
from typing import Dict, List, Tuple, Optional

# For sensor interfaces
import RPi.GPIO as GPIO
import Adafruit_ADS1x15  # For analog sensors
import board
import busio
import adafruit_bme280  # Temperature/humidity
from atlas_i2c import atlas_i2c  # For Atlas Scientific sensors (pH, EC)

# For edge AI
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aquaponics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("aquaponics")

# System configuration
CONFIG = {
    "sampling_interval": 60,  # seconds between readings
    "data_log_interval": 300,  # seconds between data logging
    "pH": {
        "target": 6.8,
        "tolerance": 0.3,
        "min": 6.0,
        "max": 7.5,
        "adjustment_interval": 900  # 15 minutes between pH adjustments
    },
    "ec": {  # Electrical conductivity (nutrient level)
        "target": 1400,  # μS/cm
        "tolerance": 100,
        "min": 800,
        "max": 1800,
        "adjustment_interval": 1800  # 30 minutes between nutrient adjustments
    },
    "temperature": {
        "target": 23.0,  # °C
        "tolerance": 2.0,
        "min": 18.0,
        "max": 28.0
    },
    "dissolved_oxygen": {
        "target": 7.0,  # mg/L
        "tolerance": 1.0,
        "min": 5.0,
        "max": 8.0
    },
    "water_level": {
        "min_percent": 80  # Minimum water level percentage
    },
    "light": {
        "on_hour": 6,  # 6 AM
        "off_hour": 20,  # 8 PM
        "intensity": 75  # Default light intensity percentage
    },
    "pins": {
        "ph_up_pump": 17,
        "ph_down_pump": 18,
        "nutrient_pump": 27,
        "water_pump": 22,
        "aerator": 23,
        "water_valve": 24,
        "heater": 25,
        "light_relay": 12,
        "alarm": 16,
        "water_level_sensor": 26
    },
    "model_path": "models/aquaponics_model.h5",
    "data_path": "data/aquaponics_data.csv"
}

class SensorModule:
    """Manages all sensor interactions and data collection"""
    
    def __init__(self):
        """Initialize sensor connections"""
        # Setup I2C
        self.i2c = busio.I2C(board.SCL, board.SDA)
        
        # ADS1115 ADC for analog sensors
        self.adc = Adafruit_ADS1x15.ADS1115()
        self.GAIN = 1  # +/- 4.096V
        
        # BME280 for air temperature and humidity
        self.bme280 = adafruit_bme280.Adafruit_BME280_I2C(self.i2c)
        
        # Atlas Scientific sensors
        self.ph_sensor = atlas_i2c(99)  # pH sensor at I2C address 99
        self.ec_sensor = atlas_i2c(100)  # EC sensor at I2C address 100
        self.do_sensor = atlas_i2c(97)   # Dissolved oxygen sensor at I2C address 97
        
        # Setup GPIO for water level sensor
        GPIO.setup(CONFIG["pins"]["water_level_sensor"], GPIO.IN)
        
        logger.info("Sensor module initialized")
    
    def read_ph(self) -> float:
        """Read pH value from Atlas pH sensor"""
        try:
            self.ph_sensor.write("R")
            time.sleep(0.9)  # Give time for the sensor to respond
            response = self.ph_sensor.read()
            return float(response)
        except Exception as e:
            logger.error(f"Error reading pH: {e}")
            return None
    
    def read_ec(self) -> float:
        """Read electrical conductivity (nutrient level) from Atlas EC sensor"""
        try:
            self.ec_sensor.write("R")
            time.sleep(0.9)
            response = self.ec_sensor.read()
            return float(response)
        except Exception as e:
            logger.error(f"Error reading EC: {e}")
            return None
    
    def read_water_temp(self) -> float:
        """Read water temperature from DS18B20 sensor via OneWire"""
        try:
            # This is a simplified implementation - in reality, you'd use the 
            # OneWire protocol to read from a DS18B20 sensor
            # For simulation, we'll read from analog
            raw = self.adc.read_adc(0, gain=self.GAIN)
            # Convert to temperature (adjust calibration as needed)
            temp = (raw * 0.0625) / 1000
            return 20 + 10 * temp  # Simulate a temperature between 20-30°C
        except Exception as e:
            logger.error(f"Error reading water temperature: {e}")
            return None
    
    def read_dissolved_oxygen(self) -> float:
        """Read dissolved oxygen from Atlas DO sensor"""
        try:
            self.do_sensor.write("R")
            time.sleep(0.9)
            response = self.do_sensor.read()
            return float(response)
        except Exception as e:
            logger.error(f"Error reading dissolved oxygen: {e}")
            return None
    
    def read_water_level(self) -> int:
        """Read water level as a percentage"""
        try:
            # In a real system, you might use multiple water level sensors
            # or an ultrasonic distance sensor
            # This is simplified for demonstration
            level_sensor = GPIO.input(CONFIG["pins"]["water_level_sensor"])
            return 100 if level_sensor else 75
        except Exception as e:
            logger.error(f"Error reading water level: {e}")
            return None
    
    def read_light(self) -> int:
        """Read light intensity as a percentage"""
        try:
            raw = self.adc.read_adc(1, gain=self.GAIN)
            # Convert to percentage
            light_percent = min(100, max(0, int(raw / 32767 * 100)))
            return light_percent
        except Exception as e:
            logger.error(f"Error reading light level: {e}")
            return None
    
    def read_air_temp_humidity(self) -> Tuple[float, float]:
        """Read air temperature and humidity from BME280"""
        try:
            return (self.bme280.temperature, self.bme280.humidity)
        except Exception as e:
            logger.error(f"Error reading air temperature/humidity: {e}")
            return (None, None)
    
    def get_all_readings(self) -> Dict:
        """Get all sensor readings as a dictionary"""
        air_temp, humidity = self.read_air_temp_humidity()
        
        readings = {
            "timestamp": datetime.datetime.now().isoformat(),
            "ph": self.read_ph(),
            "ec": self.read_ec(),
            "water_temp": self.read_water_temp(),
            "dissolved_oxygen": self.read_dissolved_oxygen(),
            "water_level": self.read_water_level(),
            "light": self.read_light(),
            "air_temp": air_temp,
            "humidity": humidity
        }
        
        return readings


class ActuatorModule:
    """Controls all actuators based on sensor data and AI recommendations"""
    
    def __init__(self):
        """Initialize actuator connections"""
        # Set GPIO mode
        GPIO.setmode(GPIO.BCM)
        
        # Configure GPIO pins for outputs
        for pin_name, pin in CONFIG["pins"].items():
            if pin_name != "water_level_sensor":  # Skip input pins
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)  # Initialize to OFF
        
        # PWM for light control (if needed)
        self.light_pwm = GPIO.PWM(CONFIG["pins"]["light_relay"], 100)
        self.light_pwm.start(0)
        
        # Keep track of last adjustment times
        self.last_adjustment = {
            "ph_up": 0,
            "ph_down": 0,
            "nutrient": 0,
            "water": 0,
        }
        
        logger.info("Actuator module initialized")
    
    def adjust_ph(self, current_ph: float) -> None:
        """Adjust pH level by adding pH up or down solution"""
        now = time.time()
        target = CONFIG["pH"]["target"]
        tolerance = CONFIG["pH"]["tolerance"]
        
        # Only adjust if outside tolerance and enough time has passed
        if current_ph is not None:
            if current_ph < target - tolerance and now - self.last_adjustment["ph_up"] > CONFIG["pH"]["adjustment_interval"]:
                logger.info(f"pH too low ({current_ph}), adding pH up solution")
                GPIO.output(CONFIG["pins"]["ph_up_pump"], GPIO.HIGH)
                time.sleep(1)  # Run pump for 1 second
                GPIO.output(CONFIG["pins"]["ph_up_pump"], GPIO.LOW)
                self.last_adjustment["ph_up"] = now
                
            elif current_ph > target + tolerance and now - self.last_adjustment["ph_down"] > CONFIG["pH"]["adjustment_interval"]:
                logger.info(f"pH too high ({current_ph}), adding pH down solution")
                GPIO.output(CONFIG["pins"]["ph_down_pump"], GPIO.HIGH)
                time.sleep(1)  # Run pump for 1 second
                GPIO.output(CONFIG["pins"]["ph_down_pump"], GPIO.LOW)
                self.last_adjustment["ph_down"] = now
    
    def adjust_nutrients(self, current_ec: float) -> None:
        """Adjust nutrient level by adding nutrient solution"""
        now = time.time()
        target = CONFIG["ec"]["target"]
        tolerance = CONFIG["ec"]["tolerance"]
        
        # Only adjust if outside tolerance and enough time has passed
        if current_ec is not None and current_ec < target - tolerance and now - self.last_adjustment["nutrient"] > CONFIG["ec"]["adjustment_interval"]:
            logger.info(f"EC too low ({current_ec}), adding nutrients")
            GPIO.output(CONFIG["pins"]["nutrient_pump"], GPIO.HIGH)
            time.sleep(2)  # Run pump for 2 seconds
            GPIO.output(CONFIG["pins"]["nutrient_pump"], GPIO.LOW)
            self.last_adjustment["nutrient"] = now
    
    def control_water_level(self, current_level: int) -> None:
        """Maintain water level by adding water if needed"""
        now = time.time()
        min_level = CONFIG["water_level"]["min_percent"]
        
        if current_level is not None and current_level < min_level and now - self.last_adjustment["water"] > 3600:  # Max once per hour
            logger.info(f"Water level too low ({current_level}%), adding water")
            GPIO.output(CONFIG["pins"]["water_valve"], GPIO.HIGH)
            time.sleep(10)  # Open valve for 10 seconds
            GPIO.output(CONFIG["pins"]["water_valve"], GPIO.LOW)
            self.last_adjustment["water"] = now
    
    def control_aeration(self, do_level: float) -> None:
        """Control aerator based on dissolved oxygen levels"""
        if do_level is not None:
            if do_level < CONFIG["dissolved_oxygen"]["min"]:
                GPIO.output(CONFIG["pins"]["aerator"], GPIO.HIGH)
            elif do_level > CONFIG["dissolved_oxygen"]["max"]:
                GPIO.output(CONFIG["pins"]["aerator"], GPIO.LOW)
    
    def control_temperature(self, water_temp: float) -> None:
        """Control heater based on water temperature"""
        if water_temp is not None:
            if water_temp < CONFIG["temperature"]["min"]:
                GPIO.output(CONFIG["pins"]["heater"], GPIO.HIGH)
            elif water_temp > CONFIG["temperature"]["max"]:
                GPIO.output(CONFIG["pins"]["heater"], GPIO.LOW)
    
    def control_lighting(self) -> None:
        """Control lighting based on time of day"""
        current_hour = datetime.datetime.now().hour
        on_hour = CONFIG["light"]["on_hour"]
        off_hour = CONFIG["light"]["off_hour"]
        
        if on_hour <= current_hour < off_hour:
            intensity = CONFIG["light"]["intensity"]
            self.light_pwm.ChangeDutyCycle(intensity)
        else:
            self.light_pwm.ChangeDutyCycle(0)
    
    def trigger_alarm(self, activate: bool) -> None:
        """Activate or deactivate alarm"""
        GPIO.output(CONFIG["pins"]["alarm"], GPIO.HIGH if activate else GPIO.LOW)
    
    def cleanup(self) -> None:
        """Clean up GPIO on program exit"""
        self.light_pwm.stop()
        GPIO.cleanup()


class EdgeAI:
    """Manages edge AI for prediction and optimization"""
    
    def __init__(self, datastore):
        """Initialize Edge AI system"""
        self.datastore = datastore
        self.model = None
        self.scaler = MinMaxScaler()
        self.load_model()
        logger.info("Edge AI module initialized")
    
    def load_model(self) -> None:
        """Load TensorFlow model for predictions"""
        try:
            if os.path.exists(CONFIG["model_path"]):
                self.model = tf.keras.models.load_model(CONFIG["model_path"])
                logger.info("AI model loaded successfully")
            else:
                logger.warning("AI model not found, will train new model after collecting data")
        except Exception as e:
            logger.error(f"Error loading AI model: {e}")
    
    def train_model(self) -> None:
        """Train a new model based on collected data"""
        try:
            # This is a simplified training process
            if len(self.datastore.data) < 100:
                logger.info("Not enough data to train model yet")
                return
            
            # Prepare data
            df = pd.DataFrame(self.datastore.data)
            
            # Feature engineering
            features = df[['ph', 'ec', 'water_temp', 'dissolved_oxygen', 'water_level']].copy()
            # Handle missing values
            features = features.fillna(method='ffill')
            
            # Target will be next hour's pH and EC
            targets = pd.DataFrame({
                'next_ph': features['ph'].shift(-6),  # 6 points = 1 hour if sampling every 10 min
                'next_ec': features['ec'].shift(-6)
            }).dropna()
            
            # Trim data to match
            features = features.iloc[:len(targets)]
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Create model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(features.shape[1],)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(2)  # Predict pH and EC
            ])
            
            model.compile(optimizer='adam', loss='mse')
            
            # Train model
            model.fit(
                features_scaled, 
                targets.values,
                epochs=50,
                batch_size=16,
                validation_split=0.2,
                verbose=0
            )
            
            # Save model
            model.save(CONFIG["model_path"])
            self.model = model
            logger.info("AI model trained and saved successfully")
            
        except Exception as e:
            logger.error(f"Error training AI model: {e}")
    
    def predict_next_values(self, current_readings: Dict) -> Dict:
        """Predict next pH and EC values based on current readings"""
        if self.model is None:
            return {}
        
        try:
            # Extract features
            features = np.array([[
                current_readings.get('ph', 7.0),
                current_readings.get('ec', 1400),
                current_readings.get('water_temp', 23.0),
                current_readings.get('dissolved_oxygen', 7.0),
                current_readings.get('water_level', 90)
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            predictions = self.model.predict(features_scaled)
            
            return {
                'predicted_ph': predictions[0][0],
                'predicted_ec': predictions[0][1]
            }
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {}
    
    def detect_anomalies(self, readings: Dict) -> List[str]:
        """Detect anomalies in sensor readings"""
        anomalies = []
        
        # Check each reading against expected ranges
        if readings.get('ph') is not None:
            if readings['ph'] < CONFIG['pH']['min'] or readings['ph'] > CONFIG['pH']['max']:
                anomalies.append(f"pH anomaly: {readings['ph']}")
        
        if readings.get('ec') is not None:
            if readings['ec'] < CONFIG['ec']['min'] or readings['ec'] > CONFIG['ec']['max']:
                anomalies.append(f"EC anomaly: {readings['ec']}")
        
        if readings.get('water_temp') is not None:
            if readings['water_temp'] < CONFIG['temperature']['min'] or readings['water_temp'] > CONFIG['temperature']['max']:
                anomalies.append(f"Temperature anomaly: {readings['water_temp']}")
        
        if readings.get('dissolved_oxygen') is not None:
            if readings['dissolved_oxygen'] < CONFIG['dissolved_oxygen']['min'] or readings['dissolved_oxygen'] > CONFIG['dissolved_oxygen']['max']:
                anomalies.append(f"Dissolved oxygen anomaly: {readings['dissolved_oxygen']}")
        
        return anomalies
    
    def optimize_parameters(self) -> Dict:
        """Optimize system parameters based on historical data"""
        # This would be more complex in a real system
        # For now, we'll just return the current configuration
        return CONFIG


class DataStore:
    """Manages data storage and retrieval"""
    
    def __init__(self):
        """Initialize data storage"""
        self.data = []
        self.load_data()
        logger.info("Data storage module initialized")
    
    def load_data(self) -> None:
        """Load historical data from CSV"""
        try:
            if os.path.exists(CONFIG["data_path"]):
                df = pd.read_csv(CONFIG["data_path"])
                self.data = df.to_dict('records')
                logger.info(f"Loaded {len(self.data)} historical data points")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def save_data(self) -> None:
        """Save data to CSV"""
        try:
            df = pd.DataFrame(self.data)
            os.makedirs(os.path.dirname(CONFIG["data_path"]), exist_ok=True)
            df.to_csv(CONFIG["data_path"], index=False)
            logger.info(f"Saved {len(self.data)} data points")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def add_reading(self, reading: Dict) -> None:
        """Add a new sensor reading to the data store"""
        self.data.append(reading)
        
        # Save data periodically
        if len(self.data) % 50 == 0:
            self.save_data()
    
    def get_recent_readings(self, count: int = 10) -> List[Dict]:
        """Get the most recent readings"""
        return self.data[-count:] if len(self.data) >= count else self.data


class AquaponicsSystem:
    """Main system controller"""
    
    def __init__(self):
        """Initialize the aquaponics system"""
        logger.info("Initializing Smart Aquaponics System")
        
        # Initialize modules
        self.datastore = DataStore()
        self.sensors = SensorModule()
        self.actuators = ActuatorModule()
        self.ai = EdgeAI(self.datastore)
        
        # State variables
        self.running = False
        self.alarm_active = False
        
        logger.info("System initialization complete")
    
    def start(self) -> None:
        """Start the aquaponics system"""
        self.running = True
        logger.info("Starting Smart Aquaponics System")
        
        # Start main control loop in a separate thread
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        # Start web server for monitoring (optional)
        # self.start_web_server()
    
    def stop(self) -> None:
        """Stop the aquaponics system"""
        logger.info("Stopping Smart Aquaponics System")
        self.running = False
        self.control_thread.join(timeout=5)
        self.actuators.cleanup()
        self.datastore.save_data()
    
    def control_loop(self) -> None:
        """Main control loop for the system"""
        next_ai_train = time.time() + 86400  # Train AI once per day
        
        while self.running:
            try:
                # Read all sensors
                readings = self.sensors.get_all_readings()
                
                # Add timestamp
                readings["timestamp"] = datetime.datetime.now().isoformat()
                
                # Store data
                self.datastore.add_reading(readings)
                
                # Check for anomalies
                anomalies = self.ai.detect_anomalies(readings)
                if anomalies:
                    logger.warning(f"Detected anomalies: {anomalies}")
                    self.actuators.trigger_alarm(True)
                    self.alarm_active = True
                elif self.alarm_active:
                    self.actuators.trigger_alarm(False)
                    self.alarm_active = False
                
                # Make predictions
                predictions = self.ai.predict_next_values(readings)
                
                # Take actions based on current readings and predictions
                self.actuators.adjust_ph(readings.get("ph"))
                self.actuators.adjust_nutrients(readings.get("ec"))
                self.actuators.control_water_level(readings.get("water_level"))
                self.actuators.control_aeration(readings.get("dissolved_oxygen"))
                self.actuators.control_temperature(readings.get("water_temp"))
                self.actuators.control_lighting()
                
                # Log current state
                logger.info(f"Current: pH={readings.get('ph')}, EC={readings.get('ec')}, "
                           f"DO={readings.get('dissolved_oxygen')}, Temp={readings.get('water_temp')}")
                if predictions:
                    logger.info(f"Predicted: pH={predictions.get('predicted_ph')}, "
                               f"EC={predictions.get('predicted_ec')}")
                
                # Train AI model periodically
                if time.time() > next_ai_train:
                    self.ai.train_model()
                    next_ai_train = time.time() + 86400  # Once per day
                
                # Wait for next reading cycle
                time.sleep(CONFIG["sampling_interval"])
                
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                time.sleep(10)  # Wait before retrying
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        try:
            readings = self.sensors.get_all_readings()
            recent_data = self.datastore.get_recent_readings(24)  # Last 24 readings
            
            # Calculate averages
            if recent_data:
                df = pd.DataFrame(recent_data)
                averages = {
                    'avg_ph': df['ph'].mean(),
                    'avg_ec': df['ec'].mean(),
                    'avg_water_temp': df['water_temp'].mean(),
                    'avg_do': df['dissolved_oxygen'].mean()
                }
            else:
                averages = {}
            
            return {
                'current': readings,
                'averages': averages,
                'alarm_active': self.alarm_active,
                'predictions': self.ai.predict_next_values(readings)
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}


# Main execution
if __name__ == "__main__":
    try:
        # Create and start the system
        system = AquaponicsSystem()
        system.start()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
    finally:
        # Clean shutdown
        if 'system' in locals():
            system.stop()
        logger.info("System shutdown complete")
