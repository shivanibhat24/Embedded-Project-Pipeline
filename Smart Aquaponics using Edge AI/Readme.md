# Smart Aquaponics System with Edge AI

## Project Overview

This project implements an intelligent aquaponics monitoring and control system using a Raspberry Pi with edge AI capabilities. The system continuously monitors water quality parameters and environmental conditions, making automated adjustments to maintain optimal conditions for both fish and plants.

## Features

- **Real-time monitoring** of critical parameters:
  - pH levels
  - Electrical conductivity (EC) / nutrient levels
  - Water temperature
  - Dissolved oxygen
  - Water level
  - Ambient temperature and humidity
  - Light levels

- **Automated control** of:
  - pH adjustment (via pH up/down dosing)
  - Nutrient levels
  - Water levels
  - Aeration
  - Temperature (heater control)
  - Lighting (time-based control)

- **Edge AI capabilities**:
  - Predictive analytics for pH and EC trends
  - Anomaly detection
  - System optimization
  - Automated decision making

- **Data management**:
  - Continuous logging of all sensor data
  - Historical data storage and retrieval
  - Status reporting and visualization

## Hardware Requirements

- Raspberry Pi 4 (2GB RAM minimum recommended)
- Atlas Scientific pH sensor
- Atlas Scientific EC (electrical conductivity) sensor
- Atlas Scientific DO (dissolved oxygen) sensor
- DS18B20 waterproof temperature sensor
- BME280 temperature/humidity sensor
- Water level sensor
- Light sensor (photoresistor with ADC)
- ADS1115 16-bit ADC (for analog sensors)
- Relay board (for controlling pumps, heater, etc.)
- Peristaltic pumps (for pH up, pH down, and nutrient solutions)
- Water pump
- Aerator
- Water valve (for automatic refilling)
- Heater
- LED grow lights
- Buzzer/alarm

## Software Dependencies

- Python 3.7+
- RPi.GPIO
- TensorFlow Lite (for edge AI)
- NumPy
- Pandas
- Adafruit_ADS1x15
- Adafruit_BME280
- Atlas I2C Python library

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/smart-aquaponics.git
   cd smart-aquaponics
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the system by editing `config.json` with your specific parameters.

4. Connect all hardware components according to the wiring diagram in the `docs` directory.

5. Run the setup script to initialize the system:
   ```
   python setup.py
   ```

## Usage

1. Start the system:
   ```
   python aquaponics_system.py
   ```

2. The system will begin monitoring and controlling your aquaponics setup automatically.

3. Access the web interface (if enabled) by navigating to `http://[raspberry-pi-ip]:8080` in your browser.

4. To stop the system gracefully, press Ctrl+C in the terminal.

## System Architecture

The software is organized into several modules:

- **SensorModule**: Handles all sensor interactions and data collection
- **ActuatorModule**: Controls pumps, valves, and other hardware components
- **EdgeAI**: Manages machine learning models for prediction and anomaly detection
- **DataStore**: Handles data logging, storage, and retrieval
- **AquaponicsSystem**: Main controller that integrates all components

## Wiring Diagram

```
+---------------------+
|    Raspberry Pi     |
+---------------------+
| GPIO 17 -> pH Up    |
| GPIO 18 -> pH Down  |
| GPIO 27 -> Nutrient |
| GPIO 22 -> Water    |
| GPIO 23 -> Aerator  |
| GPIO 24 -> Valve    |
| GPIO 25 -> Heater   |
| GPIO 12 -> Light    |
| GPIO 16 -> Alarm    |
| GPIO 26 <- WaterLvl |
+---------------------+
| I2C -> pH Sensor    |
| I2C -> EC Sensor    |
| I2C -> DO Sensor    |
| I2C -> BME280       |
| I2C -> ADS1115      |
+---------------------+
```

## Customization

- Modify the `CONFIG` dictionary in the main script to adjust target values, tolerances, and timing parameters.
- The AI model can be retrained with your specific data by calling `system.ai.train_model()`.
- Add additional sensors or actuators by extending the respective module classes.

## Troubleshooting

- Check `aquaponics.log` for detailed system logs and error messages.
- Ensure all sensors are properly calibrated before use.
- Verify that all GPIO pins are correctly configured in the `CONFIG` dictionary.
- For I2C device issues, run `i2cdetect -y 1` to verify device addresses.
