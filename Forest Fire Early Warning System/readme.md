# Edge-Powered Forest Fire Early Warning System

## Overview
This system uses a Raspberry Pi Zero to detect early signs of forest fires by monitoring abnormal temperature and humidity changes. By analyzing sensor data locally (edge computing), it can provide timely alerts without requiring continuous internet connectivity, making it suitable for remote forest deployments.

## Features
- **Real-time Monitoring**: Continuously tracks temperature and humidity changes
- **Edge Computing**: All processing happens on the device without cloud dependency
- **Anomaly Detection**: Identifies abnormal patterns that may indicate fire risk
- **Multi-level Alerts**: Visual and audible warnings based on threat level
- **Data Logging**: Records historical data for analysis and system improvement
- **Self-calibration**: Adapts to the normal conditions of your specific environment
- **Low Power**: Optimized for Raspberry Pi Zero's limited resources

## Hardware Requirements
- Raspberry Pi Zero (W recommended for remote data access)
- DHT22 temperature and humidity sensor
- LEDs (red, yellow, green) for status indication
- Buzzer for audible alerts
- Jumper wires and resistors (220Ω for LEDs)
- Power supply (battery pack with solar charging recommended for field deployment)
- Optional: Weatherproof enclosure

## Wiring Diagram
```
Raspberry Pi Zero    |    Components
--------------------|------------------
GPIO4 (Pin 7)       |    DHT22 data pin
GPIO17 (Pin 11)     |    Red LED (+) with 220Ω resistor
GPIO27 (Pin 13)     |    Yellow LED (+) with 220Ω resistor
GPIO22 (Pin 15)     |    Green LED (+) with 220Ω resistor
GPIO18 (Pin 12)     |    Buzzer (+)
3.3V (Pin 1)        |    DHT22 VCC, LED common anode (optional)
5V (Pin 2)          |    Alternative power for components
GND (Pin 9)         |    Common ground for all components
```

## Installation

### 1. Set up your Raspberry Pi Zero
If you're starting from scratch:
```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install required packages
sudo apt install -y python3-pip git

# Install required Python libraries
pip3 install adafruit-circuitpython-dht RPi.GPIO pandas numpy
```

### 2. Install the system
```bash
# Clone the repository
git clone https://github.com/yourusername/forest-fire-alert.git
cd forest-fire-alert

# Make the script executable
chmod +x forest_fire_monitor.py
```

### 3. Configure the system
Edit the configuration in the script to match your environment:
```python
CONFIG = {
    "sampling_interval": 60,            # Seconds between readings
    "data_window_size": 24*60,          # Number of readings to keep
    "temp_rise_threshold": 5.0,         # Celsius rise to trigger warning
    "humidity_drop_threshold": 15.0,    # Humidity drop to trigger warning
    "high_temp_threshold": 35.0,        # Absolute high temperature threshold
    "low_humidity_threshold": 20.0,     # Absolute low humidity threshold
    "log_file": "forest_fire_monitor.csv",
    "alert_log": "fire_alerts.json",
    "calibration_period": 24*60         # Initial calibration period
}
```

### 4. Run the system
```bash
# Run directly
python3 forest_fire_monitor.py

# Or use nohup to keep it running after logout
nohup python3 forest_fire_monitor.py > forest_monitor.log 2>&1 &
```

### 5. Set up autostart (optional)
To make the system start automatically on boot:
```bash
# Create a systemd service file
sudo nano /etc/systemd/system/forest-fire-monitor.service

# Add these lines:
[Unit]
Description=Forest Fire Early Warning System
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/forest-fire-alert/forest_fire_monitor.py
WorkingDirectory=/home/pi/forest-fire-alert
StandardOutput=inherit
StandardError=inherit
Restart=always
User=pi

[Install]
WantedBy=multi-user.target

# Enable and start the service
sudo systemctl enable forest-fire-monitor.service
sudo systemctl start forest-fire-monitor.service
```

## How It Works

### Alert Levels
- **NORMAL** (Green LED): Normal conditions, no significant changes detected
- **WARNING** (Yellow LED): Abnormal trends detected:
  - Temperature rise of 5°C or more within an hour
  - Humidity drop of 15% or more within an hour
- **DANGER** (Red LED + Buzzer): Critical conditions detected:
  - Temperature exceeds 35°C AND humidity falls below 20%
  - Immediate inspection recommended

### Calibration
The system requires an initial calibration period (default: 24 hours) to learn the normal patterns in your environment. During this time, it will collect data but won't trigger alerts.

### Data Analysis
The system maintains a rolling window of data (default: 24 hours) to detect trends and anomalies. It analyzes:
- Short-term rapid changes (hourly trends)
- Absolute threshold violations
- Pattern deviations from historical norms

## Extending the System

### Remote Monitoring
For remote monitoring capabilities:
1. Enable a web server on the Raspberry Pi
2. Set up data visualization with tools like Grafana
3. Implement LoRa communication for forest deployments without WiFi

### Multiple Sensor Nodes
Create a network of sensors to cover larger areas:
1. Deploy multiple Raspberry Pi Zero units
2. Set up a central coordinator node
3. Implement a mesh network for data aggregation

### Solar Power
For field deployment:
1. Add a solar panel (5W minimum recommended)
2. Connect a charge controller
3. Use a high-capacity battery (10,000mAh or higher)

## Troubleshooting

### Sensor Issues
- **No readings**: Check wiring connections and power supply
- **Erratic readings**: Ensure sensor is not exposed to direct sunlight
- **Constant errors**: Try replacing the DHT22 sensor

### System Issues
- **Not starting**: Check logs with `journalctl -u forest-fire-monitor.service`
- **High CPU usage**: Increase sampling interval in configuration
- **Memory errors**: Reduce data window size in configuration
