# PicoLogic Logic Analyzer

PicoLogic is a lightweight, portable logic analyzer library for ESP32 (MicroPython) and Raspberry Pi platforms. It provides digital signal capture, protocol decoding, and a web-based user interface for real-time signal visualization and analysis.

## Features

- **Multi-platform support**: Works on both ESP32 (MicroPython) and Raspberry Pi
- **High-speed sampling**: Supports sampling rates up to 1MHz (platform dependent)
- **Protocol decoding**: Built-in decoders for common protocols:
  - I²C (Inter-Integrated Circuit)
  - SPI (Serial Peripheral Interface)
  - UART (Universal Asynchronous Receiver/Transmitter)
- **Triggering system**: Customizable triggers with programmable conditions and actions
- **Web interface**: Real-time signal visualization and analysis through a browser
- **WebSocket streaming**: Live data streaming to connected clients
- **Circular buffer**: Efficient memory management for continuous capture

## Installation

### Raspberry Pi

1. Install the required dependencies:

```bash
pip install RPi.GPIO websockets aiohttp
```

2. Clone the repository or download the library files:

```bash
git clone https://github.com/yourusername/picologic.git
cd picologic
```

3. Install the library:

```bash
pip install -e .
```

### ESP32 (MicroPython)

1. Flash MicroPython to your ESP32 if you haven't already.
2. Copy the `picologic.py` file to your ESP32 using a tool like `ampy`:

```bash
ampy --port /dev/ttyUSB0 put picologic.py
```

## Hardware Setup

### Raspberry Pi

Connect your logic probes to the GPIO pins as defined in your script. Be sure to use appropriate level shifters if working with signals that aren't 3.3V compatible.

Default pin configuration (BCM numbering):

- **I²C**: SCL (GPIO 3), SDA (GPIO 2)
- **SPI**: MOSI (GPIO 10), MISO (GPIO 9), SCK (GPIO 11), CS (GPIO 8)
- **UART**: RX (GPIO 15), TX (GPIO 14)
- **Trigger**: GPIO 17
- **Status LED**: GPIO 18

### ESP32

Default pin configuration:

- **I²C**: SCL (GPIO 22), SDA (GPIO 21)
- **Trigger**: GPIO 5
- **Status LED**: GPIO 2 (built-in LED on most ESP32 development boards)

## Basic Usage

### Initializing the Logic Analyzer

```python
import picologic

# Define pins to monitor
channels = [2, 3, 4, 5]  # GPIO pins

# Create analyzer instance
analyzer = picologic.create_analyzer(
    channels=channels,
    sample_rate=500000,  # 500 kHz
    buffer_size=100000   # 100K samples
)
```

### Setting Up Triggers

```python
# Define trigger condition - trigger when channel 0 goes high
def trigger_condition(timestamp, sample):
    return sample[0] == 1

# Define action to take when triggered
def trigger_action(timestamp, sample):
    print(f"Triggered at {timestamp}")

# Add trigger to analyzer
analyzer.add_trigger(trigger_condition, trigger_action)
```

### Starting and Stopping Capture

```python
# Start capturing data
analyzer.start_capture()

# ... do something while capturing ...

# Stop capturing
analyzer.stop_capture()

# Clean up resources when done
analyzer.cleanup()
```

### Starting the Web Interface

```python
import asyncio

async def start_server():
    # Start WebSocket server for real-time data
    await analyzer.start_websocket_server(host='0.0.0.0', port=8765)
    
    # Start web server for UI
    server = picologic.WebServer(analyzer, host='0.0.0.0', http_port=8080)
    await server.start()

# Run the async event loop
loop = asyncio.get_event_loop()
loop.run_until_complete(start_server())
loop.run_forever()
```

## Protocol Decoding

### Decoding I²C

```python
# Define mapping from logical signals to physical channels
i2c_mapping = {
    'scl': 0,  # SCL is on the first channel
    'sda': 1   # SDA is on the second channel
}

# Decode I²C data
decoded = analyzer.decode_protocol('i2c', {}, i2c_mapping)
```

### Decoding SPI

```python
# Define mapping from logical signals to physical channels
spi_mapping = {
    'mosi': 0,
    'miso': 1,
    'sck': 2,
    'cs': 3
}

# SPI configuration
spi_config = {
    'cpol': 0,        # Clock polarity
    'cpha': 0,        # Clock phase
    'bit_order': 'msb', # MSB or LSB first
    'word_size': 8    # Bits per word
}

# Decode SPI data
decoded = analyzer.decode_protocol('spi', spi_config, spi_mapping)
```

### Decoding UART

```python
# Define mapping from logical signals to physical channels
uart_mapping = {
    'rx': 0
}

# UART configuration
uart_config = {
    'baud_rate': 9600,
    'data_bits': 8,
    'parity': None,   # None, 'even', or 'odd'
    'stop_bits': 1
}

# Decode UART data
decoded = analyzer.decode_protocol('uart', uart_config, uart_mapping)
```

## Example Applications

We've included example applications for both platforms:

- `rpi_example.py` - Raspberry Pi example with multi-protocol decoding
- `esp32_example.py` - ESP32 example with I²C decoding and WiFi connectivity

Run the examples to see PicoLogic in action.

## Web Interface

Once the web server is running:

1. Open your browser and navigate to `http://<device-ip>:8080`
2. The interface will connect to the WebSocket server and display real-time signals
3. Use the controls to:
   - Start/stop capture
   - Zoom in/out on the signal display
   - Select protocols to decode
   - Configure trigger settings
   - Export captured data

## Performance Considerations

- **Sample rate limitations**: Maximum reliable sample rates depend on the hardware:
  - ESP32: Up to ~5MHz for short captures, ~1MHz for continuous
  - Raspberry Pi: Up to ~10MHz for short captures, ~2MHz for continuous
- **Buffer size**: Larger buffers consume more memory. Adjust according to your device's capabilities.
- **Web interface performance**: Displaying large amounts of data may slow down the browser.

## Extending PicoLogic

### Adding Custom Protocol Decoders

Create a new decoder by subclassing `ProtocolDecoder`:

```python
class MyProtocolDecoder(picologic.ProtocolDecoder):
    def decode(self, data, channel_mapping):
        # Your decoding logic here
        result = []
        
        # Process the data
        for timestamp, sample in data:
            # Decode logic
            # Add decoded items to result
        
        return result
```

Register your custom decoder:

```python
analyzer.register_protocol_decoder('my_protocol', MyProtocolDecoder)
```

## Troubleshooting

- **Missed samples**: If you're seeing inconsistent decoding, try reducing the sample rate.
- **Connection issues**: Ensure your network settings allow connections to the web/WebSocket ports.
- **Memory errors on ESP32**: Reduce buffer size or simplify the operation.
- **Performance issues**: Disable protocol decoding when not needed to reduce CPU load.

## License

PicoLogic is released under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests.

## Credits

PicoLogic was developed by [Your Name/Organization] to provide an accessible logic analyzer solution for makers, hobbyists, and educators.
