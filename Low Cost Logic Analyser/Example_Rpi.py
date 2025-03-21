"""
Raspberry Pi PicoLogic Example - Logic Analyzer using PicoLogic library
This example sets up a logic analyzer to monitor GPIO pins on Raspberry Pi and decode multiple protocols
"""
import time
import threading
import RPi.GPIO as GPIO
import picologic
import asyncio

# Define GPIO pins to monitor - Using BCM numbering
SCL_PIN = 3    # I2C SCL (GPIO 3)
SDA_PIN = 2    # I2C SDA (GPIO 2)
MOSI_PIN = 10  # SPI MOSI (GPIO 10)
MISO_PIN = 9   # SPI MISO (GPIO 9)
SCK_PIN = 11   # SPI Clock (GPIO 11)
CS_PIN = 8     # SPI Chip Select (GPIO 8)
RX_PIN = 15    # UART RX (GPIO 15)
TX_PIN = 14    # UART TX (GPIO 14)
TRIGGER_PIN = 17  # Trigger pin (GPIO 17)

# LED indicator for status
LED_PIN = 18

def setup_pins():
    """Setup GPIO pins for status indication"""
    GPIO.setwarnings(False)
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.output(LED_PIN, GPIO.LOW)

def trigger_condition(timestamp, sample):
    """
    Define trigger condition - trigger when our trigger pin goes HIGH
    
    Args:
        timestamp: Current timestamp
        sample: List of current pin states
        
    Returns:
        Boolean indicating if trigger condition is met
    """
    # Get trigger pin index in our monitored channels list
    trigger_idx = monitored_channels.index(TRIGGER_PIN)
    return sample[trigger_idx] == 1

def trigger_action(timestamp, sample):
    """
    Action to take when trigger condition is met
    
    Args:
        timestamp: Current timestamp
        sample: List of current pin states
    """
    print(f"Trigger activated at time {timestamp:.6f}s")
    # Blink LED when triggered
    GPIO.output(LED_PIN, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(LED_PIN, GPIO.LOW)

def simulate_i2c_traffic():
    """Generate some I2C traffic for demonstration"""
    try:
        import smbus
        bus = smbus.SMBus(1)  # Use I2C bus 1
        
        # Try to read from some common I2C addresses
        addresses = [0x23, 0x3C, 0x68, 0x76]
        for addr in addresses:
            try:
                # Try to read a byte from register 0
                bus.read_byte_data(addr, 0)
                print(f"Read from I2C device at address 0x{addr:02x}")
            except:
                pass
            time.sleep(0.2)
    except ImportError:
        print("smbus module not available for I2C simulation")

def simulate_spi_traffic():
    """Generate some SPI traffic for demonstration"""
    try:
        import spidev
        spi = spidev.SpiDev()
        spi.open(0, 0)  # Bus 0, Device 0
        spi.max_speed_hz = 1000000
        
        # Send some test data
        spi.xfer2([0x01, 0x02, 0x03])
        time.sleep(0.1)
        spi.xfer2([0xAA, 0xBB, 0xCC])
        spi.close()
        print("Generated SPI test traffic")
    except (ImportError, FileNotFoundError):
        print("spidev module not available for SPI simulation")

def simulate_uart_traffic():
    """Generate some UART traffic for demonstration"""
    try:
        import serial
        ser = serial.Serial("/dev/ttyS0", 9600)
        ser.write(b"PicoLogic UART Test\r\n")
        ser.close()
        print("Generated UART test traffic")
    except (ImportError, serial.SerialException):
        print("pyserial module not available or UART access error")

def generate_test_traffic():
    """Generate test traffic for all protocols in a loop"""
    while True:
        simulate_i2c_traffic()
        simulate_spi_traffic()
        simulate_uart_traffic()
        time.sleep(1)

async def start_server(analyzer):
    """Start the web and WebSocket servers"""
    print("Starting web server...")
    server = picologic.WebServer(analyzer, host='0.0.0.0', http_port=8080, ws_port=8765)
    await server.start()
    print("Web server started at http://localhost:8080")
    print("WebSocket server started at ws://localhost:8765")

async def run_periodic_updates(analyzer):
    """Periodically broadcast status updates to clients"""
    while True:
        # Get current buffer stats
        buffer_stats = {
            'type': 'status_update',
            'buffer_used': len(analyzer.buffer),
            'buffer_max': analyzer.buffer.maxlen,
            'running': analyzer.running
        }
        
        # Broadcast to all clients
        await analyzer.broadcast_update(buffer_stats)
        await asyncio.sleep(1)

def main():
    global monitored_channels
    setup_pins()
    
    # Define all channels to monitor
    monitored_channels = [
        SDA_PIN, SCL_PIN,      # I2C
        MOSI_PIN, MISO_PIN, SCK_PIN, CS_PIN,  # SPI
        RX_PIN, TX_PIN,        # UART
        TRIGGER_PIN            # Trigger
    ]
    
    print("Setting up PicoLogic Logic Analyzer...")
    
    # Create logic analyzer with all channels
    analyzer = picologic.create_analyzer(
        channels=monitored_channels,
        sample_rate=1000000,  # 1 MHz sample rate
        buffer_size=200000    # 200K sample buffer
    )
    
    # Register trigger
    analyzer.add_trigger(trigger_condition, trigger_action)
    
    # Start the traffic simulation in a separate thread
    traffic_thread = threading.Thread(target=generate_test_traffic, daemon=True)
    traffic_thread.start()
    
    # Start data capture
    analyzer.start_capture()
    print("Logic analyzer capture started")
    
    # Setup channel mappings for protocol decoders
    i2c_mapping = {
        'scl': monitored_channels.index(SCL_PIN),
        'sda': monitored_channels.index(SDA_PIN)
    }
    
    spi_mapping = {
        'mosi': monitored_channels.index(MOSI_PIN),
        'miso': monitored_channels.index(MISO_PIN),
        'sck': monitored_channels.index(SCK_PIN),
        'cs': monitored_channels.index(CS_PIN)
    }
    
    uart_mapping = {
        'rx': monitored_channels.index(RX_PIN),
        'tx': monitored_channels.index(TX_PIN)
    }
    
    # Start the async event loop
    loop = asyncio.get_event_loop()
    
    try:
        # Start the web server and periodic updates
        loop.create_task(start_server(analyzer))
        loop.create_task(run_periodic_updates(analyzer))
        
        print("PicoLogic Logic Analyzer is running")
        print("Press Ctrl+C to exit")
        
        # Run the event loop
        loop.run_forever()
        
    except KeyboardInterrupt:
        print("\nStopping Logic Analyzer...")
    finally:
        # Clean up
        analyzer.stop_capture()
        analyzer.cleanup()
        GPIO.cleanup()
        print("Logic Analyzer stopped and resources cleaned up")

if __name__ == "__main__":
    main()
