"""
ESP32 PicoLogic Example - Simple Logic Analyzer using PicoLogic library
This example sets up a logic analyzer to monitor GPIO pins on ESP32 and decode I2C protocol
"""
import time
from machine import Pin, I2C
import network
import picologic  # Importing our PicoLogic library

# Network configuration for web interface
WIFI_SSID = "YourWiFiName"
WIFI_PASSWORD = "YourWiFiPassword"

# Define pins to be monitored
SCL_PIN = 22  # I2C clock pin
SDA_PIN = 21  # I2C data pin
TRIGGER_PIN = 5  # Pin to trigger capture

# LED indicator
led = Pin(2, Pin.OUT)

def connect_wifi():
    """Connect to WiFi network"""
    print(f"Connecting to WiFi: {WIFI_SSID}")
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if not wlan.isconnected():
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        
        # Wait for connection with timeout
        max_wait = 10
        while max_wait > 0:
            if wlan.isconnected():
                break
            max_wait -= 1
            print("Waiting for connection...")
            time.sleep(1)
            led.value(not led.value())  # Blink LED while connecting
    
    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"Connected to WiFi. IP address: {ip}")
        led.value(1)  # LED on when connected
        return ip
    else:
        print("Failed to connect to WiFi")
        led.value(0)
        return None

def setup_i2c_device():
    """Setup a simple I2C device for demonstration"""
    # This is just to generate some I2C traffic
    try:
        i2c = I2C(0, scl=Pin(SCL_PIN), sda=Pin(SDA_PIN), freq=100000)
        print("I2C initialized. Available devices:", [hex(addr) for addr in i2c.scan()])
        return i2c
    except Exception as e:
        print("Error setting up I2C:", e)
        return None

def trigger_condition(timestamp, sample):
    """Define a trigger condition - trigger when pin changes from low to high"""
    # Check if TRIGGER_PIN (at index 2 in our channel list) transitioned from 0 to 1
    return sample[2] == 1 and prev_samples and prev_samples[-1][1][2] == 0

def trigger_action(timestamp, sample):
    """Action to take when trigger condition is met"""
    print(f"Trigger activated at time {timestamp:.6f}s")
    led.value(not led.value())  # Blink LED on trigger

def generate_i2c_traffic(i2c):
    """Generate some I2C traffic for demonstration"""
    if i2c:
        # Read from common I2C addresses
        for addr in [0x23, 0x3C, 0x68, 0x76]:
            try:
                i2c.writeto(addr, b'\x00')  # Send a read command
                print(f"Sent I2C command to device 0x{addr:02x}")
            except:
                pass
            time.sleep(0.1)

def main():
    # Connect to WiFi
    ip = connect_wifi()
    if not ip:
        return
    
    print("Setting up logic analyzer...")
    
    # Define channels to monitor - we'll monitor SDA, SCL, and TRIGGER pins
    channels = [SDA_PIN, SCL_PIN, TRIGGER_PIN]
    
    # Create logic analyzer instance
    analyzer = picologic.create_analyzer(
        channels=channels,
        sample_rate=500000,  # 500 kHz sample rate
        buffer_size=50000    # 50K sample buffer
    )
    
    # Setup trigger
    prev_samples = []  # Keep track of previous samples for edge detection
    analyzer.add_trigger(trigger_condition, trigger_action)
    
    # Setup I2C for generating test signals
    i2c_device = setup_i2c_device()
    
    # Start the websocket server
    try:
        import uasyncio as asyncio
        
        async def run_server():
            print(f"Starting WebSocket server on ws://{ip}:8765")
            await analyzer.start_websocket_server(host=ip, port=8765)
            
            # Continuously generate I2C traffic while server is running
            while True:
                generate_i2c_traffic(i2c_device)
                await asyncio.sleep(1)
        
        # Start the analyzer
        analyzer.start_capture()
        print("Logic analyzer capture started")
        print(f"Access the web interface at http://{ip}:8080")
        
        # Run the event loop
        loop = asyncio.get_event_loop()
        loop.create_task(run_server())
        loop.run_forever()
        
    except ImportError:
        print("uasyncio not available, running in simple mode")
        # Start capturing data
        analyzer.start_capture()
        
        # Generate some I2C traffic in a loop
        try:
            while True:
                generate_i2c_traffic(i2c_device)
                time.sleep(1)
        except KeyboardInterrupt:
            analyzer.stop_capture()
            analyzer.cleanup()
            print("Capture stopped")
    
    finally:
        # Cleanup when exiting
        analyzer.cleanup()

if __name__ == "__main__":
    main()
