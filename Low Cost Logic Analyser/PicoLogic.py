"""
PicoLogic - Low-cost Logic Analyzer Library for ESP32 and Raspberry Pi
"""
import time
import json
import asyncio
import websockets
from collections import deque
from threading import Thread, Event
import logging

# Platform detection
try:
    import RPi.GPIO as GPIO
    PLATFORM = "RPI"
except ImportError:
    try:
        import machine
        PLATFORM = "ESP32"
    except ImportError:
        PLATFORM = "UNKNOWN"
        logging.error("Unsupported platform. This library requires ESP32 (MicroPython) or Raspberry Pi.")

class LogicAnalyzer:
    """Main Logic Analyzer class that handles data acquisition and analysis"""
    
    def __init__(self, channels, sample_rate=1000000, buffer_size=100000):
        """
        Initialize the logic analyzer
        
        Args:
            channels (list): List of GPIO pin numbers to monitor
            sample_rate (int): Target sample rate in Hz
            buffer_size (int): Size of the circular buffer for samples
        """
        self.channels = channels
        self.num_channels = len(channels)
        self.sample_rate = sample_rate
        self.buffer = deque(maxlen=buffer_size)
        self.triggers = {}
        self.running = False
        self.stop_event = Event()
        self.timestamp_offset = 0
        self.capture_thread = None
        self.protocol_decoders = {}
        
        # Set up GPIO based on platform
        if PLATFORM == "RPI":
            GPIO.setmode(GPIO.BCM)
            for channel in channels:
                GPIO.setup(channel, GPIO.IN)
        elif PLATFORM == "ESP32":
            self.pins = [machine.Pin(channel, machine.Pin.IN) for channel in channels]
        
        # Initialize websocket server
        self.websocket_server = None
        self.clients = set()
    
    def start_capture(self):
        """Start capturing data"""
        if self.running:
            return False
        
        self.stop_event.clear()
        self.running = True
        self.timestamp_offset = time.time()
        self.buffer.clear()
        
        self.capture_thread = Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        return True
    
    def stop_capture(self):
        """Stop capturing data"""
        if not self.running:
            return False
        
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        self.running = False
        
        return True
    
    def _capture_loop(self):
        """Main capture loop - runs in a separate thread"""
        interval = 1.0 / self.sample_rate
        last_sample_time = time.time()
        
        while not self.stop_event.is_set():
            current_time = time.time()
            if current_time - last_sample_time >= interval:
                sample = self._read_channels()
                timestamp = current_time - self.timestamp_offset
                self.buffer.append((timestamp, sample))
                
                # Check triggers
                self._check_triggers(timestamp, sample)
                
                last_sample_time = current_time
            
            # Yield to other threads
            time.sleep(interval / 10)
    
    def _read_channels(self):
        """Read the state of all channels"""
        if PLATFORM == "RPI":
            return [GPIO.input(ch) for ch in self.channels]
        elif PLATFORM == "ESP32":
            return [pin.value() for pin in self.pins]
        return [0] * self.num_channels
    
    def add_trigger(self, condition_func, action_func, trigger_id=None):
        """
        Add a trigger that executes when condition is met
        
        Args:
            condition_func: Function that takes (timestamp, sample) and returns bool
            action_func: Function to call when trigger condition is met
            trigger_id: Optional ID for the trigger
        
        Returns:
            ID of the added trigger
        """
        if trigger_id is None:
            trigger_id = f"trigger_{len(self.triggers)}"
        
        self.triggers[trigger_id] = {
            "condition": condition_func,
            "action": action_func,
            "enabled": True
        }
        
        return trigger_id
    
    def _check_triggers(self, timestamp, sample):
        """Check if any triggers should fire"""
        for trigger_id, trigger in self.triggers.items():
            if trigger["enabled"] and trigger["condition"](timestamp, sample):
                trigger["action"](timestamp, sample)
    
    async def start_websocket_server(self, host='0.0.0.0', port=8765):
        """Start the WebSocket server for the web UI"""
        self.websocket_server = await websockets.serve(self._handle_websocket, host, port)
        logging.info(f"WebSocket server started at ws://{host}:{port}")
        return self.websocket_server
    
    async def _handle_websocket(self, websocket, path):
        """Handle WebSocket connections"""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                await self._process_websocket_message(websocket, message)
        finally:
            self.clients.remove(websocket)
    
    async def _process_websocket_message(self, websocket, message):
        """Process incoming WebSocket messages"""
        try:
            data = json.loads(message)
            command = data.get('command')
            
            if command == 'start_capture':
                self.start_capture()
                await websocket.send(json.dumps({'status': 'capture_started'}))
            
            elif command == 'stop_capture':
                self.stop_capture()
                await websocket.send(json.dumps({'status': 'capture_stopped'}))
            
            elif command == 'get_data':
                # Get portion of buffer based on start and end time
                start_time = data.get('start_time', 0)
                end_time = data.get('end_time', float('inf'))
                
                # Filter buffer for the requested time window
                filtered_data = [(t, s) for t, s in self.buffer if start_time <= t <= end_time]
                
                await websocket.send(json.dumps({
                    'type': 'data',
                    'data': filtered_data,
                    'channels': self.channels
                }))
            
            elif command == 'decode_protocol':
                protocol = data.get('protocol')
                config = data.get('config', {})
                channel_mapping = data.get('channel_mapping', {})
                
                if protocol in self.protocol_decoders:
                    decoded = self.decode_protocol(protocol, config, channel_mapping)
                    await websocket.send(json.dumps({
                        'type': 'decoded_protocol',
                        'protocol': protocol,
                        'data': decoded
                    }))
                else:
                    await websocket.send(json.dumps({
                        'error': f'Protocol decoder {protocol} not available'
                    }))
        
        except json.JSONDecodeError:
            await websocket.send(json.dumps({'error': 'Invalid JSON'}))
        except Exception as e:
            await websocket.send(json.dumps({'error': str(e)}))
    
    async def broadcast_update(self, data):
        """Broadcast data to all connected WebSocket clients"""
        if not self.clients:
            return
        
        message = json.dumps(data)
        await asyncio.gather(*[client.send(message) for client in self.clients])
    
    def register_protocol_decoder(self, protocol_name, decoder_class):
        """Register a protocol decoder"""
        self.protocol_decoders[protocol_name] = decoder_class
    
    def decode_protocol(self, protocol, config=None, channel_mapping=None):
        """
        Decode a protocol from captured data
        
        Args:
            protocol: Name of the protocol to decode
            config: Protocol-specific configuration
            channel_mapping: Mapping of logical signals to physical channels
        
        Returns:
            Decoded protocol data
        """
        if protocol not in self.protocol_decoders:
            raise ValueError(f"Protocol decoder '{protocol}' not registered")
        
        decoder = self.protocol_decoders[protocol](config)
        return decoder.decode(list(self.buffer), channel_mapping)
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_capture()
        if PLATFORM == "RPI":
            GPIO.cleanup()


class ProtocolDecoder:
    """Base class for protocol decoders"""
    
    def __init__(self, config=None):
        """
        Initialize the protocol decoder
        
        Args:
            config: Protocol-specific configuration
        """
        self.config = config or {}
    
    def decode(self, data, channel_mapping):
        """
        Decode protocol from raw data
        
        Args:
            data: List of (timestamp, values) tuples
            channel_mapping: Dictionary mapping protocol signals to channel indices
        
        Returns:
            Decoded protocol data
        """
        raise NotImplementedError("Subclasses must implement decode()")


class I2CDecoder(ProtocolDecoder):
    """Decoder for I²C protocol"""
    
    def decode(self, data, channel_mapping):
        """Decode I²C protocol"""
        sda_channel = channel_mapping.get('sda')
        scl_channel = channel_mapping.get('scl')
        
        if sda_channel is None or scl_channel is None:
            raise ValueError("I²C decoding requires both SDA and SCL channels")
        
        result = []
        state = "IDLE"
        byte = 0
        bit_count = 0
        address = 0
        is_read = False
        
        # Detect start/stop conditions and decode bytes
        for i in range(1, len(data)):
            prev_time, prev_sample = data[i-1]
            curr_time, curr_sample = data[i]
            
            prev_sda = prev_sample[sda_channel]
            prev_scl = prev_sample[scl_channel]
            curr_sda = curr_sample[sda_channel]
            curr_scl = curr_sample[scl_channel]
            
            # START condition: SDA falls while SCL is high
            if prev_sda == 1 and curr_sda == 0 and prev_scl == 1 and curr_scl == 1:
                if state != "IDLE":
                    # Repeated START
                    result.append({
                        'type': 'REPEATED_START',
                        'timestamp': curr_time
                    })
                else:
                    result.append({
                        'type': 'START',
                        'timestamp': curr_time
                    })
                state = "ADDRESS"
                bit_count = 0
                byte = 0
            
            # STOP condition: SDA rises while SCL is high
            elif prev_sda == 0 and curr_sda == 1 and prev_scl == 1 and curr_scl == 1:
                result.append({
                    'type': 'STOP',
                    'timestamp': curr_time
                })
                state = "IDLE"
            
            # Clock rising edge - sample data
            elif prev_scl == 0 and curr_scl == 1:
                if state == "ADDRESS" and bit_count < 8:
                    # Shift in address bits, MSB first
                    byte = (byte << 1) | curr_sda
                    bit_count += 1
                    
                    if bit_count == 8:
                        # Last bit is R/W
                        is_read = byte & 0x01
                        address = byte >> 1
                        result.append({
                            'type': 'ADDRESS',
                            'address': address,
                            'read': is_read,
                            'timestamp': curr_time
                        })
                        state = "ACK1"
                
                elif state == "ACK1" or state == "ACK2":
                    # ACK bit (low = ACK, high = NACK)
                    ack = not curr_sda
                    result.append({
                        'type': 'ACK' if ack else 'NACK',
                        'timestamp': curr_time
                    })
                    
                    if state == "ACK1":
                        state = "DATA"
                        byte = 0
                        bit_count = 0
                    else:  # ACK2
                        if ack:
                            state = "DATA"
                            byte = 0
                            bit_count = 0
                        else:
                            # NACK often precedes STOP
                            state = "IDLE"
                
                elif state == "DATA" and bit_count < 8:
                    # Shift in data bits, MSB first
                    byte = (byte << 1) | curr_sda
                    bit_count += 1
                    
                    if bit_count == 8:
                        result.append({
                            'type': 'DATA',
                            'value': byte,
                            'timestamp': curr_time
                        })
                        state = "ACK2"
            
        return result


class SPIDecoder(ProtocolDecoder):
    """Decoder for SPI protocol"""
    
    def decode(self, data, channel_mapping):
        """Decode SPI protocol"""
        mosi_channel = channel_mapping.get('mosi')
        miso_channel = channel_mapping.get('miso')
        sck_channel = channel_mapping.get('sck')
        cs_channel = channel_mapping.get('cs')
        
        if sck_channel is None or (mosi_channel is None and miso_channel is None):
            raise ValueError("SPI decoding requires SCK and at least one of MOSI/MISO channels")
        
        # Get configuration
        cpol = self.config.get('cpol', 0)  # Clock polarity
        cpha = self.config.get('cpha', 0)  # Clock phase
        bit_order = self.config.get('bit_order', 'msb')  # 'msb' or 'lsb'
        word_size = self.config.get('word_size', 8)  # Bits per word
        
        result = []
        transaction_active = False
        mosi_byte = 0
        miso_byte = 0
        bit_count = 0
        
        sample_edge = (1 if cpha else 0)  # Which clock edge to sample on
        
        for i in range(1, len(data)):
            prev_time, prev_sample = data[i-1]
            curr_time, curr_sample = data[i]
            
            prev_sck = prev_sample[sck_channel] ^ cpol  # Adjust for CPOL
            curr_sck = curr_sample[sck_channel] ^ cpol
            
            # Check if CS is active (if CS pin is provided)
            if cs_channel is not None:
                cs_active = curr_sample[cs_channel] == 0  # CS is active low
                
                # Detect CS activation
                if not transaction_active and cs_active:
                    transaction_active = True
                    result.append({
                        'type': 'CS_ACTIVE',
                        'timestamp': curr_time
                    })
                    mosi_byte = 0
                    miso_byte = 0
                    bit_count = 0
                
                # Detect CS deactivation
                elif transaction_active and not cs_active:
                    transaction_active = False
                    result.append({
                        'type': 'CS_INACTIVE',
                        'timestamp': curr_time
                    })
                    # Add partial byte if any
                    if bit_count > 0:
                        result.append({
                            'type': 'PARTIAL',
                            'mosi': mosi_byte if mosi_channel is not None else None,
                            'miso': miso_byte if miso_channel is not None else None,
                            'bits': bit_count,
                            'timestamp': curr_time
                        })
            else:
                # If no CS pin, assume always active
                transaction_active = True
            
            # Check for sampling edge
            sampling_edge = (prev_sck == 0 and curr_sck == 1) if sample_edge == 1 else (prev_sck == 1 and curr_sck == 0)
            
            if transaction_active and sampling_edge:
                # Sample data on appropriate clock edge
                if mosi_channel is not None:
                    bit = curr_sample[mosi_channel]
                    if bit_order == 'msb':
                        mosi_byte = (mosi_byte << 1) | bit
                    else:
                        mosi_byte = mosi_byte | (bit << bit_count)
                
                if miso_channel is not None:
                    bit = curr_sample[miso_channel]
                    if bit_order == 'msb':
                        miso_byte = (miso_byte << 1) | bit
                    else:
                        miso_byte = miso_byte | (bit << bit_count)
                
                bit_count += 1
                
                # Word complete
                if bit_count == word_size:
                    result.append({
                        'type': 'DATA',
                        'mosi': mosi_byte if mosi_channel is not None else None,
                        'miso': miso_byte if miso_channel is not None else None,
                        'timestamp': curr_time
                    })
                    mosi_byte = 0
                    miso_byte = 0
                    bit_count = 0
        
        return result


class UARTDecoder(ProtocolDecoder):
    """Decoder for UART protocol"""
    
    def decode(self, data, channel_mapping):
        """Decode UART protocol"""
        rx_channel = channel_mapping.get('rx')
        if rx_channel is None:
            raise ValueError("UART decoding requires an RX channel")
        
        # Get configuration
        baud_rate = self.config.get('baud_rate', 9600)
        data_bits = self.config.get('data_bits', 8)
        parity = self.config.get('parity', None)  # None, 'even', or 'odd'
        stop_bits = self.config.get('stop_bits', 1)
        
        # Calculate bit duration in seconds
        bit_duration = 1.0 / baud_rate
        
        result = []
        state = "IDLE"
        byte = 0
        bit_count = 0
        start_time = 0
        
        for i in range(1, len(data)):
            prev_time, prev_sample = data[i-1]
            curr_time, curr_sample = data[i]
            
            prev_rx = prev_sample[rx_channel]
            curr_rx = curr_sample[rx_channel]
            
            # Detect start bit (high to low transition)
            if state == "IDLE" and prev_rx == 1 and curr_rx == 0:
                state = "START"
                start_time = curr_time
                result.append({
                    'type': 'START_BIT',
                    'timestamp': curr_time
                })
                bit_count = 0
                byte = 0
            
            # Sample in the middle of each bit
            elif state == "START" and curr_time - start_time >= bit_duration * 1.5:
                # Finished start bit, move to data bits
                state = "DATA"
                start_time = curr_time
            
            elif state == "DATA" and curr_time - start_time >= bit_duration:
                # Sample the data bit
                if bit_count < data_bits:
                    # LSB first for UART
                    byte |= (curr_rx << bit_count)
                    bit_count += 1
                    
                    if bit_count == data_bits:
                        if parity:
                            state = "PARITY"
                        else:
                            state = "STOP"
                            result.append({
                                'type': 'DATA',
                                'value': byte,
                                'timestamp': curr_time
                            })
                
                start_time = curr_time
            
            elif state == "PARITY" and curr_time - start_time >= bit_duration:
                # Check parity
                expected_parity = None
                if parity == "even":
                    # Count 1s in byte
                    bit_sum = sum([(byte >> i) & 1 for i in range(data_bits)])
                    expected_parity = 0 if (bit_sum % 2 == 0) else 1
                elif parity == "odd":
                    bit_sum = sum([(byte >> i) & 1 for i in range(data_bits)])
                    expected_parity = 1 if (bit_sum % 2 == 0) else 0
                
                parity_ok = curr_rx == expected_parity
                result.append({
                    'type': 'PARITY',
                    'value': curr_rx,
                    'expected': expected_parity,
                    'error': not parity_ok,
                    'timestamp': curr_time
                })
                
                state = "STOP"
                start_time = curr_time
            
            elif state == "STOP" and curr_time - start_time >= bit_duration * stop_bits:
                # Check stop bit(s)
                if curr_rx == 1:
                    result.append({
                        'type': 'STOP_BIT',
                        'ok': True,
                        'timestamp': curr_time
                    })
                else:
                    result.append({
                        'type': 'STOP_BIT',
                        'ok': False,
                        'error': True,
                        'timestamp': curr_time
                    })
                
                state = "IDLE"
        
        return result


# Web server for UI
class WebServer:
    """Web server to host the UI"""
    
    def __init__(self, analyzer, host='0.0.0.0', http_port=8080, ws_port=8765):
        """
        Initialize the web server
        
        Args:
            analyzer: LogicAnalyzer instance
            host: Host address to bind to
            http_port: HTTP port for the web UI
            ws_port: WebSocket port for real-time data
        """
        self.analyzer = analyzer
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.http_server = None
    
    async def start(self):
        """Start both HTTP and WebSocket servers"""
        try:
            import aiohttp
            from aiohttp import web
        except ImportError:
            logging.error("aiohttp package required for web server functionality")
            return False
        
        app = web.Application()
        app.router.add_get('/', self._handle_index)
        app.router.add_get('/api/status', self._handle_status)
        app.router.add_static('/static', 'static')
        
        # Start both servers
        await self.analyzer.start_websocket_server(self.host, self.ws_port)
        
        runner = web.AppRunner(app)
        await runner.setup()
        self.http_server = web.TCPSite(runner, self.host, self.http_port)
        await self.http_server.start()
        
        logging.info(f"HTTP server started at http://{self.host}:{self.http_port}")
        
        return True
    
    async def _handle_index(self, request):
        """Serve the main UI page"""
        try:
            with open('static/index.html', 'r') as file:
                content = file.read()
                content = content.replace('{{WS_PORT}}', str(self.ws_port))
                return web.Response(text=content, content_type='text/html')
        except FileNotFoundError:
            return web.Response(text="UI files not found", status=404)
    
    async def _handle_status(self, request):
        """API endpoint for status information"""
        status = {
            'running': self.analyzer.running,
            'channels': self.analyzer.channels,
            'sample_rate': self.analyzer.sample_rate,
            'buffer_size': self.analyzer.buffer.maxlen,
            'buffer_used': len(self.analyzer.buffer)
        }
        return web.json_response(status)


# Helper functions
def create_analyzer(channels, sample_rate=1000000, buffer_size=100000):
    """Create and set up a logic analyzer instance"""
    analyzer = LogicAnalyzer(channels, sample_rate, buffer_size)
    
    # Register protocol decoders
    analyzer.register_protocol_decoder('i2c', I2CDecoder)
    analyzer.register_protocol_decoder('spi', SPIDecoder)
    analyzer.register_protocol_decoder('uart', UARTDecoder)
    
    return analyzer

async def start_web_server(analyzer, host='0.0.0.0', http_port=8080, ws_port=8765):
    """Start the web server and UI"""
    server = WebServer(analyzer, host, http_port, ws_port)
    await server.start()
    return server
