#!/usr/bin/env python3
"""
IoT Network Embedded VPN System
===============================
A lightweight VPN solution designed specifically for IoT networks to provide
secure remote access while protecting devices from external threats.
"""

import os
import sys
import logging
import argparse
import time
import signal
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Networking libraries
import socket
import ssl
import ipaddress
import netifaces

# Cryptography libraries
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509.oid import NameOID
import datetime

# For device discovery
import zeroconf
import mdns

# Configuration
DEFAULT_CONFIG_PATH = "/etc/iotvpn/config.json"
DEFAULT_CERT_PATH = "/etc/iotvpn/certs"
DEFAULT_LOG_PATH = "/var/log/iotvpn.log"
VPN_PORT = 8443
DISCOVERY_PORT = 5353

class IoTVPNServer:
    """Core VPN server implementation for IoT networks"""
    
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.running = False
        self.clients = {}
        self.devices = {}
        self.load_config()
        self.setup_logging()
        self.setup_certificates()
        
    def load_config(self) -> None:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                logging.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            logging.warning(f"Configuration file not found at {self.config_path}, using defaults")
            self.config = {
                "server_name": "iotvpn-server",
                "vpn_subnet": "10.10.0.0/24",
                "device_subnets": ["192.168.1.0/24"],
                "allowed_clients": [],
                "certificate_validity_days": 365,
                "log_level": "INFO",
                "enable_device_discovery": True,
                "enable_traffic_filtering": True,
                "enable_intrusion_detection": True
            }
            # Create default config
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                
    def setup_logging(self) -> None:
        """Configure logging"""
        log_level = getattr(logging, self.config.get("log_level", "INFO"))
        log_path = self.config.get("log_path", DEFAULT_LOG_PATH)
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        logging.info("Logging initialized")
    
    def setup_certificates(self) -> None:
        """Generate or load SSL/TLS certificates"""
        cert_path = Path(self.config.get("cert_path", DEFAULT_CERT_PATH))
        cert_path.mkdir(parents=True, exist_ok=True)
        
        server_key_path = cert_path / "server.key"
        server_cert_path = cert_path / "server.crt"
        ca_key_path = cert_path / "ca.key"
        ca_cert_path = cert_path / "ca.crt"
        
        # Check if certificates already exist
        if not (server_key_path.exists() and server_cert_path.exists() and 
                ca_key_path.exists() and ca_cert_path.exists()):
            logging.info("Certificates not found, generating new ones...")
            self._generate_certificates(
                server_key_path, 
                server_cert_path,
                ca_key_path,
                ca_cert_path
            )
        else:
            logging.info("Loading existing certificates")
            
        # Load the certificates
        with open(server_key_path, 'rb') as f:
            self.server_key = serialization.load_pem_private_key(
                f.read(),
                password=None
            )
            
        with open(server_cert_path, 'rb') as f:
            self.server_cert = x509.load_pem_x509_certificate(f.read())
            
        with open(ca_key_path, 'rb') as f:
            self.ca_key = serialization.load_pem_private_key(
                f.read(),
                password=None
            )
            
        with open(ca_cert_path, 'rb') as f:
            self.ca_cert = x509.load_pem_x509_certificate(f.read())
    
    def _generate_certificates(
        self, 
        server_key_path: Path, 
        server_cert_path: Path,
        ca_key_path: Path,
        ca_cert_path: Path
    ) -> None:
        """Generate CA and server certificates"""
        # Generate CA key
        ca_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        
        # Create CA self-signed certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "IoT Network"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IoT VPN CA"),
            x509.NameAttribute(NameOID.COMMON_NAME, "IoT VPN Root CA"),
        ])
        
        ca_cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            ca_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=self.config.get("certificate_validity_days", 365) * 2)
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False
            ), critical=True
        ).sign(ca_key, hashes.SHA256())
        
        # Generate server key
        server_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Create server certificate signed by CA
        server_subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "IoT Network"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IoT VPN Server"),
            x509.NameAttribute(NameOID.COMMON_NAME, self.config.get("server_name", "iotvpn-server")),
        ])
        
        server_cert = x509.CertificateBuilder().subject_name(
            server_subject
        ).issuer_name(
            ca_cert.subject
        ).public_key(
            server_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=self.config.get("certificate_validity_days", 365))
        ).add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False
            ), critical=True
        ).add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
            ]), critical=False
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(self.config.get("server_name", "iotvpn-server")),
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))
            ]), critical=False
        ).sign(ca_key, hashes.SHA256())
        
        # Save certificates and keys
        with open(ca_key_path, 'wb') as f:
            f.write(ca_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
            
        with open(ca_cert_path, 'wb') as f:
            f.write(ca_cert.public_bytes(serialization.Encoding.PEM))
            
        with open(server_key_path, 'wb') as f:
            f.write(server_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
            
        with open(server_cert_path, 'wb') as f:
            f.write(server_cert.public_bytes(serialization.Encoding.PEM))
            
        logging.info("Generated and saved new certificates")
        
    def generate_client_certificate(self, client_name: str) -> Tuple[bytes, bytes, bytes]:
        """Generate a client certificate for VPN authentication"""
        # Generate client key
        client_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Create client certificate signed by CA
        client_subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "IoT Network"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IoT VPN Client"),
            x509.NameAttribute(NameOID.COMMON_NAME, client_name),
        ])
        
        client_cert = x509.CertificateBuilder().subject_name(
            client_subject
        ).issuer_name(
            self.ca_cert.subject
        ).public_key(
            client_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=self.config.get("certificate_validity_days", 365))
        ).add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False
            ), critical=True
        ).add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
            ]), critical=False
        ).sign(self.ca_key, hashes.SHA256())
        
        # Add to allowed clients
        if client_name not in self.config["allowed_clients"]:
            self.config["allowed_clients"].append(client_name)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            
        # Return key, cert and CA cert
        client_key_pem = client_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        client_cert_pem = client_cert.public_bytes(serialization.Encoding.PEM)
        ca_cert_pem = self.ca_cert.public_bytes(serialization.Encoding.PEM)
        
        return client_key_pem, client_cert_pem, ca_cert_pem

    def start_server(self) -> None:
        """Start the VPN server"""
        self.running = True
        logging.info(f"Starting IoT VPN server on port {VPN_PORT}")
        
        # Set up SSL context
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_cert_chain(
            certfile=os.path.join(self.config.get("cert_path", DEFAULT_CERT_PATH), "server.crt"),
            keyfile=os.path.join(self.config.get("cert_path", DEFAULT_CERT_PATH), "server.key")
        )
        context.load_verify_locations(
            cafile=os.path.join(self.config.get("cert_path", DEFAULT_CERT_PATH), "ca.crt")
        )
        
        # Set up server socket
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', VPN_PORT))
        server.listen(5)
        
        # Start device discovery if enabled
        if self.config.get("enable_device_discovery", True):
            self.start_device_discovery()
        
        # Start traffic filtering if enabled
        if self.config.get("enable_traffic_filtering", True):
            self.setup_traffic_filtering()
            
        # Start intrusion detection if enabled
        if self.config.get("enable_intrusion_detection", True):
            self.start_intrusion_detection()
        
        try:
            # Handle client connections
            server.setblocking(False)
            logging.info("Server started successfully, waiting for connections")
            
            while self.running:
                try:
                    # Non-blocking accept with timeout
                    server.settimeout(1.0)
                    client_socket, client_address = server.accept()
                    logging.info(f"Connection from {client_address}")
                    
                    # Wrap with SSL
                    try:
                        ssl_socket = context.wrap_socket(client_socket, server_side=True)
                        # Extract client certificate info
                        cert = ssl_socket.getpeercert()
                        client_cn = None
                        for rdn in cert['subject']:
                            for attr in rdn:
                                if attr[0] == 'commonName':
                                    client_cn = attr[1]
                                    break
                        
                        if client_cn in self.config["allowed_clients"]:
                            logging.info(f"Authenticated client: {client_cn}")
                            # Start a new thread to handle this client
                            # In a real implementation, we would spawn a thread here
                            self.clients[client_cn] = {
                                "socket": ssl_socket,
                                "address": client_address,
                                "connected_at": datetime.datetime.now().isoformat()
                            }
                            # This would be handled in a separate thread
                            self.handle_client(client_cn)
                        else:
                            logging.warning(f"Unauthorized client attempt: {client_cn}")
                            ssl_socket.close()
                    except ssl.SSLError as e:
                        logging.error(f"SSL Error: {e}")
                        client_socket.close()
                except socket.timeout:
                    # This is expected for the non-blocking socket with timeout
                    continue
                except Exception as e:
                    if self.running:  # Only log if we're still supposed to be running
                        logging.error(f"Error accepting connection: {e}")
                time.sleep(0.1)  # Small sleep to prevent CPU hogging
                
        except KeyboardInterrupt:
            logging.info("Server shutdown requested")
        finally:
            self.stop_server(server)
    
    def stop_server(self, server_socket: socket.socket = None) -> None:
        """Stop the VPN server and clean up resources"""
        self.running = False
        logging.info("Stopping IoT VPN server")
        
        # Close all client connections
        for client_name, client_data in self.clients.items():
            try:
                client_data["socket"].close()
                logging.info(f"Closed connection to {client_name}")
            except Exception as e:
                logging.error(f"Error closing client connection {client_name}: {e}")
        
        # Close server socket if provided
        if server_socket:
            try:
                server_socket.close()
                logging.info("Closed server socket")
            except Exception as e:
                logging.error(f"Error closing server socket: {e}")
        
        # Stop device discovery
        self.stop_device_discovery()
        
        # Stop traffic filtering
        self.stop_traffic_filtering()
        
        # Stop intrusion detection
        self.stop_intrusion_detection()
        
        logging.info("Server shutdown complete")
    
    def handle_client(self, client_name: str) -> None:
        """Handle communication with a connected client"""
        # In a real implementation, this would run in a separate thread
        client_socket = self.clients[client_name]["socket"]
        client_address = self.clients[client_name]["address"]
        
        try:
            # Send VPN configuration to client
            vpn_config = {
                "client_id": client_name,
                "vpn_subnet": self.config["vpn_subnet"],
                "device_subnets": self.config["device_subnets"],
                "server_name": self.config["server_name"],
                "connected_devices": list(self.devices.keys())
            }
            
            client_socket.send(json.dumps(vpn_config).encode('utf-8'))
            
            # Establish VPN tunnel
            # In a real implementation, this would set up actual tunneling
            logging.info(f"VPN tunnel established with {client_name}")
            
            # Main communication loop
            buffer_size = 4096
            while self.running:
                try:
                    # Non-blocking receive with timeout
                    client_socket.settimeout(1.0)
                    data = client_socket.recv(buffer_size)
                    if not data:
                        # Client disconnected
                        logging.info(f"Client {client_name} disconnected")
                        break
                    
                    # Process client requests
                    # Here we would handle VPN tunnel data
                    # For now, just echo back
                    client_socket.send(data)
                    
                except socket.timeout:
                    # This is expected for non-blocking socket with timeout
                    continue
                except Exception as e:
                    logging.error(f"Error communicating with {client_name}: {e}")
                    break
                    
                time.sleep(0.1)  # Prevent CPU hogging
                
        except Exception as e:
            logging.error(f"Error handling client {client_name}: {e}")
        finally:
            # Clean up
            try:
                client_socket.close()
                logging.info(f"Closed connection to {client_name}")
            except:
                pass
            
            # Remove client from active clients
            if client_name in self.clients:
                del self.clients[client_name]
    
    def start_device_discovery(self) -> None:
        """Start IoT device discovery service using mDNS/Zeroconf"""
        logging.info("Starting IoT device discovery")
        # This would use zeroconf for device discovery
        # For now, just add some mock devices for demonstration
        self.devices = {
            "thermostat-living-room": {
                "ip": "192.168.1.100",
                "type": "thermostat",
                "manufacturer": "SmartHome",
                "last_seen": datetime.datetime.now().isoformat()
            },
            "camera-front-door": {
                "ip": "192.168.1.101",
                "type": "camera",
                "manufacturer": "SecurityCo",
                "last_seen": datetime.datetime.now().isoformat()
            },
            "light-kitchen": {
                "ip": "192.168.1.102",
                "type": "light",
                "manufacturer": "SmartLighting",
                "last_seen": datetime.datetime.now().isoformat()
            }
        }
        logging.info(f"Discovered {len(self.devices)} IoT devices")
    
    def stop_device_discovery(self) -> None:
        """Stop IoT device discovery service"""
        logging.info("Stopping device discovery")
        # In a real implementation, would clean up zeroconf
    
    def setup_traffic_filtering(self) -> None:
        """Set up traffic filtering rules to protect IoT devices"""
        logging.info("Setting up traffic filtering for IoT protection")
        # In a real implementation, would set up iptables rules
        # For now, just log the intention
        
    def stop_traffic_filtering(self) -> None:
        """Remove traffic filtering rules"""
        logging.info("Removing traffic filtering rules")
        # In a real implementation, would remove iptables rules
        
    def start_intrusion_detection(self) -> None:
        """Start intrusion detection system for IoT network"""
        logging.info("Starting intrusion detection for IoT network")
        # In a real implementation, would start IDS services
        
    def stop_intrusion_detection(self) -> None:
        """Stop intrusion detection system"""
        logging.info("Stopping intrusion detection")
        # In a real implementation, would stop IDS services


class IoTVPNClient:
    """Client implementation for IoT VPN"""
    
    def __init__(self, config_path: str = "client_config.json"):
        self.config_path = config_path
        self.running = False
        self.load_config()
        self.setup_logging()
        
    def load_config(self) -> None:
        """Load client configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                logging.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            logging.warning(f"Configuration file not found at {self.config_path}, using defaults")
            self.config = {
                "client_name": "iotvpn-client",
                "server_address": "localhost",
                "server_port": VPN_PORT,
                "cert_path": "./client_certs",
                "log_level": "INFO"
            }
            # Create default config
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                
    def setup_logging(self) -> None:
        """Configure logging"""
        log_level = getattr(logging, self.config.get("log_level", "INFO"))
        log_path = self.config.get("log_path", "iotvpn_client.log")
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        logging.info("Client logging initialized")
        
    def connect_to_server(self) -> bool:
        """Connect to the IoT VPN server"""
        logging.info(f"Connecting to server at {self.config['server_address']}:{self.config['server_port']}")
        
        # Set up SSL context
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.load_cert_chain(
            certfile=os.path.join(self.config["cert_path"], "client.crt"),
            keyfile=os.path.join(self.config["cert_path"], "client.key")
        )
        context.load_verify_locations(
            cafile=os.path.join(self.config["cert_path"], "ca.crt")
        )
        context.check_hostname = False  # Typically you'd enable this
        
        try:
            # Create socket and connect
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(10.0)  # 10 second timeout for connection
            
            # Wrap with SSL
            self.socket = context.wrap_socket(
                client_socket,
                server_side=False,
                server_hostname=self.config["server_address"]
            )
            
            # Connect to server
            self.socket.connect((self.config["server_address"], self.config["server_port"]))
            logging.info(f"Connected to {self.config['server_address']}:{self.config['server_port']}")
            
            # Receive VPN configuration
            data = self.socket.recv(4096)
            if data:
                vpn_config = json.loads(data.decode('utf-8'))
                logging.info(f"Received VPN configuration: {vpn_config}")
                
                # Set up VPN tunnel
                # In a real implementation, would set up actual tunneling
                self.running = True
                return True
            else:
                logging.error("No configuration received from server")
                self.socket.close()
                return False
                
        except Exception as e:
            logging.error(f"Error connecting to server: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the VPN server"""
        self.running = False
        try:
            self.socket.close()
            logging.info("Disconnected from VPN server")
        except Exception as e:
            logging.error(f"Error disconnecting: {e}")
    
    def start_vpn(self) -> None:
        """Start the VPN client and main loop"""
        if not self.connect_to_server():
            logging.error("Failed to connect to VPN server")
            return
            
        try:
            # Main communication loop
            buffer_size = 4096
            while self.running:
                try:
                    # Non-blocking receive with timeout
                    self.socket.settimeout(1.0)
                    data = self.socket.recv(buffer_size)
                    if not data:
                        # Server disconnected
                        logging.info("Server disconnected")
                        break
                    
                    # Process server data
                    # Here we would handle VPN tunnel data
                    logging.debug(f"Received {len(data)} bytes from server")
                    
                except socket.timeout:
                    # This is expected for non-blocking socket with timeout
                    continue
                except Exception as e:
                    logging.error(f"Error communicating with server: {e}")
                    break
                    
                time.sleep(0.1)  # Prevent CPU hogging
                
        except KeyboardInterrupt:
            logging.info("Client shutdown requested")
        finally:
            self.disconnect()


def main():
    """Main entry point for the IoT VPN system"""
    parser = argparse.ArgumentParser(description='IoT Network Embedded VPN System')
    parser.add_argument('--mode', choices=['server', 'client'], required=True,
                        help='Operation mode: server or client')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--generate-client', type=str, default=None,
                        help='Generate client certificate with the specified name (server mode only)')
    
    args = parser.parse_args()
    
    if args.mode == 'server':
        config_path = args.config or DEFAULT_CONFIG_PATH
        server = IoTVPNServer(config_path)
        
        if args.generate_client:
            # Generate client certificate
            client_key, client_cert, ca_cert = server.generate_client_certificate(args.generate_client)
            
            # Save to files
            client_cert_dir = f"client_certs_{args.generate_client}"
            os.makedirs(client_cert_dir, exist_ok=True)
            
            with open(os.path.join(client_cert_dir, "client.key"), 'wb') as f:
                f.write(client_key)
            
            with open(os.path.join(client_cert_dir, "client.crt"), 'wb') as f:
                f.write(client_cert)
                
            with open(os.path.join(client_cert_dir, "ca.crt"), 'wb') as f:
                f.write(ca_cert)
                
            print(f"Generated client certificates in {client_cert_dir}")
            print(f"Configuration for this client:")
            print(json.dumps({
                "client_name": args.generate_client,
                "server_address": "SERVER_IP_ADDRESS",
                "server_port": VPN_PORT,
                "cert_path": client_cert_dir,
                "log_level": "INFO"
            }, indent=4))
            
        else:
            # Start the server
            # Register signal handlers for clean shutdown
            def signal_handler(sig, frame):
                print("Shutdown signal received")
                server.stop_server()
                
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            server.start_server()
            
    elif args.mode == 'client':
        config_path = args.config or "client_config.json"
        client = IoTVPNClient(config_path)
        
        # Register signal handlers for clean shutdown
        def signal_handler(sig, frame):
            print("Shutdown signal received")
            client.disconnect()
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signalnal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    client.start_vpn()

if __name__ == "__main__":
    main()
