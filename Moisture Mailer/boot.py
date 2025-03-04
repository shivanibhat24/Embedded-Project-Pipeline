import network
import time
import machine

ssid = 'Your_SSID'
password = 'Your_Password'

station = network.WLAN(network.STA_IF)

station.active(True)
station.connect(ssid, password)

while not station.isconnected():
    pass

print('Connection successful')
print(station.ifconfig())
