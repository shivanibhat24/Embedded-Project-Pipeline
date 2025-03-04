import umail
import network
import time
import machine
from machine import Pin

# Wi-Fi details
ssid = 'YOUR_WIFI_SSID'
password = 'YOUR_WIFI_PASSWORD'

# Email details
sender_email = 'YOUR_EMAIL'
sender_name = 'NAME_OF_THE EMAIL_ACCOUNT_HOLDER'
sender_app_password ='YOUR_APP_PASSWORD'  #found in the account settings
recipient_email = 'shivani.gbhat@gmail.com'
email_subject = 'Soil Moisture'

# Initialize the soil moisture sensor
sensor_pin = 34  # Change this to your sensor's pin number
soil_moisture_sensor = machine.ADC(sensor_pin)

#initialize the touch sensor
touch_pin= 32
touch_sensor=machine.ADC(touch_pin)
# Connect to Wi-Fi
station = network.WLAN(network.STA_IF)

station.active(True)
while not station.isconnected():
    time.sleep(1)

print('Connection successful')
print(station.ifconfig())

def read_soil_moisture():
    # Read the ADC value (0-4095 for ESP32)
    return soil_moisture_sensor.read()

def send_email(moisture_value):
    # Send the email with soil moisture data
    smtp = umail.SMTP('smtp.gmail.com', 465, ssl=True)
    smtp.login(sender_email, sender_app_password)
    smtp.to(recipient_email)
    smtp.write("From: {} <{}>\n".format(sender_name, sender_email))
    smtp.write("Subject: {}\n".format(email_subject))
    smtp.write("Current Moisture Level is {}\n".format(moisture_value))
    smtp.send()
    smtp.quit()

def read_touch_sensor():
    return touch_sensor.read()

# Main loop
while True:
    moisture_value = read_soil_moisture()
    print("Current soil moisture level: {}".format(moisture_value))
    send_email(moisture_value)
    x=touch_sensor.read()
    if x==1:
        break
    time.sleep(60)  
